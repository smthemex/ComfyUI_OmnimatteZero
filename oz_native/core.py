# OmnimatteZero comfy-native core.
#
# Replicates the diffusers-0.37 LTXConditionPipeline semantics used by the original
# OmnimatteZero wrapper EXACTLY (verified against an instrumented reference run):
#
#   1. RNG draw order (single torch.Generator seeded once):
#        #1  randn((1,128,f,h,w), fp32)   BCHW   -> initial noise for the object region
#        #2  randn((1,128,f,h,w), bf16)   BCHW   -> VAE posterior sample of the video
#        #3  randn((1,128,f,h,w), bf16)   BCHW   -> VAE posterior sample of the mask
#        #4+ randn((1,T,128), fp32)       TOKEN  -> per-step conditioning re-noise
#      Draw #4 is token-major (diffusers packs latents to (B,T,C) before the loop), a
#      *different* element order than BCHW: drawing in the wrong layout silently changes
#      every noise value while still looking plausible.
#   2. Per-token timesteps: background tokens are evaluated at t=0 (frozen context),
#      object tokens at t. comfy does this natively through `denoise_mask` +
#      LTXV.process_timestep (t * mask), so an int64 mask is passed through apply_model.
#   3. Hard-conditioning re-noise: the background latents fed to the model are
#      `clean_video_latent + cond_noise_scale * t^2 * noise`, refreshed every step.
#   4. fp32 latents, bf16 model input, bf16 CFG arithmetic.
#   5. Euler update `x <- x + (sigma_next - sigma) * v` on object tokens only.
import contextlib

import torch

import comfy.model_management as mm


def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    """Verbatim port of diffusers.pipelines.ltx.pipeline_ltx_condition.linear_quadratic_schedule."""
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
    const = quadratic_coef * (linear_steps ** 2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1], dtype=torch.float32)


def build_schedule(steps):
    """Official timestep / sigma tables.

    diffusers: `sigmas = linear_quadratic_schedule(n)`, `timesteps = sigmas * 1000`, then
    `scheduler.set_timesteps(timesteps=...)` stores `sigmas = timesteps / 1000` (shift=1.0)
    and appends a terminal 0. The round-tripped sigma matters: the wrapper feeds `t/1000`
    (from the timestep table) into the conditioning re-noise, not the raw sigma.
    """
    sig = linear_quadratic_schedule(steps)
    timesteps = sig * 1000.0
    sigmas = timesteps / 1000.0
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return timesteps, sigmas


# --------------------------------------------------------------------------- RNG
def randn_bchw(shape, gen, device, dtype=torch.float32):
    """diffusers randn_tensor with a CPU generator: draw on CPU, then move."""
    return torch.randn(tuple(shape), generator=gen, dtype=dtype, device="cpu").to(device)


def randn_tokens(shape_bchw, gen, device, dtype=torch.float32):
    """Per-step noise in diffusers' packed (B, T, C) token-major order -> BCHW."""
    b, c, f, h, w = shape_bchw
    n = torch.randn((b, f * h * w, c), generator=gen, dtype=dtype, device="cpu")
    return n.view(b, f, h, w, c).permute(0, 4, 1, 2, 3).to(device)


# ------------------------------------------------------------------ model driving
@contextlib.contextmanager
def raw_velocity(model_sampling):
    """Make `apply_model` return the raw velocity instead of x0.

    comfy's BaseModel.apply_model ends with `calculate_denoised`; recovering v from x0
    afterwards (`(x - x0)/sigma`) destroys precision at small sigma, so shortcut it.
    """
    orig = model_sampling.calculate_denoised
    model_sampling.calculate_denoised = lambda sigma, model_output, model_input: model_output
    try:
        yield
    finally:
        model_sampling.calculate_denoised = orig


def predict_velocity(base_model, latents, sigma, ctx2, mask2, dm2,
                     transformer_options=None, frame_rate=25):
    """CFG-batched velocity prediction (batch = [negative, positive], like the official run)."""
    to = {} if transformer_options is None else transformer_options
    x = latents.to(device=mm.get_torch_device(), dtype=torch.bfloat16)
    x2 = torch.cat([x, x], dim=0)
    sig = torch.full((2,), float(sigma), device=x.device, dtype=torch.float32)
    kwargs = dict(c_crossattn=ctx2, transformer_options=to, frame_rate=int(frame_rate))
    if mask2 is not None:
        kwargs["attention_mask"] = mask2
    if dm2 is not None:
        # int64 keeps comfy from upcasting the per-token timestep to bf16: convert_tensor
        # only casts non-integer dtypes, so int64 * float32 stays float32 inside
        # LTXV.process_timestep (matching the official fp32 timestep table).
        kwargs["denoise_mask"] = dm2
    with raw_velocity(base_model.model_sampling):
        out = base_model.apply_model(x2, sig, **kwargs)
    return out  # (2, C, f, h, w) fp32 velocity


def cfg_combine(v, cfg):
    """CFG in bf16, exactly like `noise_pred_uncond + scale * (text - uncond)`."""
    vb = v.to(torch.bfloat16)
    if vb.shape[0] == 1:
        return vb[0].float()
    return (vb[0] + cfg * (vb[1] - vb[0])).float()


# ------------------------------------------------------------------- denoise loops
def removal_loop(base_model, latents, z, cms, ctx2, mask2, sigmas, cfg, gen,
                 cond_noise_scale=0.15, transformer_options=None, frame_rate=25, pbar=None):
    """Official OmnimatteZero removal loop.

    z       : (1,128,f,h,w) fp32 normalized video latents (hard conditioning target)
    cms     : (1,1,f,h,w) float, 1 = object (regenerate), 0 = background (frozen)
    latents : (1,128,f,h,w) fp32 starting state (background=z, object=noise #1)
    """
    obj = cms > 0.5
    bg = ~obj
    dm = obj.to(torch.long)
    dm2 = torch.cat([dm, dm], dim=0)
    n = int(sigmas.shape[0]) - 1

    for i in range(n):
        t_sig = float(sigmas[i])
        t_next = float(sigmas[i + 1])
        noise = randn_tokens(z.shape, gen, z.device)                       # draw #4+
        latents = torch.where(bg, z + cond_noise_scale * noise * (t_sig ** 2), latents)

        v = cfg_combine(predict_velocity(base_model, latents, t_sig, ctx2, mask2, dm2,
                                         transformer_options, frame_rate), cfg)
        den = latents - (t_sig - t_next) * v                               # sample + dt * (-v)
        latents = torch.where(obj, den, latents)

        if pbar is not None:
            pbar.update(1)
        mm.throw_exception_if_processing_interrupted()
    return latents


def compose_loop(base_model, z, ctx2, mask2, sigmas, start, cfg, gen,
                 transformer_options=None, frame_rate=25, pbar=None):
    """Official foreground-compose refinement: plain flow-Euler over the schedule tail."""
    sig = sigmas[start:]
    n = len(sig) - 1
    noise = randn_bchw(z.shape, gen, z.device)
    latents = sig[0] * noise + (1.0 - sig[0]) * z
    for i in range(n):
        t_sig = float(sig[i])
        t_next = float(sig[i + 1])
        v = cfg_combine(predict_velocity(base_model, latents, t_sig, ctx2, mask2, None,
                                         transformer_options, frame_rate), cfg)
        latents = latents + (t_next - t_sig) * v
        if pbar is not None:
            pbar.update(1)
        mm.throw_exception_if_processing_interrupted()
    return latents
