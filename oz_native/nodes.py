# OmnimatteZero comfy-native nodes (no diffusers dependency).
# Removal / total-mask / foreground-compose on comfy's native LTXV stack.
import os
import torch
import folder_paths
import comfy.utils
import comfy.model_management as mm

from . import core
from .ltx_vae import OZLTXVAE

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAPPER_DIR = os.path.join(PLUGIN_DIR, "wrapper")
FRAME_RATE = 25  # LTX default; scales the temporal RoPE coordinate like the official pipeline


def _upscale_video(images, width, height):
    pixels = comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center")
    return pixels.movedim(1, -1)[:, :, :, :3]


def _to_uint8(x):
    """comfy IMAGE -> legacy wrapper's PIL round trip (uint8 quantisation included).

    The original node did `tensor.mul(255).clamp(0,255).byte()` -> PIL -> diffusers
    VAEImageProcessor (x/255, then (v-0.5)/0.5). Skipping the quantisation shifts the VAE
    input by up to 1/255 everywhere, which is enough to break bit-level parity.
    """
    return (x * 255.0).clamp(0.0, 255.0).to(torch.uint8).to(torch.float32) / 255.0


def _mask_to_pixels(mask, num_frames, width, height):
    """comfy MASK -> official mask video in {-1,1} -> {0,100} (3 channels)."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1:
        mask = mask.repeat(num_frames, 1, 1)
    mask = mask[:num_frames]
    mask = comfy.utils.common_upscale(mask.unsqueeze(1), width, height, "bilinear", "center").squeeze(1)
    m = _to_uint8(mask)                       # 0/1 after the legacy round trip
    m = m * 2.0 - 1.0                         # [-1,1]
    # official: mask[mask < 0] = 0 ; mask[mask > 0] = 100   (on the normalised tensor)
    m = torch.where(m < 0, torch.zeros_like(m), torch.full_like(m, 100.0))
    return m.unsqueeze(-1).repeat(1, 1, 1, 3)


def _default_bias(key):
    masks = torch.load(os.path.join(WRAPPER_DIR, "attn_masks.pt"), map_location="cpu", weights_only=False)
    return _as_bias(masks[key])


def _as_bias(mask):
    """Normalize a text attention mask into the additive bias the DiT expects.

    diffusers uses `(1 - mask) * -10000.0`; comfy's LTXV only converts *bool* masks, and
    CONDITIONING tensors get upcast to bf16 before reaching it, so the bias is built here.
    """
    m = mask.detach().float()
    if m.ndim == 1:
        m = m.unsqueeze(0)
    if m.dim() == 2:
        if m.min().item() < -0.5:                      # already a bias (B, N)
            return m[:, None, None, :]
        return ((1.0 - m) * -10000.0)[:, None, None, :]
    if m.dim() == 4:
        return m
    if m.dim() == 3:
        return m.unsqueeze(1)
    raise ValueError(f"unsupported attention mask shape {tuple(mask.shape)}")


def _cond_pair(positive, negative):
    pos = positive[0]
    neg = negative[0]
    pm = pos[1].get("attention_mask", None)
    nm = neg[1].get("attention_mask", None)
    pm = _default_bias("prompt") if pm is None else _as_bias(pm)
    nm = _default_bias("negative") if nm is None else _as_bias(nm)
    return pos[0], pm, neg[0], nm


def _ensure_causal_fix(model):
    """Enable comfy's causal temporal positioning on an LTXV diffusion model.

    diffusers' `_scale_video_ids` ALWAYS applies `pixel_t = 8k + 1 - 8` (clamped), while
    comfy's LTXVModel defaults `causal_temporal_positioning=False`, which leaves the
    temporal RoPE of every frame after the first shifted by 7px. The flag is only read in
    `_process_input`, so flipping it here is plugin-side configuration, not a comfy patch.

    Lives in the sampler (instead of the loader) so the MODEL can come from comfy's own
    `UNETLoader` / gguf / any other model source.
    """
    base = getattr(model, "model", model)
    diffusion_model = getattr(base, "diffusion_model", None)
    if diffusion_model is not None and hasattr(diffusion_model, "causal_temporal_positioning"):
        diffusion_model.causal_temporal_positioning = True


class OmnimatteZeroVAELoader:
    """LTX-Video VAE loader for the native nodes (VAE only - use comfy's UNETLoader for the model).

    comfy's own `VAELoader` cannot be used here: the official pipeline needs the VAE
    posterior parameters (mean + logvar -> the sampler replays the official posterior
    *sampling* draws), the RAW un-normalized latent for the object-mask threshold, and the
    diffusers tiled encoder (tile 512 / stride 448) that `enable_tiling()` triggers at
    768x480. comfy's `VAE.encode()` exposes none of those - it returns the normalized mean
    of the untiled encoder. Hence this thin loader keeps the diffusers-exact implementation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": (folder_paths.get_filename_list("vae"),)}}

    RETURN_TYPES = ("OZ_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "OmnimatteZero"

    def load(self, vae):
        return (_load_oz_vae(folder_paths.get_full_path("vae", vae)),)


def _load_oz_vae(vae_path):
    # diffusers-0.37-exact VAE port; config comes bundled with the plugin
    config_path = os.path.join(PLUGIN_DIR, "LTX-Video-0.9.7-diffusers", "vae", "config.json")
    try:
        return OZLTXVAE(vae_path, config_path, dtype=torch.bfloat16)
    except Exception as exc:  # noqa: BLE001 - surface a usable message instead of a key dump
        raise RuntimeError(
            f"无法按 LTX-Video VAE 加载 {os.path.basename(vae_path)}: {exc}\n"
            "OmnimatteZero 需要 **diffusers 格式** 的 LTX-Video VAE"
            "（键名形如 encoder./decoder./per_channel_statistics，例如 "
            "LTX-Video-0.9.7-vae-diffusers.safetensors），"
            "不是 comfy 的 first_stage_model 格式。"
        ) from exc




class OmnimatteZeroRemoveConditioning:
    """Encodes the input video into normalized latents and keeps the VAE posterior
    parameters of both video and mask, so the sampler can replay the official RNG
    sequence (init noise -> video posterior -> mask posterior -> per-step noise)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("OZ_VAE",),
                "images": ("IMAGE",),
                "object_mask": ("MASK",),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 32}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "OmnimatteZero"

    def encode(self, vae, images, object_mask, width, height):
        num_frames = images.shape[0]
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
            images = images[:num_frames]
        pixels = _upscale_video(images, width, height)                 # (T,H,W,3) 0-1
        px = _to_uint8(pixels) * 2.0 - 1.0                             # official [-1,1]

        mask_px = _mask_to_pixels(object_mask, num_frames, width, height)  # (T,H,W,3) {0,100}

        h_video = vae.encode_h(px.permute(3, 0, 1, 2).unsqueeze(0))    # (1,256,f,h,w) bf16 cpu
        h_mask = vae.encode_h(mask_px.permute(3, 0, 1, 2).unsqueeze(0))

        # mean-only latent for preview / fallback (the sampler samples the posterior)
        mean = h_video[:, :h_video.shape[1] // 2].float()
        lat_mean, lat_std = vae.latent_stats()
        z = (mean - lat_mean) / lat_std
        mask_mean = h_mask[:, :h_mask.shape[1] // 2].float()[0].mean(0)
        cms = ((mask_mean > 0.01) | (mask_mean < -0.01)).float()[None]

        return ({"samples": z, "noise_mask": cms,
                 "oz_h_video": h_video, "oz_h_mask": h_mask,
                 "oz_lat_mean": lat_mean, "oz_lat_std": lat_std},)


class OmnimatteZeroSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "mode": (["removal", "compose"],),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cond_noise_scale": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "OmnimatteZero"

    def sample(self, model, latent, mode, steps, cfg, seed, denoise, cond_noise_scale):
        device = mm.get_torch_device()
        timesteps, sigmas = core.build_schedule(steps)
        pos = torch.load(os.path.join(WRAPPER_DIR, "positive_c.pt" if mode == "compose" else "positive.pt"),
                                 map_location="cpu", weights_only=False).to(torch.bfloat16)
        neg = torch.load(os.path.join(WRAPPER_DIR, "negative_c.pt" if mode == "compose" else "negative.pt"),
                            map_location="cpu", weights_only=False).to(torch.bfloat16)
        pm = _default_bias("prompt_c" if mode == "compose" else "prompt")
        nm = _default_bias("negative")
        positive = [[pos, {"attention_mask": pm}]]
        negative = [[neg, {"attention_mask": nm}]]
        pos_emb, pos_mask, neg_emb, neg_mask = _cond_pair(positive, negative)
        ctx2 = torch.cat([neg_emb.to(device, torch.bfloat16), pos_emb.to(device, torch.bfloat16)], dim=0)
        mask2 = torch.cat([neg_mask, pos_mask], dim=0).to(device, torch.bfloat16)

        z = latent["samples"].to(device, torch.float32)
        gen = torch.Generator().manual_seed(int(seed) % (2 ** 64))
        pbar = comfy.utils.ProgressBar(steps)

        _ensure_causal_fix(model)  # works for MODEL from comfy's UNETLoader as well
        mm.load_models_gpu([model])
        base_model = model.model
        try:
            if mode == "compose":
                num = min(int(steps * denoise), steps)
                start = max(steps - num, 0)
                out = core.compose_loop(base_model, z, ctx2, mask2, sigmas, start, cfg, gen,
                                        transformer_options=model.model_options.get("transformer_options", {}),
                                        frame_rate=FRAME_RATE, pbar=pbar)
            else:
                noise0 = core.randn_bchw(z.shape, gen, device)             # RNG draw #1
                z, cms = self._conditioning(latent, gen, device, z)
                latents = torch.where(cms > 0.5, noise0, z)
                out = core.removal_loop(base_model, latents, z, cms, ctx2, mask2, sigmas, cfg, gen,
                                        cond_noise_scale=cond_noise_scale,
                                        transformer_options=model.model_options.get("transformer_options", {}),
                                        frame_rate=FRAME_RATE, pbar=pbar)
        finally:
            mm.unload_all_models()
            mm.soft_empty_cache()

        return ({"samples": out.float().cpu()},)

    @staticmethod
    def _conditioning(latent, gen, device, z_fallback):
        """Replay the official VAE posterior sampling (RNG draws #2 and #3)."""
        h_v = latent.get("oz_h_video", None)
        h_m = latent.get("oz_h_mask", None)
        if h_v is None or h_m is None:
            cms = latent.get("noise_mask", None)
            if cms is None:
                cms = torch.ones_like(z_fallback[:, :1])
            return z_fallback, cms.to(device, torch.float32)

        h_v = h_v.to(device)
        h_m = h_m.to(device)
        lat_mean = latent["oz_lat_mean"].to(device)
        lat_std = latent["oz_lat_std"].to(device)

        samp_v = OZLTXVAE.posterior_sample(h_v, gen)                   # draw #2
        samp_m = OZLTXVAE.posterior_sample(h_m, gen)                   # draw #3
        z = ((samp_v - lat_mean.to(torch.bfloat16)) / lat_std.to(torch.bfloat16)).float()
        mean_ch = samp_m[0].mean(0)
        cms = ((mean_ch > 0.01) | (mean_ch < -0.01)).float()[None, None]
        return z, cms


class OmnimatteZeroComposeLatent:
    """Official foreground compositing: latent-space foreground/background
    composition, decode, pixel blend with the original foreground, re-encode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("OZ_VAE",),
                "foreground": ("IMAGE",),
                "video_bg": ("IMAGE",),
                "video_new_bg": ("IMAGE",),
                "object_mask": ("MASK",),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 32}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compose"
    CATEGORY = "OmnimatteZero"

    def compose(self, vae, foreground, video_bg, video_new_bg, object_mask, width, height):
        n = min(foreground.shape[0], video_bg.shape[0], video_new_bg.shape[0])
        fg = _upscale_video(foreground[:n], width, height)
        bg = _upscale_video(video_bg[:n], width, height)
        nbg = _upscale_video(video_new_bg[:n], width, height)

        z_all = vae.encode(fg)
        z_bg = vae.encode(bg)
        z_new_bg = vae.encode(nbg)

        mask = object_mask if object_mask.ndim == 3 else object_mask.unsqueeze(0)
        if mask.shape[0] == 1:
            mask = mask.repeat(n, 1, 1)
        mask = mask[:n]
        mask = comfy.utils.common_upscale(mask.unsqueeze(1), width, height, "bilinear", "center").squeeze(1)
        m_bin = (mask > 0.5).to(z_all.dtype)                       # (T,H,W) 0/1

        # official: z_mask comes from the binary 0/1 mask video in the [-1,1] domain
        mask_pixels = (m_bin * 2.0 - 1.0).unsqueeze(-1).repeat(1, 1, 1, 3)
        z_mask = vae.encode(mask_pixels)
        z_mask_smooth = torch.sigmoid(z_mask * 5.0)
        z_fg = (z_all - z_bg) * z_mask_smooth
        z_result = z_new_bg + z_fg

        comp = vae.decode(z_result, decode_timestep=0.0)   # official: temb=0 in the compose VAE forward
        blend = m_bin.to(comp.dtype).unsqueeze(-1)
        blended = comp * (1.0 - blend) + fg.to(comp.dtype) * blend

        condition_latents = vae.encode(blended)
        return ({"samples": condition_latents},)


class OmnimatteZeroDecode:
    """VAE decode with official decode_timestep / decode_noise_scale semantics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("OZ_VAE",),
                "latent": ("LATENT",),
                "decode_timestep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "decode_noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "OmnimatteZero"

    def decode(self, vae, latent, decode_timestep, decode_noise_scale, seed):
        z = latent["samples"].float()
        if decode_noise_scale > 0:
            gen = torch.Generator().manual_seed(int(seed) % (2 ** 64))
            noise = core.randn_bchw(z.shape, gen, z.device, dtype=torch.bfloat16)
            z = (1.0 - decode_noise_scale) * z + decode_noise_scale * noise.float()
        img = vae.decode(z, decode_timestep=decode_timestep)  # (T,H,W,C) 0-1
        return (img.float().cpu(),)


NODE_CLASS_MAPPINGS = {
    "OmnimatteZeroVAELoader": OmnimatteZeroVAELoader,
    "OmnimatteZeroRemoveConditioning": OmnimatteZeroRemoveConditioning,
    "OmnimatteZeroSampler": OmnimatteZeroSampler,
    "OmnimatteZeroComposeLatent": OmnimatteZeroComposeLatent,
    "OmnimatteZeroDecode": OmnimatteZeroDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmnimatteZeroVAELoader": "OmnimatteZero VAE Loader (LTX-Video)",
    "OmnimatteZeroRemoveConditioning": "OmnimatteZero Video+Mask Conditioning",
    "OmnimatteZeroSampler": "OmnimatteZero Sampler",
    "OmnimatteZeroComposeLatent": "OmnimatteZero Compose Latent",
    "OmnimatteZeroDecode": "OmnimatteZero VAE Decode",
}
