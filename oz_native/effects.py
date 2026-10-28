# Self-attention based effects-mask extraction (total_mask mode), replicating
# OmnimatteZero's SelfAttentionMapExtraction on top of comfy's native LTXV model.
#
# Deviation note: the official extractor recomputes q/k from the block input
# BEFORE RoPE; comfy's attention pipeline hands the override q/k AFTER RoPE.
# The captured attention maps therefore include positional rotation. The
# effects mask is an adaptive-threshold heuristic, validated qualitatively
# against the official total_mask.mp4 samples.
import torch
import torch.nn.functional as F


class AttentionCapture:
    """optimized_attention_override that captures per-layer self-attention maps."""

    def __init__(self, video_token_count):
        self.video_token_count = video_token_count
        self.maps = {}
        self.layer_idx = 0

    def __call__(self, func, q, k, v, heads, **kwargs):
        out = func(q, k, v, heads, **kwargs)
        # self-attention only: key length equals query length (cross-attn keys are text tokens)
        if q.ndim == 3 and k.shape[1] == q.shape[1] == self.video_token_count:
            b, t, hd = q.shape
            d = hd // heads
            qh = q.view(b, t, heads, d).transpose(1, 2).float()
            kh = k.view(b, t, heads, d).transpose(1, 2).float()
            probs = torch.softmax(qh @ kh.transpose(-2, -1) * (d ** -0.5), dim=-1)
            self.maps[self.layer_idx] = probs.mean(dim=1).detach().to("cpu")  # (B,T,T)
            self.layer_idx += 1
        return out

    def reset(self):
        self.maps = {}
        self.layer_idx = 0


def fix_num_frames_for_vae(num_frames, temporal_compression_ratio=8):
    k = (num_frames - 1) // temporal_compression_ratio
    valid = k * temporal_compression_ratio + 1
    return max(valid, temporal_compression_ratio + 1)


def extract_effects_mask(model, video_latent_fn, mask_binary, embeds, attn_mask,
                         extraction_timestep=0.5, threshold=None, dilation_size=3,
                         video_shape=None, seed=0):
    """Official extract_effects_mask port.

    video_latent_fn: callable() -> normalized video latents (1,128,f,h,w) in model dtype
    mask_binary: (1,1,T,H,W) float object mask (pixel space, same T as video)
    embeds: (1,128,4096) prompt embeds; attn_mask: (1,N) bool
    video_shape: (T,H,W) pixel dims
    Returns: effects mask tensor (1,1,T,H,W) binary float, on video device
    """
    device = comfy_device()
    dtype = model.model.diffusion_model.dtype

    latents = video_latent_fn().to(device=device, dtype=dtype)
    B, C, f_, h, w = latents.shape
    T, H, W = video_shape

    gen = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(latents.shape, generator=gen, dtype=torch.float32).to(latents.dtype)
    sigma = float(extraction_timestep)
    noisy = (1.0 - sigma) * latents + sigma * noise

    capture = AttentionCapture(f_ * h * w)
    to = {"optimized_attention_override": capture}
    with torch.no_grad():
        model.model.apply_model(
            noisy, torch.tensor([sigma], device=device),
            c_crossattn=embeds.to(device=device, dtype=dtype),
            attention_mask=attn_mask.to(device),
            transformer_options=to,
        )
    if len(capture.maps) == 0:
        raise RuntimeError("no self-attention maps captured")
    print(f"[OmnimatteZero] captured {len(capture.maps)} self-attention layers")

    spatial = h * w
    # object mask downsampled per latent frame (official logic)
    mask_latent_frames = []
    for t in range(f_):
        pixel_t = min(t * 8, T - 1)
        mf = mask_binary[:, :, pixel_t, :, :]
        mask_latent_frames.append(F.interpolate(mf, size=(h, w), mode="nearest"))
    mask_latent_all = torch.stack(mask_latent_frames, dim=2).to(device=device, dtype=dtype)  # (1,1,f,h,w)

    per_frame_effects = []
    for frame_t in range(f_):
        frame_effects_sum = torch.zeros(B, spatial, device=device, dtype=dtype)
        for layer_idx, attn in capture.maps.items():
            attn = attn.to(device).to(dtype)
            if attn.shape[-1] != f_ * spatial:
                continue
            attn_reshaped = attn.view(B, f_, spatial, f_, spatial)
            attn_from_frame_t = attn_reshaped[:, frame_t]
            for src_t in range(f_):
                src_mask = mask_latent_all[:, 0, src_t, :, :].reshape(B, -1)
                attn_to_obj = (attn_from_frame_t[:, :, src_t, :] * src_mask.unsqueeze(1)).sum(dim=-1)
                frame_effects_sum += attn_to_obj
        frame_effects = frame_effects_sum / (len(capture.maps) + 1e-8)
        per_frame_effects.append(frame_effects.view(B, h, w))

    effects_latent = torch.stack(per_frame_effects, dim=1)  # (B,f,h,w)

    effects_spatial = F.interpolate(
        effects_latent.reshape(B * f_, 1, h, w), size=(H, W), mode="bilinear", align_corners=False
    ).view(B, f_, H, W)
    effects_temporal = F.interpolate(
        effects_spatial.unsqueeze(1), size=(T, H, W), mode="trilinear", align_corners=False
    ).squeeze(1)
    effects_mask = effects_temporal.unsqueeze(1)  # (B,1,T,H,W)

    if threshold is None:
        flat = effects_mask.float().view(-1)
        threshold = float(flat.mean() + 0.5 * flat.std())
        print(f"[OmnimatteZero] adaptive threshold: {threshold:.4f}")

    effects_binary = (effects_mask.float() > threshold).float()
    if dilation_size > 0:
        kernel = torch.ones(1, 1, dilation_size, dilation_size, device=device)
        frames = []
        for t in range(T):
            fr = effects_binary[:, :, t, :, :]
            fr = F.conv2d(fr, kernel, padding=dilation_size // 2)
            frames.append((fr > 0).float())
        effects_binary = torch.stack(frames, dim=2)
    return effects_binary


def comfy_device():
    import comfy.model_management as mm
    return mm.get_torch_device()
