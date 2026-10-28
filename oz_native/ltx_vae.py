# Self-contained torch port of diffusers 0.37's AutoencoderKLLTXVideo
# (the reference implementation for LTX-Video 0.9.7 VAE semantics).
# Weight names are identical to the diffusers checkpoint, so the original
# LTX-Video-0.9.7-vae-diffusers.safetensors loads with strict=True.
# No diffusers import anywhere.
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    if name in ("swish", "silu"):
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(name)


def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, scale=1, max_period=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift, scale=1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        return get_timestep_embedding(timesteps, self.num_channels, self.flip_sin_to_cos, self.downscale_freq_shift, self.scale)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, out_dim=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim, bias=True)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """Only the timestep path is needed (resolution/aspect ratio unused by LTX VAE)."""

    def __init__(self, embedding_dim, size_emb_dim=0):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, resolution=None, aspect_ratio=None, batch_size=None, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(hidden_dtype))
        return timesteps_emb


class LTXVideoCausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 padding_mode="zeros", is_causal=True):
        super().__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride,) * 3
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size, stride=stride,
                              dilation=dilation, groups=groups, padding=(0, height_pad, width_pad),
                              padding_mode=padding_mode)

    def forward(self, hidden_states):
        time_kernel_size = self.kernel_size[0]
        if self.is_causal:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
            hidden_states = torch.cat([pad_left, hidden_states], dim=2)
        else:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)
        return self.conv(hidden_states)


class LTXVideoResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, eps=1e-6, elementwise_affine=False,
                 non_linearity="swish", is_causal=True, inject_noise=False, timestep_conditioning=False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.nonlinearity = get_activation(non_linearity)
        self.norm1 = nn.RMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.conv1 = LTXVideoCausalConv3d(in_channels, out_channels, kernel_size=3, is_causal=is_causal)
        self.norm2 = nn.RMSNorm(out_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LTXVideoCausalConv3d(out_channels, out_channels, kernel_size=3, is_causal=is_causal)

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = LTXVideoCausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, is_causal=is_causal)

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels ** 0.5)

    def forward(self, inputs, temb=None, generator=None):
        hidden_states = self.norm1(inputs.movedim(1, -1)).movedim(-1, 1)
        if self.scale_shift_table is not None:
            t = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = t.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if self.per_channel_scale1 is not None:
            noise = torch.randn(hidden_states.shape[-2:], generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (noise * self.per_channel_scale1)[None, :, None, ...]
        hidden_states = self.norm2(hidden_states.movedim(1, -1)).movedim(-1, 1)
        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.per_channel_scale2 is not None:
            noise = torch.randn(hidden_states.shape[-2:], generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (noise * self.per_channel_scale2)[None, :, None, ...]
        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)
        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)
        return hidden_states + inputs


class LTXVideoDownsampler3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, is_causal=True, padding_mode="zeros"):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.group_size = (in_channels * self.stride[0] * self.stride[1] * self.stride[2]) // out_channels
        out_ch = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])
        self.conv = LTXVideoCausalConv3d(in_channels, out_ch, kernel_size=3, stride=1, is_causal=is_causal, padding_mode=padding_mode)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)
        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size))
        residual = residual.mean(dim=2)
        hidden_states = self.conv(hidden_states)
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        return hidden_states + residual


class LTXVideoUpsampler3d(nn.Module):
    def __init__(self, in_channels, stride=1, is_causal=True, residual=False, upscale_factor=1, padding_mode="zeros"):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.residual = residual
        self.upscale_factor = upscale_factor
        out_channels = (in_channels * self.stride[0] * self.stride[1] * self.stride[2]) // upscale_factor
        self.conv = LTXVideoCausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, is_causal=is_causal, padding_mode=padding_mode)

    def forward(self, hidden_states):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        if self.residual:
            residual = hidden_states.reshape(
                batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
            )
            residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
            repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
            residual = residual.repeat(1, repeats, 1, 1, 1)
            residual = residual[:, :, self.stride[0] - 1:]
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        hidden_states = hidden_states[:, :, self.stride[0] - 1:]
        if self.residual:
            hidden_states = hidden_states + residual
        return hidden_states


class LTXVideoDownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_layers=1, dropout=0.0, resnet_eps=1e-6,
                 resnet_act_fn="swish", spatio_temporal_scale=True, is_causal=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.resnets = nn.ModuleList([
            LTXVideoResnetBlock3d(in_channels, in_channels, dropout, resnet_eps, non_linearity=resnet_act_fn, is_causal=is_causal)
            for _ in range(num_layers)
        ])
        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList([
                LTXVideoCausalConv3d(in_channels, in_channels, kernel_size=3, stride=(2, 2, 2), is_causal=is_causal)
            ])
        self.conv_out = None
        if in_channels != out_channels:
            self.conv_out = LTXVideoResnetBlock3d(in_channels, out_channels, dropout, resnet_eps, non_linearity=resnet_act_fn, is_causal=is_causal)

    def forward(self, hidden_states, temb=None, generator=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                hidden_states = d(hidden_states)
        if self.conv_out is not None:
            hidden_states = self.conv_out(hidden_states, temb, generator)
        return hidden_states


class LTXVideo095DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_layers=1, dropout=0.0, resnet_eps=1e-6,
                 resnet_act_fn="swish", spatio_temporal_scale=True, is_causal=True, downsample_type="conv"):
        super().__init__()
        out_channels = out_channels or in_channels
        self.resnets = nn.ModuleList([
            LTXVideoResnetBlock3d(in_channels, in_channels, dropout, resnet_eps, non_linearity=resnet_act_fn, is_causal=is_causal)
            for _ in range(num_layers)
        ])
        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()
            if downsample_type == "conv":
                self.downsamplers.append(LTXVideoCausalConv3d(in_channels, in_channels, kernel_size=3, stride=(2, 2, 2), is_causal=is_causal))
            elif downsample_type == "spatial":
                self.downsamplers.append(LTXVideoDownsampler3d(in_channels, out_channels, stride=(1, 2, 2), is_causal=is_causal))
            elif downsample_type == "temporal":
                self.downsamplers.append(LTXVideoDownsampler3d(in_channels, out_channels, stride=(2, 1, 1), is_causal=is_causal))
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(LTXVideoDownsampler3d(in_channels, out_channels, stride=(2, 2, 2), is_causal=is_causal))
            else:
                raise ValueError(downsample_type)

    def forward(self, hidden_states, temb=None, generator=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                hidden_states = d(hidden_states)
        return hidden_states


class LTXVideoMidBlock3d(nn.Module):
    def __init__(self, in_channels, num_layers=1, dropout=0.0, resnet_eps=1e-6, resnet_act_fn="swish",
                 is_causal=True, inject_noise=False, timestep_conditioning=False):
        super().__init__()
        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)
        self.resnets = nn.ModuleList([
            LTXVideoResnetBlock3d(in_channels, in_channels, dropout, resnet_eps, non_linearity=resnet_act_fn,
                                  is_causal=is_causal, inject_noise=inject_noise, timestep_conditioning=timestep_conditioning)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, temb=None, generator=None):
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(), resolution=None, aspect_ratio=None,
                batch_size=hidden_states.size(0), hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)
        return hidden_states


class LTXVideoUpBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_layers=1, dropout=0.0, resnet_eps=1e-6,
                 resnet_act_fn="swish", spatio_temporal_scale=True, is_causal=True, inject_noise=False,
                 timestep_conditioning=False, upsample_residual=False, upscale_factor=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)
        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = LTXVideoResnetBlock3d(in_channels, out_channels, dropout, resnet_eps,
                                                 non_linearity=resnet_act_fn, is_causal=is_causal,
                                                 inject_noise=inject_noise, timestep_conditioning=timestep_conditioning)
        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList([
                LTXVideoUpsampler3d(out_channels * upscale_factor, stride=(2, 2, 2), is_causal=is_causal,
                                    residual=upsample_residual, upscale_factor=upscale_factor)
            ])
        self.resnets = nn.ModuleList([
            LTXVideoResnetBlock3d(out_channels, out_channels, dropout, resnet_eps, non_linearity=resnet_act_fn,
                                  is_causal=is_causal, inject_noise=inject_noise, timestep_conditioning=timestep_conditioning)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, temb=None, generator=None):
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator)
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(), resolution=None, aspect_ratio=None,
                batch_size=hidden_states.size(0), hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
        if self.upsamplers is not None:
            for u in self.upsamplers:
                hidden_states = u(hidden_states)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)
        return hidden_states


class LTXVideoEncoder3d(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, block_out_channels=(128, 256, 512, 512),
                 down_block_types=("LTXVideoDownBlock3D",) * 4, spatio_temporal_scaling=(True,) * 4,
                 layers_per_block=(4, 3, 3, 3, 4), downsample_type=("conv",) * 4, patch_size=4,
                 patch_size_t=1, resnet_norm_eps=1e-6, is_causal=True):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size ** 2
        output_channel = block_out_channels[0]
        self.conv_in = LTXVideoCausalConv3d(self.in_channels, output_channel, kernel_size=3, stride=1, is_causal=is_causal)

        is_ltx_095 = down_block_types[-1] == "LTXVideo095DownBlock3D"
        num_block_out_channels = len(block_out_channels) - (1 if is_ltx_095 else 0)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            if not is_ltx_095:
                output_channel = block_out_channels[i + 1] if i + 1 < num_block_out_channels else block_out_channels[i]
            else:
                output_channel = block_out_channels[i + 1]
            if down_block_types[i] == "LTXVideoDownBlock3D":
                down_block = LTXVideoDownBlock3D(input_channel, output_channel, num_layers=layers_per_block[i],
                                                 resnet_eps=resnet_norm_eps, spatio_temporal_scale=spatio_temporal_scaling[i], is_causal=is_causal)
            elif down_block_types[i] == "LTXVideo095DownBlock3D":
                down_block = LTXVideo095DownBlock3D(input_channel, output_channel, num_layers=layers_per_block[i],
                                                    resnet_eps=resnet_norm_eps, spatio_temporal_scale=spatio_temporal_scaling[i],
                                                    is_causal=is_causal, downsample_type=downsample_type[i])
            else:
                raise ValueError(down_block_types[i])
            self.down_blocks.append(down_block)

        self.mid_block = LTXVideoMidBlock3d(output_channel, num_layers=layers_per_block[-1],
                                            resnet_eps=resnet_norm_eps, is_causal=is_causal)
        self.norm_out = nn.RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
        self.conv_act = nn.SiLU()
        self.conv_out = LTXVideoCausalConv3d(output_channel, out_channels + 1, kernel_size=3, stride=1, is_causal=is_causal)

    def forward(self, hidden_states):
        p = self.patch_size
        p_t = self.patch_size_t
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        hidden_states = hidden_states.reshape(
            batch_size, num_channels, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
        )
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)
        return hidden_states


class LTXVideoDecoder3d(nn.Module):
    def __init__(self, in_channels=128, out_channels=3, block_out_channels=(128, 256, 512, 512),
                 spatio_temporal_scaling=(True, True, True, False), layers_per_block=(4, 3, 3, 3, 4),
                 patch_size=4, patch_size_t=1, resnet_norm_eps=1e-6, is_causal=False,
                 inject_noise=(False,) * 4, timestep_conditioning=False,
                 upsample_residual=(False,) * 4, upsample_factor=(1, 1, 1, 1)):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size ** 2
        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        output_channel = block_out_channels[0]

        self.conv_in = LTXVideoCausalConv3d(in_channels, output_channel, kernel_size=3, stride=1, is_causal=is_causal)
        self.mid_block = LTXVideoMidBlock3d(output_channel, num_layers=layers_per_block[0], resnet_eps=resnet_norm_eps,
                                            is_causal=is_causal, inject_noise=inject_noise[0],
                                            timestep_conditioning=timestep_conditioning)
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]
            up_block = LTXVideoUpBlock3d(input_channel, output_channel, num_layers=layers_per_block[i + 1],
                                         resnet_eps=resnet_norm_eps, spatio_temporal_scale=spatio_temporal_scaling[i],
                                         is_causal=is_causal, inject_noise=inject_noise[i + 1],
                                         timestep_conditioning=timestep_conditioning,
                                         upsample_residual=upsample_residual[i], upscale_factor=upsample_factor[i])
            self.up_blocks.append(up_block)

        self.norm_out = nn.RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
        self.conv_act = nn.SiLU()
        self.conv_out = LTXVideoCausalConv3d(output_channel, self.out_channels, kernel_size=3, stride=1, is_causal=is_causal)

        self.time_embedder = None
        self.scale_shift_table = None
        self.timestep_scale_multiplier = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2, 0)
            self.scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel ** 0.5)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.conv_in(hidden_states)
        if self.timestep_scale_multiplier is not None:
            temb = temb * self.timestep_scale_multiplier
        hidden_states = self.mid_block(hidden_states, temb)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb)
        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(), resolution=None, aspect_ratio=None,
                batch_size=hidden_states.size(0), hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        p = self.patch_size
        p_t = self.patch_size_t
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return hidden_states


class AutoencoderKLLTXVideo(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_channels=128,
                 block_out_channels=(128, 256, 512, 512), down_block_types=("LTXVideoDownBlock3D",) * 4,
                 decoder_block_out_channels=(128, 256, 512, 512), layers_per_block=(4, 3, 3, 3, 4),
                 decoder_layers_per_block=(4, 3, 3, 3, 4), spatio_temporal_scaling=(True, True, True, False),
                 decoder_spatio_temporal_scaling=(True, True, True, False),
                 decoder_inject_noise=(False,) * 5, downsample_type=("conv",) * 4,
                 upsample_residual=(False,) * 4, upsample_factor=(1, 1, 1, 1),
                 timestep_conditioning=False, patch_size=4, patch_size_t=1, resnet_norm_eps=1e-6,
                 scaling_factor=1.0, encoder_causal=True, decoder_causal=False,
                 spatial_compression_ratio=None, temporal_compression_ratio=None):
        super().__init__()
        self.encoder = LTXVideoEncoder3d(
            in_channels=in_channels, out_channels=latent_channels, block_out_channels=block_out_channels,
            down_block_types=down_block_types, spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block, downsample_type=downsample_type, patch_size=patch_size,
            patch_size_t=patch_size_t, resnet_norm_eps=resnet_norm_eps, is_causal=encoder_causal,
        )
        self.decoder = LTXVideoDecoder3d(
            in_channels=latent_channels, out_channels=out_channels, block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling, layers_per_block=decoder_layers_per_block,
            patch_size=patch_size, patch_size_t=patch_size_t, resnet_norm_eps=resnet_norm_eps,
            is_causal=decoder_causal, timestep_conditioning=timestep_conditioning, inject_noise=decoder_inject_noise,
            upsample_residual=upsample_residual, upsample_factor=upsample_factor,
        )
        self.register_buffer("latents_mean", torch.zeros((latent_channels,)), persistent=True)
        self.register_buffer("latents_std", torch.ones((latent_channels,)), persistent=True)
        self.spatial_compression_ratio = (
            patch_size * 2 ** sum(spatio_temporal_scaling) if spatial_compression_ratio is None else spatial_compression_ratio
        )
        self.temporal_compression_ratio = (
            patch_size_t * 2 ** sum(spatio_temporal_scaling) if temporal_compression_ratio is None else temporal_compression_ratio
        )
        self.use_tiling = False
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

    def enable_tiling(self):
        self.use_tiling = True

    # ---- tiling helpers (verbatim semantics from diffusers) ----
    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x):
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                time = self.encoder(x[:, :, :, i: i + self.tile_sample_min_height, j: j + self.tile_sample_min_width])
                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))
        return torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]

    def tiled_decode(self, z, temb):
        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                time = self.decoder(z[:, :, :, i: i + tile_latent_min_height, j: j + tile_latent_min_width], temb)
                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))
        return torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

    def _encode(self, x):
        batch_size, num_channels, num_frames, height, width = x.shape
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)
        return self.encoder(x)

    def encode(self, x):
        """x: (B,C,T,H,W) in [-1,1]. Returns h (B, 2C, t, h, w) where h[:, :C] is the mean."""
        return self._encode(x)

    def _decode(self, z, temb):
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, temb)
        return self.decoder(z, temb)

    def decode(self, z, temb=None):
        """z: (B,C,t,h,w) in raw (denormalized) space -> (B,C,T,H,W) in [-1,1]."""
        if temb is None:
            temb = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        return self._decode(z, temb)

    @classmethod
    def from_config_file(cls, config_path):
        cfg = json.load(open(config_path))
        keys = ["in_channels", "out_channels", "latent_channels", "block_out_channels", "down_block_types",
                "decoder_block_out_channels", "layers_per_block", "decoder_layers_per_block",
                "spatio_temporal_scaling", "decoder_spatio_temporal_scaling", "decoder_inject_noise",
                "downsample_type", "upsample_residual", "upsample_factor", "timestep_conditioning",
                "patch_size", "patch_size_t", "resnet_norm_eps", "scaling_factor", "encoder_causal",
                "decoder_causal", "spatial_compression_ratio", "temporal_compression_ratio"]
        kwargs = {k: cfg[k] for k in keys if k in cfg}
        return cls(**kwargs)


class OZLTXVAE:
    """Plugin-side VAE wrapper with diffusers-0.37-exact semantics and simple
    CPU<->CUDA memory management (the DiT stays on comfy's native management)."""

    def __init__(self, weights_path, config_path, dtype=torch.bfloat16):
        from safetensors.torch import load_file
        self.model = AutoencoderKLLTXVideo.from_config_file(config_path)
        sd = load_file(weights_path, device="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        self.model = self.model.to(dtype).eval()
        self.model.enable_tiling()  # matches official pipe.vae.enable_tiling()
        self.dtype = dtype
        self._mean = None
        self._std = None
        self._device = torch.device("cpu")

    def _ensure_cuda(self):
        if self._device.type != "cuda":
            self.model.to("cuda")
            self._device = torch.device("cuda")

    def release(self):
        if self._device.type != "cpu":
            self.model.to("cpu")
            self._device = torch.device("cpu")
            torch.cuda.empty_cache()

    def _stats(self):
        if self._mean is None:
            self._mean = self.model.latents_mean.view(1, -1, 1, 1, 1).float()
            self._std = self.model.latents_std.view(1, -1, 1, 1, 1).float()
        return self._mean, self._std

    def latent_stats(self):
        """(mean, std) of the latent channel statistics, fp32 cpu (1,128,1,1,1)."""
        m, s = self._stats()
        return m.cpu().clone(), s.cpu().clone()

    def encode(self, pixels):
        """pixels: (T,H,W,C) float 0-1 (any device) -> normalized latent (1,128,t,h,w) cpu float."""
        self._ensure_cuda()
        x = pixels.to(self._device, self.dtype).permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0
        with torch.no_grad():
            h = self.model.encode(x)
            mean = h[:, : h.shape[1] // 2]
            self._stats()
            z = (mean.float() - self._mean) / self._std
        return z.cpu()

    def encode_h(self, pixels_bchw):
        """Deterministic encoder output (posterior parameters).

        pixels_bchw: (1,3,T,H,W) in [-1,1]. Returns h (1,2C,f,h,w) bf16 on CPU.
        The posterior *sample* must be drawn by the caller in the exact order the
        official pipeline uses (video then mask, after the init noise draw).
        """
        self._ensure_cuda()
        with torch.no_grad():
            h = self.model.encode(pixels_bchw.to(self._device, self.dtype))
        return h.cpu()

    @staticmethod
    def posterior_sample(h, gen):
        """diffusers DiagonalGaussianDistribution.sample, bit-exact.

        mean = h[:, :C]; logvar = clamp(h[:, C:], -30, 20); std = exp(0.5*logvar);
        eps drawn on CPU with a CPU generator (bf16), then moved - the same sequence
        diffusers' randn_tensor produces for a cpu generator + cuda tensor.
        """
        c = h.shape[1] // 2
        mean = h[:, :c]
        logvar = torch.clamp(h[:, c:], -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.shape, generator=gen, dtype=torch.bfloat16, device="cpu").to(mean.device)
        return mean + std * eps

    def decode(self, z, decode_timestep=0.0):
        """z: normalized latent (1,128,t,h,w) -> (T,H,W,C) float 0-1 cpu.

        Denormalisation runs in fp32 then casts to the VAE dtype, matching the
        official `_denormalize_latents(...).to(bf16)` order.
        """
        self._ensure_cuda()
        m, s = self._stats()
        m, s = m.cpu(), s.cpu()
        with torch.no_grad():
            z_den = (z.float().cpu() * s + m).to(self.dtype).to(self._device)
            dec = self.model.decode(z_den, temb=torch.tensor(float(decode_timestep), device=self._device, dtype=self.dtype))
            px = (dec.float() + 1.0) / 2.0
        px = px.clamp(0, 1)[0].permute(1, 2, 3, 0).cpu()
        return px

    def encode_mask(self, pixels_0_100):
        """pixels_0_100: (T,H,W,C) with values {0,100} (object=100, bg=0).

        Mirrors the official OmnimatteZero mask path EXACTLY: the mask is fed
        RAW ({0,100}, with NO internal 2x-1) straight to the encoder, and the
        returned latent is the RAW mean (NOT latent-normalized). The official
        derives conditioning_mask_shape = |raw_latent.mean(0)| > 0.01 from this
        raw latent. Normalizing here changes the binary mask: a black background's
        raw mean is ~0 (so |mean| > 0.01 is False and bg is excluded), but after
        normalization z_mean ~ -latents_mean/latents_std is ~0.5 (well above 0.01),
        which wrongly flags bg as object and silently flips the whole mask to
        "object" -- breaking object removal. Returns float32 (1,128,f,h,w).
        """
        self._ensure_cuda()
        x = pixels_0_100.to(self._device, self.dtype).permute(3, 0, 1, 2).unsqueeze(0)  # (B,C,T,H,W), values {0,100}
        with torch.no_grad():
            h = self.model.encode(x)
            mean = h[:, : h.shape[1] // 2]
        return mean.cpu().float()
