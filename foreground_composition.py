from typing import Optional, Union
import torch
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video, load_video
from PIL import Image
from .OmnimatteZero import OmnimatteZero
from .model_loader_utils import tensor2pillist_upscale
from diffusers.video_processor import VideoProcessor
from transformers import  T5TokenizerFast
from .object_removal import get_diffusers_mask
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
def tensor_video_to_pil_images(video_tensor):
    """
    Converts a PyTorch tensor representing a video to a list of PIL Images.

    Args:
        video_tensor (torch.Tensor): A tensor of shape (1, frames, height, width, 3).
                                     Corresponds to batch size, frames, height, width, and RGB channels.

    Returns:
        List[Image.Image]: List of frames as PIL Images.
    """
    # Remove the batch dimension (shape: (frames, height, width, 3))
    video_tensor = video_tensor.squeeze(0)

    # Ensure the tensor is on CPU and convert to NumPy
    video_numpy = video_tensor.cpu().numpy()

    # Convert each frame to a PIL Image
    pil_images = [Image.fromarray(frame.astype('uint8')) for frame in video_numpy]

    return pil_images

def normalize_latents_(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

def denormalize_latents_(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

class MyAutoencoderKLLTXVideo(AutoencoderKLLTXVideo):
    def forward(
            self,
            sample: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        all, bg,  new_bg, mask = sample

        z_all = self.encode_normalize(all, sample_posterior, generator)
        z_bg = self.encode_normalize(bg, sample_posterior, generator) 
        z_new_bg = self.encode_normalize(new_bg, sample_posterior, generator)
        z_mask = self.encode_normalize(mask, sample_posterior, generator)
        
        #z_foreground = z_all * z_mask - z_bg * z_mask  

        z_mask_smooth = torch.sigmoid(z_mask * 5) 
        z_foreground = (z_all - z_bg) * z_mask_smooth

        z_result=z_new_bg +  z_foreground  
        #z_result = z_new_bg * (1 - z_mask) + z_foreground

        z_result = denormalize_latents_(z_result, self.latents_mean, self.latents_std)
        dec = self.decode(z_result, temb)

        if not return_dict:
            return (dec)
        return dec
    
    def encode_normalize(self, x, sample_posterior, generator):
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        #return z
        return normalize_latents_(z, self.latents_mean, self.latents_std)

    def forward_encode(
            self,
            sample: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        if not return_dict:
            return (z,)
        return z


def pre_compose_data(video_p, video_bg,  video_new_bg,nframes,w,h):
    nframes = min(video_new_bg.shape[0], video_p.shape[0],video_bg.shape[0],nframes)
    video_p=tensor2pillist_upscale(video_p, width=w, height=h)[:nframes]
    video_bg=tensor2pillist_upscale(video_bg, width=w, height=h)[:nframes]
    video_new_bg=tensor2pillist_upscale(video_new_bg, width=w, height=h)[:nframes]
    video_processor=VideoProcessor(vae_scale_factor=32)
    video_p=video_processor.preprocess_video(video_p, width=w, height=h).bfloat16().cuda()
    video_bg=video_processor.preprocess_video(video_bg, width=w, height=h).bfloat16().cuda()
    video_new_bg=video_processor.preprocess_video(video_new_bg, width=w, height=h).bfloat16().cuda()
    return video_p, video_bg,  video_new_bg,nframes

def compose_video(pipe, positive, negative,num_inference_steps,seed,guidance_scale,condition_latents,expected_height, expected_width,num_frames, device):

    with torch.no_grad():
    
        prompt = ""
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
        #expected_height, expected_width = 512, 768
        tokenizer=T5TokenizerFast.from_pretrained(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/tokenizer"))
        prompt_attention_mask=get_diffusers_mask(tokenizer,prompt,device)
        negative_prompt_attention_mask=get_diffusers_mask(tokenizer,negative_prompt,device) #torch.Size([1, 128])
        #expected_height, expected_width = video[0].size[1], video[0].size[0]
        #num_frames = len(video)
        
        torch.cuda.empty_cache()
        # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
        video = pipe(
            prompt=None,
            negative_prompt=None,
            width=expected_width,
            height=expected_height,
            num_frames=num_frames,
            denoise_strength=0.3,
            num_inference_steps=num_inference_steps,
            latents=condition_latents,
            decode_timestep=0.05,
            image_cond_noise_scale=0.025,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            prompt_embeds=positive,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            output_type="pil",
        ).frames[0]
        return video
