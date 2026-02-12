import os
import torch
from diffusers import  LTXLatentUpsamplePipeline,AutoencoderKLLTXVideo
from .wrapper.transformer_ltx import LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from transformers import  T5TokenizerFast
from safetensors.torch import load_file
from .OmnimatteZero import OmnimatteZero
from contextlib import contextmanager
import sys
from diffusers import GGUFQuantizationConfig
from accelerate import init_empty_weights
@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % 32)
    width = width - (width % 32)
    return height, width

def  load_upsample_model(vae_path, model_path,cur_dir,):
    latent_upsampler_config=LTXLatentUpsamplerModel.load_config(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/upsampler/latent_upsampler/config.json"))
    pipe=LTXLatentUpsamplerModel.from_config(latent_upsampler_config,torch_dtype=torch.bfloat16)
    u_dict=load_file(model_path)
    pipe.load_state_dict(u_dict, strict=False)
    del u_dict
    vae=AutoencoderKLLTXVideo.from_single_file(vae_path,config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"), torch_dtype=torch.bfloat16)

    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(os.path.join(cur_dir, "upsampler"),latent_upsampler=pipe, vae=vae, torch_dtype=torch.bfloat16)
    pipe_upsample.to("cuda")
    pipe_upsample.vae.enable_tiling()
    return pipe_upsample


def load_model(model_path,gguf_path,vae_path,cur_dir,compose_mode=False):
    if compose_mode:
        from .foreground_composition import MyAutoencoderKLLTXVideo
        with temp_patch_module_attr("diffusers", "AutoencoderKLLTXVideo", MyAutoencoderKLLTXVideo):
            try:
                vae_config=MyAutoencoderKLLTXVideo.load_config(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"))
                vae=MyAutoencoderKLLTXVideo.from_config(vae_config,torch_dtype=torch.bfloat16)
                vae.load_state_dict(load_file(vae_path), strict=False)
                vae.eval().to("cuda",torch.bfloat16)
            except:
                print("load vae error,use normal load mode")
                vae=MyAutoencoderKLLTXVideo.from_single_file(vae_path,config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"), torch_dtype=torch.bfloat16)
    else:
        try:
            vae_config=AutoencoderKLLTXVideo.load_config(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"))
            vae=AutoencoderKLLTXVideo.from_config(vae_config,torch_dtype=torch.bfloat16)
            vae.load_state_dict(load_file(vae_path), strict=False)
            vae.eval().to("cuda",torch.bfloat16)
        except:
            print("load vae error,use normal load mode")
            vae=AutoencoderKLLTXVideo.from_single_file(vae_path,config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/vae/config.json"), torch_dtype=torch.bfloat16)
    with temp_patch_module_attr("diffusers", "LTXVideoTransformer3DModel", LTXVideoTransformer3DModel):
        if gguf_path is not None:
            transformer = LTXVideoTransformer3DModel.from_single_file(
                gguf_path,
                config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/transformer"),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16) 
        else:
            try:
                transformer=LTXVideoTransformer3DModel.from_single_file(model_path,config=os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/transformer/config.json"), torch_dtype=torch.bfloat16)         
            except:
                print("load model error,use normal load mode")
                transformer_config=LTXVideoTransformer3DModel.load_config(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/transformer/config.json"))
                with init_empty_weights():
                    transformer=LTXVideoTransformer3DModel.from_config(transformer_config,torch_dtype=torch.bfloat16)
                transformer.load_state_dict(load_file(model_path), strict=False,assign=True)
    pipe = OmnimatteZero.from_pretrained(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers"),text_encoder=None,vae=vae,transformer=transformer, torch_dtype=torch.bfloat16,)
    pipe.vae.enable_tiling()
    return pipe


def get_diffusers_mask(tokenizer,prompt,device):
    text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)
    prompt_attention_mask = prompt_attention_mask.view(1, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(1, 1)
    return prompt_attention_mask
    

def inference(pipe, prompt_embeds, negative_prompt_embeds, video, mask, num_frames, expected_height, expected_width,seed,guidance_scale,
              num_inference_steps, cur_dir,device):
    
    condition1 = LTXVideoCondition(video=video, frame_index=0)
    condition2 = LTXVideoCondition(video=mask, frame_index=0)

    prompt = "Empty"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    #expected_height, expected_width = 512, 768
    tokenizer=T5TokenizerFast.from_pretrained(os.path.join(cur_dir, "LTX-Video-0.9.7-diffusers/tokenizer"))
    prompt_attention_mask=get_diffusers_mask(tokenizer,prompt,device)
    negative_prompt_attention_mask=get_diffusers_mask(tokenizer,negative_prompt,device) #torch.Size([1, 128])
    #print(prompt_attention_mask.shape)
    downscaled_height, downscaled_width = int(expected_height), int(expected_width)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height,downscaled_width)
    video = pipe.my_call(
        conditions=[condition1, condition2],
        prompt=None,
        negative_prompt=None,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        output_type="pil",
    )
    video = video.frames[0]
    return video

