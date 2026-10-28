# OmnimatteZero  
[Official implementation of OmnimatteZero](https://github.com/dvirsamuel/OmnimatteZero/tree/main): Training-Free Video Matting and Compositing via Latent Diffusion Models

# Update
* 改成comfyUI的patcher封装/ change to ComfyUI patcher encapsulation

# 1. Installation

In the './ComfyUI /custom_nodes' directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_OmnimatteZero.git
```
---

# 2. Models
* Vae [a-r-r-o-w/LTX-Video-0.9.7-diffusers](https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.7-diffusers/tree/main)
* Dit or gguf or int8  [aliyun](https://pan.quark.cn/s/c0bc1f335f8c)  or huggingface [smthem/LTX-Video-0.9.7-diffusers-merge](https://huggingface.co/smthem/LTX-Video-0.9.7-diffusers-merge/tree/main)

```
--  ComfyUI/models/vae
    |-- LTX-Video-0.9.7-vae-diffusers.safetensors #vae rename
--  ComfyUI/models/diffusion_models # optional 
    |-- LTX-Video-0.9.7-comfy-int8_convrot.safetensors #  LTX-Video-0.9.7-diffusers.safetensors
--  ComfyUI/models/gguf  # optional 
    |-- LTX-Video-0.9.7-diffusers-Q8_0.gguf # need gguf nodes
```

# 3. Example
![](https://github.com/smthemex/ComfyUI_OmnimatteZero/blob/main/example_workflows/example.png)


# 4. Citation
```
@inproceedings{samuel2025omnimattezero,
  author    = {Dvir Samuel and Matan Levy and Nir Darshan and Gal Chechik and Rami Ben-Ari},
  title     = {OmnimatteZero: Fast Training-free Omnimatte with Pre-trained Video Diffusion Models},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  year      = {2025}
}
``
