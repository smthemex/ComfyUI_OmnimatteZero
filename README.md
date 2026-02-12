# OmnimatteZero  
[Official implementation of OmnimatteZero](https://github.com/dvirsamuel/OmnimatteZero/tree/main): Training-Free Video Matting and Compositing via Latent Diffusion Models

# Update
* add compose mode，but  got a normal effect.. 
* Test Vram 12G,Ram 64,  video 1280x720 5s ,if lower Vram, keep 'block_number' <10 and >0, if  'block_number'=0 ,will run in full mode;
* 小显存block_number 设置1-10，大的可以设置为0 或者大于10，直接调整，distill 模型对于简单的去水印可以试试


# 1. Installation

In the './ComfyUI /custom_nodes' directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_OmnimatteZero.git
```
---

# 2. Requirements  
* No need,Perhaps someone may be missing the library.没什么特殊的库,懒得删了
```
pip install -r requirements.txt
```
# 3. Models
* Vae [a-r-r-o-w/LTX-Video-0.9.7-diffusers](https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.7-diffusers/tree/main)
* Dit or gguf  [aliyun](https://pan.quark.cn/s/c0bc1f335f8c)  or huggingface [smthem/LTX-Video-0.9.7-diffusers-merge](https://huggingface.co/smthem/LTX-Video-0.9.7-diffusers-merge/tree/main)

```
--  ComfyUI/models/vae
    |-- LTX-Video-0.9.7-vae-diffusers.safetensors #vae rename
--  ComfyUI/models/diffusion_models # optional 
    |-- LTX-Video-0.9.7-diffusers.safetensors
--  ComfyUI/models/gguf  # optional 
    |-- LTX-Video-0.9.7-diffusers-Q8_0.gguf
```

# 4. Example
![](https://github.com/smthemex/ComfyUI_OmnimatteZero/blob/main/example_workflows/example.png)
* compose 
![](https://github.com/smthemex/ComfyUI_OmnimatteZero/blob/main/example_workflows/example_c.png)

# 5. Citation
```
@inproceedings{samuel2025omnimattezero,
  author    = {Dvir Samuel and Matan Levy and Nir Darshan and Gal Chechik and Rami Ben-Ari},
  title     = {OmnimatteZero: Fast Training-free Omnimatte with Pre-trained Video Diffusion Models},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  year      = {2025}
}
``
