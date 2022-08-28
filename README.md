# Awesome Stable-Diffusion

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a list of [Free](https://en.wikipedia.org/wiki/Free_software) Software and Resources for the [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) AI Model by StabilityAI. 

🖊️ marks content that requires sign-up or account creation for a third party service outside github
💵 marks content non-free, commercial content that may require payment.

See [Contributing](.github/CONTRIBUTING.md).

## Official Repositories

 **[CompVis/Stable Diffusion](https://github.com/CompVis/stable-diffusion)** - The official release of Stable Diffusion with cli, an AI-based Safety Classifier to (detect and suppresses sexualized content and all the necessary files to get running.
 
## Forks

All forks listed here add additional features and optimisations and are generally faster than the original release as they will keep the model in memory rather than reloading it every prompt. Most forks seem to remove the Safety Classifier, making them potentially unsuitable for chat-bots and other environments that may deploy automated detection

* [basujindal/stable-diffusion](https://github.com/basujindal/stable-diffusion) - "Optimized Stable Diffusion", fork with dramatically reduced VRAM requirements through model splitting, enabling stable diffusion on lower end video cards. Includes gradio web interface, prompt weights
* [lstein/stable-diffusion](https://github.com/lstein/stable-diffusion) - Fork adding a conversational cli interface, basic web interface, GFPGAN, ESRGAN, prompt weights, textual-inversion
* [progrock-stable](https://github.com/lowfuel/progrock-stable) - Fork with optional Web GUI, GoBIG/ESRGAN upscaling
* [hlky/stable-diffusion](https://github.com/hlky/stable-diffusion) - Fork with optional [Gradio UI](https://github.com/hlky/stable-diffusion-webui) with support for txt2img, img2img inpainting, GFPGAN, ESGRAN, prompt weights and optional textual-inversion


## Models and Weights

Models (.ckpt files) must be separately downloaded and are required to run Stable Diffusion.

* 🖊️ [The offical model card](https://huggingface.co/CompVis/stable-diffusion) - Official Model Card Hugging Face with all versions of the model. Download requires sign in and acceptance of terms of service.
[stable-diffusion-v-1-4-original.chkpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) - The latest model from the model card, required to run Stable Diffusion.
* [RealESRGAN Models](https://github.com/xinntao/Real-ESRGAN/releases/) - Download location for the latest RealESRGAN models required to use upscaling by many forks implementing it. Different models exist for realistic and anime content. Please refer to the fork documentation to identify the ones you need.

 
## Online Demos & Collabs

* [HuggingFace/StabilityAI](https://huggingface.co/spaces/stabilityai/stable-diffusion) - The official demo on HuggingFace Spaces.
* 🖊️💵  [Offical Collab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) - The official, optimized collab for running SD on Google Cloud. Due to VRAM requirements required Collab Pro to create images.

## Complimentary Models and Tools (open source)

## Customisation
* [textual-inversion](https://github.com/rinongal/textual_inversion) - Adding your own personalized content to Stable Diffusion. 

### Upscaling
* [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face Restoration GAN included in several forks for automatically fixing deformed faces commonly found in SD output.* * [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) - ESRGAN Upscaling (2x, 4x) and content restoration, useful for overcoming. Python: ```pip install realesrgan```
* [Cupscale](https://github.com/n00mkrad/cupscale) - Graphical User Interface to run various upscaling models, including ESRGAN and RealESRGAN
* [BasicSR](https://github.com/XPixelGroup/BasicSR) - Open Source Upscaling and Restoration toolbox supporting.
* [lama-cleaner](https://github.com/Sanster/lama-cleaner) - Content aware AI inpainting tool with useful for removing unwanted objects or defects from images. Python: ```pip install lama-cleaner```


### Task Chaining
* [chaiNNer](https://github.com/joeyballentine/chaiNNer) - Graphical Node Based Editor for chaining image processing tasks.

## Tutorials & Comparisons
* [Stable Diffusion How To](https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/) - A basic tutorial on how to get Stable Diffusion up and running.
* ["Ultimate GUI Retard Guide](https://rentry.org/GUItard) - Tutorial for installing the [hlky fork](https://github.com/hlky/stable-diffusion) along with it's [WebUI](https://github.com/hlky/stable-diffusion-webui).
* [Building a SD Discord Bot](https://replicate.com/blog/build-a-robot-artist-for-your-discord-server-with-stable-diffusion) - Tutorial on how to build a stable diffusion discord bot using python.
- [AI Image Generator Comparison](https://petapixel.com/2022/08/22/ai-image-generators-compared-side-by-side-reveals-stark-differences/) Visual comparison between Dall-e, Stable Diffusion and Midjourney by PetaPixel.com 

## Prompt Building
* [Prompt Mania](https://promptomania.com/) - A visual prompt construction tool.
* [Lexica.art](https://lexica.art/) - A searchable, visual database of images and the prompts settings to recreate them.

## Discords
* [Official Discord](https://discord.gg/stablediffusion) - The official Stable Diffusion Discord by StabilityAI

## Online Services implementing Stable Diffusion

* 🖊️💵 [Dream Studio](http://beta.dreamstudio.ai/) - Online Generation Platform by StabilityAI, the creators of Stable Diffusion. Similar to services like Dall-e or Midjourney, this operates on a credit model with some free allowance of credits given to signed up users on a monthly basis.
*  🖊️💵 [dream.ai](https://www.dream.ai/) - Online Art Generation Service (mobile apps available) by Wombo.ai implementing Stable Diffusion.





