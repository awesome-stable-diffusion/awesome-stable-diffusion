# Awesome Stable-Diffusion

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a list of software and resources for the [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) AI model.

- 🖊️ marks content that requires sign-up or account creation for a third party service outside GitHub
- 💵 marks [Non-Free](https://en.wikipedia.org/wiki/Free_software) content; commercial content that may require any kind of payment.

Due to the fast-moving nature of the topic, entries in the list may be removed at an expedited rate until the ecosystem matures.

See [Contributing](.github/CONTRIBUTING.md).

## Official Resources

* **[CompVis/Stable Diffusion](https://github.com/CompVis/stable-diffusion)** - The official release of Stable Diffusion including a CLI, an AI-based Safety Classifier, which detects and suppresses sexualized content, and all the necessary files to get running.
* [stability-AI/stability-sdk](https://github.com/stability-AI/stability-sdk) - The official SDK used to build python applications integrated with StabilityAI's cloud platform instead of hosting the model locally. Operation requires an API Key (🖊️💵).
* [Public Release Announcement](https://stability.ai/blog/stable-diffusion-public-release) - StabilityAI's announcement about the public release of Stable Diffusion.
* 🖊️ [Official Discord](https://discord.gg/stablediffusion) - The official Stable Diffusion Discord by StabilityAI.

## Actively Maintained Forks & Containers

All forks listed here add additional features and optimisations and are generally faster than the original release, as they keep the model in memory rather than reloading it after every prompt. Most forks seem to remove the Safety Classifier which may present a risk if used to provide public-facing services, such as Discord bots.

* [basujindal/stable-diffusion](https://github.com/basujindal/stable-diffusion) - "Optimized Stable Diffusion"—a fork with dramatically reduced VRAM requirements through model splitting, enabling Stable Diffusion on lower-end graphics cards; includes a GradIO web interface and support for weighted prompts. 

* [bes-dev/stable_diffusion.openvino](https://github.com/bes-dev/stable_diffusion.openvino) - Fork for running the model using a CPU compatible with OpenVINO

* [hlky/stable-diffusion](https://github.com/hlky/stable-diffusion) - Very active fork with optional, highly featureful [Gradio UI](https://github.com/hlky/stable-diffusion-webui) and support for txt2img, img2img inpainting, GFPGAN, ESRGAN, weighted prompts, optimized low memory version, optional [textual-inversion](https://textual-inversion.github.io/) and more.
* [lstein/stable-diffusion](https://github.com/lstein/stable-diffusion) - Very active fork adding a conversational CLI, basic web interface and support for GFPGAN, ESRGAN, weighted prompts, img2img, [textual-inversion](https://textual-inversion.github.io/) as well as inference on Apple M1.
* [lowfuel/progrock-stable](https://github.com/lowfuel/progrock-stable) - Fork with optional Web GUI and a different approach to upscaling (GoBIG/ESRGAN)
  * [txt2imghd](https://github.com/jquesnelle/txt2imghd) - Fork of progrock diffusion that creates detailed, higher-resolution images by first generating an image from a prompt, upscaling it, then running img2img on smaller pieces of the upscaled image, and blending the results back into the original image.
* [replicate/copg-stable-diffusion](https://github.com/replicate/cog-stable-diffusion) - [Cog machine learning container](https://github.com/replicate/cog) of SD v1.4
* [stable-diffusion-jupyterlab-docker](https://github.com/pieroit/stable-diffusion-jupyterlab-docker) - A Docker setup ready to go with Jupyter notebooks for Stable Diffusion. 

## Models and Weights

Models (.ckpt files) must be separately downloaded and are required to run Stable Diffusion. The latest model release is v1.4.

* 🖊️ [Official Model Card](https://huggingface.co/CompVis/stable-diffusion) - Official Model Card on Hugging Face with all versions of the model. Download requires sign-in and acceptance of terms of service.
 * [stable-diffusion-v-1-4-original.chkpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) - The latest model's card
* [RealESRGAN Models](https://github.com/xinntao/Real-ESRGAN/releases/) - Download location for the latest RealESRGAN models required to use the upscaling features implemented by many forks. Different models exist for realistic and anime content. Please refer to the fork documentation to identify the ones you need.

 
## Online Demos & Notebooks

* [HuggingFace/StabilityAI](https://huggingface.co/spaces/stabilityai/stable-diffusion) - The official demo on HuggingFace Spaces.
* 🖊️💵 [Offical Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) - The official, optimized colab for running SD on Google Cloud. Due to VRAM requirements required Colab Pro to create images.
* [andreasjansson/stable-diffusion-animation](https://replicate.com/andreasjansson/stable-diffusion-animation) - Animate between prompts.

## Complimentary Models and Tools

Tools and models for use in conjuction with Stable Diffusion.

### Customisation
* [textual-inversion](https://github.com/rinongal/textual_inversion) - Adding your own personalized content to Stable Diffusion without retraining the model ([Paper](https://textual-inversion.github.io/), [Paper2](https://dreambooth.github.io/)). 

### GUIS

* [hlky/stable-diffusion-webui](https://github.com/hlky/stable-diffusion-webui) - GradIO based web UI from the hlsk/stable-diffusion fork with support for txt2img, img2img (with basic mask editor), GAFPGAN and RealESRGAN.
* [Stable Diffusion GRisk GUI ]([https://grisk.itch.io/stable-diffusion-gui]) - Windows GUI binary for SD.


### Upscaling
* [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) - ESRGAN Upscaling (2x, 4x) and content restoration, useful for overcoming. Python: ```pip install realesrgan```
* [Cupscale](https://github.com/n00mkrad/cupscale) - Graphical User Interface to run various upscaling models, including ESRGAN and RealESRGAN
* [BasicSR](https://github.com/XPixelGroup/BasicSR) - Open Source Upscaling and Restoration toolbox supporting.
* [BSRGAN](https://github.com/cszn/BSRGAN) - BSRGAN - Another upscaling solution specialized in upscaling degraded images.

### Content Restoration
* [lama-cleaner](https://github.com/Sanster/lama-cleaner) - Content aware AI inpainting tool with useful for removing unwanted objects or defects from images. Python: ```pip install lama-cleaner```
* [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face Restoration GAN included in several forks for automatically fixing deformed faces commonly found in SD output.* 
* [CodeFormer](https://github.com/sczhou/CodeFormer) - Another Face Restoration Model [Paper: Towards Robust Blind Face Restoration with Codebook Lookup Transformer](https://arxiv.org/abs/2206.11253).

### Task Chaining
* [chaiNNer](https://github.com/joeyballentine/chaiNNer) - Graphical Node Based Editor for chaining image processing tasks.
* [ai-art-generator](https://github.com/rbbrdckybk/ai-art-generator) - AI Art Generation Suite combing SD and Models for high volume art generation.

### Prompt Building

Prompts are the inputs used with diffusion models to manipulate their output. 

* [PromptoMania](https://promptomania.com/) - A visual prompt construction tool.
* [Lexica.art](https://lexica.art/) - A searchable, visual database of images and the prompts settings used to recreate them.
* 🖊️[Phraser](https://phraser.tech/) - A visual prompt builder drawing on a database of examples. (Requires account creation)
* [ai-art.com/modifiers](https://www.the-ai-art.com/modifiers) - A visual reference guide to keywords.
* [pharmapsychotic/clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator) - Jupiter Notebook for reverse engineering prompts from images.
* [img2prompt](https://replicate.com/methexis-inc/img2prompt) - Upload image, get possible prompts for similar results (clip interrogator)
* [rom1504/clip-retrieval](https://github.com/rom1504/clip-retrieval) - Reverse searches prompt keywords into the image dataset used by Stable Diffusion and other models. [Online GUI](https://rom1504.github.io/clip-retrieval/) available.


## Tutorials & Comparisons

Tutorials and high quality edutcational resources.

### Getting up and running

* [Stable Diffusion How To](https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/) - A basic tutorial on how to get Stable Diffusion up and running.
* [Installing on Windows](https://rentry.org/SDInstallation) - A guide to install and run SD on Windows with c
* [Running on M1 Apple Silicon](https://www.reddit.com/r/StableDiffusion/comments/wx0tkn/stablediffusion_runs_on_m1_chips/) - Reddit thread with instructions on how to run Stable Diffusion on Apple M1 CPU and GPU.
* [Easy CPU-only Stable Diffusion](https://rentry.org/cpu_stable_diffusion_guide) - Easy CPU-only stable diffusion without littering too much your GNU/Linux system
* ["Ultimate GUI Retard Guide](https://rentry.org/GUItard) - Tutorial for installing the [hlky fork](https://github.com/hlky/stable-diffusion) along with it's [WebUI](https://github.com/hlky/stable-diffusion-webui).

### Learning and mastering

* [Stable Diffusion Akashic Records](https://github.com/Maks-s/sd-akashic) - Comprehensive curated list of guides, studies, keywords, promts and artists.
* [Sunny's Tips & Tricks](https://docs.google.com/document/u/1/d/1K6EqcsRut0InU-8jB0yOvBMGesf5Dndg5FwyuaYLqNc/mobilebasic) - Sunny's SD Tips & Tricks Google Doc with lots of visual comparisons and useful information.
* [AI Image Generator Comparison](https://petapixel.com/2022/08/22/ai-image-generators-compared-side-by-side-reveals-stark-differences/) Visual comparison between Dall-e, Stable Diffusion and Midjourney by PetaPixel.com 
* [Gettint great results at Stable Diffusion](https://old.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/) Guide on how to get images that don't suck. 

### Extending functionality
* [Building a SD Discord Bot](https://replicate.com/blog/build-a-robot-artist-for-your-discord-server-with-stable-diffusion) - Tutorial on how to build a stable diffusion discord bot using python.


## Community resources
* [1 week of Stable Diffusion](https://multimodal.art/news/1-week-of-stable-diffusion) - Curated list of SD services, adaptations, user interfaces and integrations

## Social Media
* [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/) - Stable Diffusion Reddit.

## Online Services implementing Stable Diffusion
* 🖊️💵 [Dream Studio](http://beta.dreamstudio.ai/) - Online Generation Platform by StabilityAI, the creators of Stable Diffusion. Similar to services like Dall-e or Midjourney, this operates on a credit model with some free allowance of credits given to signed up users on a monthly basis.
* 🖊️💵 [dream.ai](https://www.dream.ai/) - Online Art Generation Service (mobile apps available) by Wombo.ai implementing Stable Diffusion.




