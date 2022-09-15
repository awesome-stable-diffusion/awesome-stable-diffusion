# Awesome Stable-Diffusion

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a list of software and resources for the [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) AI model.

- 🖊️ marks content that requires sign-up or account creation for a third party service outside GitHub.
- 💵 marks [Non-Free](https://en.wikipedia.org/wiki/Free_software) content: commercial content that may require any kind of payment.

Due to the fast-moving nature of the topic, entries in the list may be removed at an expedited rate until the ecosystem matures.

See [Contributing](.github/CONTRIBUTING.md).

## Official Resources

* **[CompVis/Stable Diffusion](https://github.com/CompVis/stable-diffusion)** - The official release of Stable Diffusion including a CLI, an AI-based Safety Classifier, which detects and suppresses sexualized content, and all the necessary files to get running.
* [stability-AI/stability-sdk](https://github.com/stability-AI/stability-sdk) - The official SDK used to build python applications integrated with StabilityAI's cloud platform instead of hosting the model locally. Operation requires an API Key (🖊️💵).
* [Public Release Announcement](https://stability.ai/blog/stable-diffusion-public-release) - StabilityAI's announcement about the public release of Stable Diffusion.
* 🖊️ [Official Discord](https://discord.gg/stablediffusion) - The official Stable Diffusion Discord by StabilityAI.
* [laion-aesthetic](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/images) - The dataset used train stable diffusion, useful for querying to see if a concept is represented. 
## Actively Maintained Forks and Containers

All forks listed here add additional features and optimisations and are generally faster than the original release, as they keep the model in memory rather than reloading it after every prompt. Most forks seem to remove the Safety Classifier which may present a risk if used to provide public-facing services, such as Discord bots.

* [basujindal/stable-diffusion](https://github.com/basujindal/stable-diffusion) - "Optimized Stable Diffusion"—a fork with dramatically reduced VRAM requirements through model splitting, enabling Stable Diffusion on lower-end graphics cards; includes a GradIO web interface and support for weighted prompts. 

* [bes-dev/stable_diffusion.openvino](https://github.com/bes-dev/stable_diffusion.openvino) - A fork for running the model using a CPU compatible with OpenVINO.

* [sd-webui/stable-diffusion-webui](https://github.com/sd-webui/stable-diffusion-webui) - Very active fork with optional, highly featureful Gradio UI and support for txt2img, img2img inpainting, GFPGAN, ESRGAN, weighted prompts, optimized low memory version, optional [textual-inversion](https://textual-inversion.github.io/) and more.

* [lstein/stable-diffusion](https://github.com/lstein/stable-diffusion) - Very active fork adding a conversational CLI, basic web interface and support for GFPGAN, ESRGAN, weighted prompts, img2img, [textual-inversion](https://textual-inversion.github.io/) as well as inference on Apple M1.

* [lowfuel/progrock-stable](https://github.com/lowfuel/progrock-stable) - Fork with optional Web GUI and a different approach to upscaling (GoBIG/ESRGAN)
  * [txt2imghd](https://github.com/jquesnelle/txt2imghd) - Fork of progrock diffusion that creates detailed, higher-resolution images by first generating an image from a prompt, upscaling it, then running img2img on smaller pieces of the upscaled image, and blending the results back into the original image.
* [replicate/copg-stable-diffusion](https://github.com/replicate/cog-stable-diffusion) - [Cog machine learning container](https://github.com/replicate/cog) of SD v1.4.
* [stable-diffusion-jupyterlab-docker](https://github.com/pieroit/stable-diffusion-jupyterlab-docker) - A Docker setup ready to go with Jupyter notebooks for Stable Diffusion. 

## Models and Weights

Models (.ckpt files) must be separately downloaded and are required to run Stable Diffusion. The latest model release is v1.4.

* 🖊️ [Official Model Card](https://huggingface.co/CompVis/stable-diffusion) - Official Model Card on Hugging Face with all versions of the model. Download requires sign-in and acceptance of terms of service.
 * [stable-diffusion-v-1-4-original.chkpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) - The latest model's card
* [RealESRGAN Models](https://github.com/xinntao/Real-ESRGAN/releases/) - Download location for the latest RealESRGAN models required to use the upscaling features implemented by many forks. Different models exist for realistic and anime content. Please refer to the fork documentation to identify the ones you need.

 
## Online Demos and Notebooks

* [HuggingFace/StabilityAI](https://huggingface.co/spaces/stabilityai/stable-diffusion) - The official demo on HuggingFace Spaces.
* 🖊️💵 [Offical Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) - The official, optimized colab for running SD on Google Cloud. Due to VRAM requirements required Colab Pro to create images.
* [andreasjansson/stable-diffusion-animation](https://replicate.com/andreasjansson/stable-diffusion-animation) - Animate between prompts.
* 🖊️ [Stable Diffusion Interpolation](https://colab.research.google.com/drive/1EHZtFjQoRr-bns1It5mTcOVyZzZD9bBc?usp=sharing) - AA simple implementation of generating N interpolated images (Colab)

## Complementary Models and Tools

Tools and models for use in conjuction with Stable Diffusion
### Img2Img

* [Prompt to Prompt](https://github.com/bloc97/CrossAttentionControl) - Unofficial Implementation of Cross-attention-control for prompt to prompt image editing. 

### Customisation
* [textual-inversion](https://github.com/rinongal/textual_inversion) - Addition of personalized content to Stable Diffusion without retraining the model ([Paper](https://textual-inversion.github.io/), [Paper2](https://dreambooth.github.io/)). 

### GUIS
* 🖊️💵 [Auto SD Workflow](https://www.patreon.com/auto_sd_workflow) - A UI for [lstein/stable-diffusion](https://github.com/lstein/stable-diffusion)'s dream.py with optimized UX for large-scale/production workflow around image synthesis. [Video Walkthrough](https://vimeo.com/748114237).
* 🖊️ [DiffusionUI](https://github.com/leszekhanusz/diffusion-ui) - web UI made with Vue.js inspired by Dall-e using [diffusers](https://github.com/huggingface/diffusers), perfect for inpainting. [Video demo](https://www.youtube.com/watch?v=AFZvW5qURes)
* 🖊️ [SD-MUI](https://sd-mui.vercel.app/) - mobile-first PWA made with React + MaterialUI.  Run free locally or use free & paid credits on the live site.  ([Source Code](https://github.com/gadicc/stable-diffusion-react-nextjs-mui-pwa)) `MIT License` `TypeScript`
* [sd-webui/stable-diffusion-webui](https://github.com/sd-webui/stable-diffusion-webui) - GradIO based web UI included in an active fork.
* [Stable Diffusion GRisk GUI ]([https://grisk.itch.io/stable-diffusion-gui]) - Windows GUI binary for SD.

### Upscaling
* [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) - ESRGAN Upscaling (2x, 4x) and content restoration. Python: ```pip install realesrgan```
* [Cupscale](https://github.com/n00mkrad/cupscale) - GUI for running various upscaling models, including ESRGAN and RealESRGAN.
* [BasicSR](https://github.com/XPixelGroup/BasicSR) - Open-source upscaling and restoration toolbox supporting several models.
* [BSRGAN](https://github.com/cszn/BSRGAN) - BSRGAN—another upscaling solution specialized in upscaling degraded images.

### Content Restoration
* [lama-cleaner](https://github.com/Sanster/lama-cleaner) - Content aware AI inpainting tool useful for removing unwanted objects or defects from images. Python: ```pip install lama-cleaner```
* [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face Restoration GAN included in several forks for automatically fixing the face deformation commonly found in SD output.
* [CodeFormer](https://github.com/sczhou/CodeFormer) - Another Face Restoration model ([Paper](https://arxiv.org/abs/2206.11253)).

### Task Chaining
* [chaiNNer](https://github.com/joeyballentine/chaiNNer) - Graphical node-based editor for chaining image processing tasks.
* [ai-art-generator](https://github.com/rbbrdckybk/ai-art-generator) - AI art generation suite combining Stable Diffusion and other models for high volume art generation.

### Prompt Building

Prompts are the instructions given to diffusion models to manipulate their output. 

* [PromptoMania](https://promptomania.com/) - A visual prompt construction tool.
* [Lexica.art](https://lexica.art/) - A searchable, visual database of images and the prompts settings used to create them.
* 🖊️[Phraser](https://phraser.tech/) - A visual prompt builder drawing on a database of examples. (Requires account creation)
* [ai-art.com/modifiers](https://www.the-ai-art.com/modifiers) - A visual reference guide for keywords.
* [pharmapsychotic/clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator) - Jupyter notebook uses CLIP models to suggest a prompt for images similar to a given image ([Demo](https://replicate.com/methexis-inc/img2prompt)).
* [rom1504/clip-retrieval](https://github.com/rom1504/clip-retrieval) - Searches for prompt keywords in the datasets used in training Stable Diffusion and other models ([Online GUI](https://rom1504.github.io/clip-retrieval/)).


## Tutorials and Comparisons

Tutorials and high quality educational resources

### Getting Up and Running
* [Stable Diffusion How To](https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/) - A basic tutorial on getting Stable Diffusion up and running.
* [Installing on Windows](https://rentry.org/SDInstallation) - A guide on installing and runing Stable Diffusion on Windows.
* [Running on M1 Apple Silicon](https://www.reddit.com/r/StableDiffusion/comments/wx0tkn/stablediffusion_runs_on_m1_chips/) - Reddit thread with instructions on running Stable Diffusion on Apple M1 CPU and GPU.
* [Easy CPU-only Stable Diffusion](https://rentry.org/cpu_stable_diffusion_guide) - A guide on setting up CPU-only Stable Diffusion for GNU/Linux without littering the system with dependencies.
* ["Ultimate GUI Retard Guide"](https://rentry.org/GUItard) - Tutorial for installing the [sd-webui fork](https://github.com/sd-webui/stable-diffusion-webui).


### Learning and Mastering

* [Stable Diffusion Akashic Records](https://github.com/Maks-s/sd-akashic) - A comprehensive curated list of guides, studies, keywords, prompts and artists.
* [Sunny's Tips & Tricks](https://docs.google.com/document/u/1/d/1K6EqcsRut0InU-8jB0yOvBMGesf5Dndg5FwyuaYLqNc/mobilebasic) - Sunny's SD Tips & Tricks Google Doc with lots of visual comparisons and useful information.
* [AI Image Generator Comparison](https://petapixel.com/2022/08/22/ai-image-generators-compared-side-by-side-reveals-stark-differences/) - A visual comparison between Dall-e, Stable Diffusion and Midjourney by PetaPixel.com.
* [Getting great results at Stable Diffusion](https://old.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/) - A guide on generating images that don't suck.

### Studies
* [Modifier Studies](https://proximacentaurib.notion.site/2b07d3195d5948c6a7e5836f9d535592?v=b5b75a67cc52483c9965cfc141f6f582) - Visual study of popular modifiers/keywords.
* [Artist Studies](https://remidurant.com/artists/#) - Visual study of various artists.

### Extending Functionality
* [Building a SD Discord Bot](https://replicate.com/blog/build-a-robot-artist-for-your-discord-server-with-stable-diffusion) - A tutorial on building a Stable Diffusion Discord bot using Python.


## Community Resources
* [1 week of Stable Diffusion](https://multimodal.art/news/1-week-of-stable-diffusion) - A curated list of Stable Diffusion services, adaptations, user interfaces and integrations.
* [pharmapsychotic.com/tools](https://pharmapsychotic.com/tools.htm) - A curated list of Tools and Resources for AI Art, including but not limited to Stable Diffusion.

## Social Media
* [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/) - Stable Diffusion Subreddit.

## Online Services implementing Stable Diffusion
* 🖊️💵 [Dream Studio](http://beta.dreamstudio.ai/) - Online art generation service by StabilityAI, the creators of Stable Diffusion. Similar to services like DALL-E or Midjourney, this operates on a credit model with a free allowance of credits given to signed up users on a monthly basis.
* 🖊️💵 [dream.ai](https://www.dream.ai/) - Online art generation service by Wombo.ai (mobile apps available).




