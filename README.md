# Awesome Stable-Diffusion

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a list of software and resources for the [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) AI model.

- 🖊️ marks content that requires sign-up or account creation for a third party service outside GitHub.
- ❓ marks content with unclear licensing conditions (e.g. lack of a license on Github)
- 💵 marks [Non-Free](https://en.wikipedia.org/wiki/Free_software) content: commercial content that may require any kind of payment.

Due to the fast-moving nature of the topic, entries in the list may be removed at an expedited rate until the ecosystem matures.

See [Contributing](.github/CONTRIBUTING.md).

## TL;DR

The easiest way to get started for most people is to pick one of the available [GUIs](https://github.com/awesome-stable-diffusion/awesome-stable-diffusion/blob/main/README.md#guis) based on the desired platform and follow it's installation instructions.

Alternatively, most of the more developed forks (such as InvokeAI) come with their own user interfaces.

To get up and running, one usually needs:
- A GUI or CLI based installation on a supported platform (usually NVIDIA GPU but support for Apple Silicon, AMD GPUS and even CPU exists in some branches)
- The weights (.ckpt files) for the model downloaded from Huggingface
- Sufficent RAM/VRAM available on the system.


## Official Resources
* [Stable Diffusion 2.1 Release Notes](https://stability.ai/blog/stablediffusion2-1-release7-dec-2022) - Stable Diffusion 2.1 release notes.
* **https://github.com/Stability-AI/stablediffusion - Official Stability AI Stable Diffusion 2.1 release.**
* [CompVis/Stable Diffusion](https://github.com/CompVis/stable-diffusion) - The official release of Stable Diffusion including a CLI, an AI-based Safety Classifier, which detects and suppresses sexualized content, and all the necessary files to get running.
* [stability-AI/stability-sdk](https://github.com/stability-AI/stability-sdk) - The official SDK used to build python applications integrated with StabilityAI's cloud platform instead of hosting the model locally. Operation requires an API Key (🖊️💵).
* [Public Release Announcement](https://stability.ai/blog/stable-diffusion-public-release) - StabilityAI's announcement about the public release of Stable Diffusion.
* 🖊️ [Official Discord](https://discord.gg/stablediffusion) - The official Stable Diffusion Discord by StabilityAI.
* [laion-aesthetic](https://laion-aesthetic.datasette.io/laion-aesthetic-6pls/images) - The dataset used train stable diffusion, useful for querying to see if a concept is represented. 

## Actively Maintained Forks and Containers

All forks listed here add additional features and optimisations and are generally faster than the original release, as they keep the model in memory rather than reloading it after every prompt. Most forks seem to remove the Safety Classifier which may present a risk if used to provide public-facing services, such as Discord bots. The two most popular forks at this point are Automatic1111 and InvokeAI

* [AbdBarho/stable-diffusion-webui-docker](https://github.com/AbdBarho/stable-diffusion-webui-docker) - Easy Docker setup for SD with multiple user-friendly UI options including [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), [sd-webui/stable-diffusion-webui](https://github.com/sd-webui/stable-diffusion-webui) and [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI).

* [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Likely the fastest moving, most feature rich branch at the moment. Gradio based UI with extensive features such as in and outpainting, previews, xy plots, upscaling, clip-interrogation, textual inversion, negative prompting, a variety of upscaling features, training, checkpoint merging and switching capabilities and more. Comes with a handy install script that takes care of most dependencies and addons. 

* [basujindal/stable-diffusion](https://github.com/basujindal/stable-diffusion) - "Optimized Stable Diffusion"—a fork with dramatically reduced VRAM requirements through model splitting, enabling Stable Diffusion on lower-end graphics cards; includes a GradIO web interface and support for weighted prompts. 

* [bes-dev/stable_diffusion.openvino](https://github.com/bes-dev/stable_diffusion.openvino) - A fork for running the model using a CPU compatible with OpenVINO.
* [DreamArtist](https://github.com/7eu7d7/DreamArtist-stable-diffusion) - With just one training image DreamArtist learns the content and style in it, generating diverse high-quality images with high controllability. Embeddings of DreamArtist can be easily combined with additional descriptions, as well as two learned embeddings. (standalone version) 
* [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) - Speed focused fork with Dreambooth integration. 
* [Gauss](https://github.com/justjake/Gauss) - Native MacOS Stable Diffusion App.
* [imaginAIry](https://github.com/brycedrennan/imaginAIry) - Pythonic generation of stable diffusion images. Unique in that it supports complex text-based masking. Has an interactive CLI, upscaling, face enhancement, tiling, and other standard features. No GUI.

* [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI) - (formerly known as lstein/stable-diffusion) - Very active fork adding a conversational CLI, basic web interface and support for GFPGAN, ESRGAN, Codeformer, weighted prompts, prompt blending, negative prompting, img2img, tiling, [textual-inversion](https://textual-inversion.github.io/) as well as inference on Apple M1.

* [KerasCV StableDiffusion](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/) - High performance implementation of stable diffusion on KerasCV.

* [NickLucche/stable-diffusion-nvidia-docker](https://github.com/NickLucche/stable-diffusion-nvidia-docker) - Multi (Nvidia) GPU capable docker setup of SD
* [replicate/copg-stable-diffusion](https://github.com/replicate/cog-stable-diffusion) - [Cog machine learning container](https://github.com/replicate/cog) of SD v1.4.
* [stable-diffusion-jupyterlab-docker](https://github.com/pieroit/stable-diffusion-jupyterlab-docker) - A Docker setup ready to go with Jupyter notebooks for Stable Diffusion. 

* [runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion) - Stable Diffusion Branch by [RunwayML](https://runwayml.com) with specifically trained inpainting model for high quality inpainting.

### Potentially Stale/Less active branches

These branches have not been updated in more than a months and are kept here for historic purposes
* [neonsecret/stable-diffusion](https://github.com/neonsecret/stable-diffusion) - Fork focusing on bigger resolutions with less vram at the expense of speed, automatically adjusting to the GPUs abilities. Also includes upscaling, facial restoration via CodeFormer and [custom UI](https://github.com/neonsecret/stable-diffusion/blob/main/GUI_TUTORIAL.md)



## Checkpoints and Weights

Checkpoints (.ckpt files) must be separately downloaded and are required to run Stable Diffusion. The latest stability ai release is 2.1. The last model containing NSFW concepts was 1.5 released by RunwayML.

For 2.x releases, there is a 768x768px resolution capable model trained off the base model (512x512 pixels). You need **one** of these models to use stable diffusion and generally want to chose the latest one that fits your needs.


* 🖊️ **[sd-2.1 Model Cards (768x768px)](https://huggingface.co/stabilityai/stable-diffusion-2-1) - Model Cards/Weights for Stable Diffusion 2.1 (diffusion, upscaling and inpainting checkpoints)**
* 🖊️ [sd-2.1 Base Model Cards (512px512px)](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) - Model Cards/Weights for Stable Diffusion 2.1 (diffusion, upscaling and inpainting checkpoints)
https://huggingface.co/stabilityai/stable-diffusion-2-1-base
* 🖊️ [sd-2.0 Model Cards](https://huggingface.co/StabilityAI) - Model Cards/Weights for Stable Diffusion 2.0 (diffusion, upscaling and inpainting checkpoints)
* 🖊️ [sd-v1-5 from RunwayML](https://huggingface.co/runwayml/stable-diffusion-v1-5?) - Stable Diffusion 1.5 Checkpoint released by runwayML. 
* 🖊️ [Official Model Card 1.5](https://huggingface.co/CompVis/stable-diffusion) - Official Stability AI Model Card on Hugging Face with all versions of the model. Download requires sign-in and acceptance of terms of service.
  * [stable-diffusion-v-1-4-original.chkpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) - The original 1.4 model's card
* [RealESRGAN Models](https://github.com/xinntao/Real-ESRGAN/releases/) - Download location for the latest RealESRGAN models required to use the upscaling features implemented by many forks. Different models exist for realistic and anime content. Please refer to the fork documentation to identify the ones you 
* [sd-v1-5-inpainting from RunwayML](https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt) - Checkpoint optimized for inpainting on SD 1.5, released by runwayML.

## Online Demos and Notebooks

* [HuggingFace/StabilityAI](https://huggingface.co/spaces/stabilityai/stable-diffusion) - The official demo on HuggingFace Spaces.
* 🖊️💵 [Offical Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) - The official, optimized colab for running SD on Google Cloud. Due to VRAM requirements required Colab Pro to create images.
* [andreasjansson/stable-diffusion-animation](https://replicate.com/andreasjansson/stable-diffusion-animation) - Animate between prompts.
* [Deforum](https://github.com/deforum/stable-diffusion) - Advanced notebook for Stable Diffusion with 2D, 3D, Video Input, and Interpolation animations. Includes inpainting, prompt batching, and more.
* 🖊️ [Stable Diffusion Interpolation](https://colab.research.google.com/drive/1EHZtFjQoRr-bns1It5mTcOVyZzZD9bBc?usp=sharing) - AA simple implementation of generating N interpolated images (Colab)
* [huggingface/diffuse-the-rest](https://huggingface.co/spaces/huggingface/diffuse-the-rest) - Diffuse the Rest - img2img from simple sketches or uploaded images.

## Complementary Models and Tools

Tools and models for use in conjuction with Stable Diffusion
* [Civitai Model Repository](https://civitai.com/) - Repository for custom models and textual inversion embeddings.
* [Prompt to Prompt](https://github.com/bloc97/CrossAttentionControl) - Unofficial Implementation of Cross-attention-control for prompt to prompt image editing. 
* [sd-prompt-graph](https://github.com/trevbook/sd-prompt-graph) - This is a React-based curve editor GUI for prompt interpolation animations made with Stable Diffusion.
* [DAAM](https://github.com/castorini/daam) - Diffusion attention attribution maps, generating heatmaps modelling the impact of specific terms and tokens in the prompt on the final diffusion result.
* [sd-concepts-library](https://huggingface.co/sd-concepts-library) - A library of user created [textual-inversion](https://textual-inversion.github.io/) embeddings to add new concepts to stable diffusion

## Training


* [Dreambooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) - Implementation of [Google's DreamBooth](https://arxiv.org/abs/2208.12242) for stable diffusion, allowing fine-tuning of the model for specific concepts.

* [LORA](https://github.com/cloneofsimo/lora) - LORA (Low-rank Adaptation) for Faster Text-to-Image Diffusion Fine-tuning (UNET+CLIP), a faster, more efficient Fine tuning method requiring less RAM than Dreambooth. [Huggingface demo here](https://huggingface.co/spaces/ysharma/Low-rank-Adaptation).

* [Stable Tuner](https://github.com/devilismyfriend/StableTuner) - Stable Tuner, an easy to install Dreambooth trainer with a very comfortable user interface.

* [Stable Diffusion Trainer](https://github.com/CCRcmcpe/scal-sdt) - Stable Diffusion trainer with scalable dataset size and hardware usage. Requires 10G of VRAM.

* [textual-inversion](https://github.com/rinongal/textual_inversion) - Addition of personalized content to Stable Diffusion without retraining the model ([Paper](https://textual-inversion.github.io/), [Paper2](https://dreambooth.github.io/)). 

* [Stable Dreamfusion](https://github.com/ashawkey/stable-dreamfusion) - Text to 3D dreamfusion implementation based on stable diffusion. 



### GUIS

Most of these GUIS, unless mentioned otherwise in their documentation, include stable-diffusion.

* 🖊️💵 [Auto SD Workflow](https://www.patreon.com/auto_sd_workflow) - A UI for [lstein/stable-diffusion](https://github.com/lstein/stable-diffusion)'s dream.py with optimized UX for large-scale/production workflow around image synthesis. [Video Walkthrough](https://vimeo.com/748114237).
* [Carefree Creator (local version)](https://github.com/carefree0910/carefree-creator#webui--local-deployment) - User friendly GUI with a creator/artist centric workflow.      
* [cmdr2/stable-diffusion-ui](https://github.com/cmdr2/stable-diffusion-ui) - Another, simple to use UI for windows and Linux.

* [DiffusionBee](https://github.com/divamgupta/diffusionbee-stable-diffusion-ui) - Self contained binary app for MacOS.
* 🖊️ [DiffusionUI](https://github.com/leszekhanusz/diffusion-ui) - web UI made with Vue.js inspired by Dall-e using [diffusers](https://github.com/huggingface/diffusers), perfect for inpainting. [Video demo](https://www.youtube.com/watch?v=AFZvW5qURes)
* 🖊️ [KIRI.ART](https://kiri.art/) (formerly SD-MUI) - mobile-first PWA with multiple models (incl. waifu diffusion).  Run free locally or use free & paid credits on the live site.  Built with React + MaterialUI.  ([Source Code](https://github.com/gadicc/stable-diffusion-react-nextjs-mui-pwa)) `MIT License` `TypeScript`
* 💵 [NMKD GUI](https://nmkd.itch.io/t2i-gui) - Windows UI, fully featured. Closed source. Pick your own price.
* [sd-webui/stable-diffusion-webui](https://github.com/sd-webui/stable-diffusion-webui) - Very active fork with optional, highly featureful Gradio UI and support for txt2img, img2img inpainting, GFPGAN, ESRGAN, weighted prompts, optimized low memory version, optional [textual-inversion](https://textual-inversion.github.io/) and more.
* [Stable Diffusion GRisk GUI]([https://grisk.itch.io/stable-diffusion-gui) - Windows GUI binary for SD. Closed source so use at your own risk.
* [Stable Diffusion Infinity](https://github.com/lkwq007/stablediffusion-infinity) - A proof of concept for outpainting with an infinite canvas interface. (requires powerful GPU).
* [Unstable Fusion](https://github.com/ahrm/UnstableFusion) - A Stable Diffusion desktop frontend with inpainting, img2img and more
* [stable-diffusion-webui-docker](https://github.com/AbdBarho/stable-diffusion-webui-docker) - A docker based frontend integrating the most popular forks.
* 🖊️ [Synesthetic](https://synesthetic.ai) - Prompt search engine and generator focused on remixing
* [StableDiffusionUnityTools](https://github.com/KonH/StableDiffusionUnityTools) - Unity Editor assets generation

### Upscaling
* [BasicSR](https://github.com/XPixelGroup/BasicSR) - Open-source upscaling and restoration toolbox supporting several models.
* [BSRGAN](https://github.com/cszn/BSRGAN) - BSRGAN—another upscaling solution specialized in upscaling degraded images.
* [Cupscale](https://github.com/n00mkrad/cupscale) - GUI for running various upscaling models, including ESRGAN and RealESRGAN.
* [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) - ESRGAN Upscaling (2x, 4x) and content restoration. Python: ```pip install realesrgan```
* [jquesnelle/txt2imghd](https://github.com/jquesnelle/txt2imghd) - A port of the GOBIG mode from progrockdiffusion, providing high quality upscaling on top of txt2img.
* [Upscale Wiki Model Database](https://upscale.wiki/wiki/Model_Database) - Wiki and model database for checkpoints compatible with ESRGAN et al.

### Content Restoration
* [lama-cleaner](https://github.com/Sanster/lama-cleaner) - Content aware AI inpainting tool useful for removing unwanted objects or defects from images. Python: ```pip install lama-cleaner```
* [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face Restoration GAN included in several forks for automatically fixing the face deformation commonly found in SD output.
* [CodeFormer](https://github.com/sczhou/CodeFormer) - Another Face Restoration model ([Paper](https://arxiv.org/abs/2206.11253)).

### Task Chaining
* [chaiNNer](https://github.com/joeyballentine/chaiNNer) - Graphical node-based editor for chaining image processing tasks.
* [ai-art-generator](https://github.com/rbbrdckybk/ai-art-generator) - AI art generation suite combining Stable Diffusion and other models for high volume art generation.
* [dfserver](https://github.com/huo-ju/dfserver) distributed backend AI pipeline server for building self-hosted distributed GPU cluster to run the Stable Diffusion and various AI image or prompt building model.

### Prompt Building

Prompts are the instructions given to diffusion models to manipulate their output. 

* [Stable diffusion prompt book](https://openart.ai/promptbook) - OpenAI's stable diffusion prompt book, a very comprehensive resource on prompt engineering.

* [ai-art.com/modifiers](https://www.the-ai-art.com/modifiers) - A visual reference guide for keywords.
* [aipromptguide.com](https://aipromptguide.com) - Visual Database of styles, modifier, artists and persons
* 🖊️ [aiTransformer Prompt Builder](https://aitransformer.net/PromptBuilder) - Help build text prompts intuitively using an Image To Text tool to generate detailed text descriptions from image, and 800+ special Textual Inversion terms and often-used generic terms with sample images.
* [krea.ai](https://www.krea.ai/) - Prompt search engine that also recommends similar prompts to the one that you click on. 
  - 🖊️ With account creation, you can like and save prompts in your own collections.
* [Lexica.art](https://lexica.art/) - A searchable, visual database of images and the prompts settings used to create them.
* [pharmapsychotic/clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator) - Jupyter notebook uses CLIP models to suggest a prompt for images similar to a given image ([Demo](https://replicate.com/methexis-inc/img2prompt)).
* 🖊️[Phraser](https://phraser.tech/) - A visual prompt builder drawing on a database of examples. (Requires account creation)
* 🖊️[Prompthero](https://prompthero.com/) = Another visual prompt builder and reference library.
* [PromptoMania](https://promptomania.com/) - A visual prompt construction tool.
* [rom1504/clip-retrieval](https://github.com/rom1504/clip-retrieval) - Searches for prompt keywords in the datasets used in training Stable Diffusion and other models ([Online GUI](https://rom1504.github.io/clip-retrieval/)). Some GUIS like Automatic1111 include this functionality.
* [Stable Diffusion Prompt Generator](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion) - Gives suggestions for improving a given text prompt.
* [PromptSearch](https://pagebrain.ai/promptsearch/?q=&page=1) - Yet another Stable Diffusion search engine but with public API
* [Same Energy](https://same.energy/) - A visual search engine that returns images that have the same 'energy'. 
* [PublicPrompts](https://publicprompts.art/) - *Collection* of PublicPrompts

### Specialized Usecases

* [dream-textures](https://github.com/carson-katri/dream-textures) - A blender addon leveraging stable diffusion for texture creation.
* [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) - An open source implementation of Google's text-to-3D dreamfusion paper with imagegen replaced by stable diffusion. 

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
* [Sunny's Tips & Tricks](https://docs.google.com/document/d/1R2UZi5G-DXiz2HcCrfAFLYJoer_JPDEoZmV7wy1tEz0/mobilebasic) - Sunny's SD Tips & Tricks Google Doc with lots of visual comparisons and useful information.
* [AI Image Generator Comparison](https://petapixel.com/2022/08/22/ai-image-generators-compared-side-by-side-reveals-stark-differences/) - A visual comparison between Dall-e, Stable Diffusion and Midjourney by PetaPixel.com.
* [Getting great results at Stable Diffusion](https://old.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/) - A guide on generating images that don't suck.
* [Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) - An illustrated primer on how Stable Diffusion works.
* [Practical deep learning for coders](https://course.fast.ai/) - high quality course by fast.ai aimed at coders that covers many aspects of deep learning, including stable-diffusion.
* [Top SD Artists](https://www.urania.ai/top-sd-artists) - Searchable list of artists known by Stable Diffusion with example images.


### Studies
* [Modifier Studies](https://proximacentaurib.notion.site/2b07d3195d5948c6a7e5836f9d535592?v=b5b75a67cc52483c9965cfc141f6f582) - Visual study of popular modifiers/keywords.
* [Artist Studies](https://remidurant.com/artists/#) - Visual study of various artists.


### Extending Functionality
* [Building a SD Discord Bot](https://replicate.com/blog/build-a-robot-artist-for-your-discord-server-with-stable-diffusion) - A tutorial on building a Stable Diffusion Discord bot using Python.


## Community Resources
* [1 week of Stable Diffusion](https://multimodal.art/news/1-week-of-stable-diffusion) - A curated list of Stable Diffusion services, adaptations, user interfaces and integrations.
* [pharmapsychotic.com/tools](https://pharmapsychotic.com/tools.htm) - A curated list of Tools and Resources for AI Art, including but not limited to Stable Diffusion.
* [Stable Diffusion Resources](https://stackdiary.com/stable-diffusion-resources/) - A thorough resource for answering pressing questions about Stable Diffusion, including guides, tutorials, and best software.

## Social Media
* [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/) - Stable Diffusion Subreddit. (Semi-official)
* [r/sdforall](https://www.reddit.com/r/sdforall) - SDForAll
* [Diffusion Pulse](https://88stacks.com/the-stable-diffusion-newsletter/) - Weekly Stable Diffusion newsletter

## Plugins for third party apps
* [Blender Plugin](https://github.com/benrugg/AI-Render) - Plugin for the free 3D modelling software Blender
* [Gimp Plugin](https://github.com/blueturtleai/gimp-stable-diffusion) - Gimp Plugin. 
* [Krita Plugin](https://github.com/nousr/koi) - A krita and Gimp SD plugin
* [Krita 5.0 Plugin](https://github.com/sddebz/stable-diffusion-krita-plugin) - Another Krita Plugin based on the popular Automatic1111 fork. 
* 🖊️ [Photoshop Plugin](https://christiancantrell.com/#ai-ml) - SD for Photoshop (Adobe Exchange)


## Commercial SaaS and apps implementing Stable Diffusion
* 🖊️💵 [AI Art Generator (IOS)](https://apps.apple.com/app/apple-store/id1644315225?pt=94765902&ct=github&mt=8) - iOS App to generate art using Stable Diffusion.
* [aiimagegenerator.org](https://www.aiimagegenerator.org/) - Free online art generator for generating images using Stable Diffusion. Supports txt2img and img2img generation and inpainting.
* 🖊️ [aiTransformer](https://aitransformer.net) - Free online image/video transformation tools, including Super Stylizer based on Stable Diffusion, supports txt2img, img2img and img2txt (clip-interrogator).
* 🖊️💵 [Barium.ai](https://barium.ai/) - Generate PBR (physics based rendering) textures from text. Free and paid plans.
* 🖊️💵 [Canva text-to-image](https://www.canva.com/apps/text-to-image-(beta) ) - Text-to-image (beta) service from Canva
* 🖊️💵 [Dream Studio](http://beta.dreamstudio.ai/) - Online art generation service by StabilityAI, the creators of Stable Diffusion. Similar to services like DALL-E or Midjourney, this operates on a credit model with a free allowance of credits given to signed up users on a monthly basis.
* 🖊️💵 [dream.ai](https://www.dream.ai/) - Online art generation service by Wombo.ai (mobile apps available).
* 🖊️💵 [GimmeAPIs](https://gimmeapis.com) - Third party REST API Stable Diffusion service. 
* 🖊️💵 [Image Computer](https://image.computer) - Easy-to-use service aimed at non-technical people (comes with free trial credits)
* 🖊️💵 [Neural.love](https://neural.love/ai-art-generator) - Another online art generator with generous free credits as of Oct 2022. 
* 🖊️💵 [replicate.com stable diffusion](https://replicate.com/stability-ai/stable-diffusion) - Another SaaS offering for Stable Diffusion.
* 🖊️💵 [Starry AI (IOS)](https://apps.apple.com/us/app/starryai-create-art-with-ai/id1580512844) - Another IOS app offering stable diffusion with preset art styles. 
* 🖊️ [Stable Horde](https://stablehorde.net/) - Distributed stable diffusion cluster (think folding@home) with web, discord and telegram interfaces where joining with your GPU gives you priority. 
* 🖊️💵 [Stable Diffusion as API](https://stablediffusionapi.com/) - Third party REST API into table Diffusion service.

## The Attic - previous links, retired for inactivity

* [lowfuel/progrock-stable](https://github.com/lowfuel/progrock-stable) - Fork with optional Web GUI and a different approach to upscaling (GoBIG/ESRGAN)
  * [txt2imghd](https://github.com/jquesnelle/txt2imghd) - Fork of progrock diffusion that creates detailed, higher-resolution images by first generating an image from a prompt, upscaling it, then running img2img on smaller pieces of the upscaled image, and blending the results back into the original image.
