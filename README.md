# Creating object detection model for edge AI using synthetic training images created with Nvidia Cosmos-predict1

## Preface
Nvidia has released a family of World Fundation Models (WFMs). All models have to some degree overlapping areas of use and can be combined for an efficient pipeline. This article is about **Cosmos Predict**, **Cosmos Transfer** and **Cosmos Reason** are covered in separate articles.

- [**Cosmos Transfer**](https://github.com/nvidia-cosmos/cosmos-transfer1) can amplify text and video input to create variations of environment and lighting conditions for training data for visual AI. Multiple input signals enable control of physics-aware world generation. We can compose a 3D scene in NVIDIA Omniverse and have Cosmos Transfer create the variation needed to train robust models for visual computing.
- [**Cosmos Reason**](https://github.com/nvidia-cosmos/cosmos-reason1) is capable of reasoning based on spatial and temporal understanding of multimodal input. It can interpet what a sensor is seeing and predict consequences. It can also be a helpful tool to automatically evaluate the quality of synthetic training data.
- [**Cosmos Predict**](https://github.com/nvidia-cosmos/cosmos-predict2) can create training data, both single image and video clip, for visual AI based on text- and image input.

All models are pre-trained for autonomous vehicle and robotic scenarios, and support post-training for specific use cases.

## Prerequisites
- Python
- Optional: Access to Nvidia GPU with at least 32 GB VRAM
- Optional: Nvidia Omniverse Isaac Sim

## What to expect
This tutorial shows how to use [**Nvidia Cosmos-predict2**](https://github.com/nvidia-cosmos/cosmos-predict2), released June 2025 to generate physics aware synthetic images for training models for visual computing. The tutorial will show hands-on approaches on how to generate text- and image-prompted images and videos, automatically segment and label objects of interest and prepare data for model training in [Edge Impulse Studio](https://edgeimpulse.com/).

We'll start with a comparison of other methods and models, walk through an easy to perform web-based demo, dive into self-hosting the smaller models, use local AI-labeling models, take full advantage of single-image generation with batching and prompt enrichment with LLMs, before we move to the larger models and video generation.

## Alternative methods
The traditional method to create training data for visual computing is to manually capture images in the field and to label all objects of interest. As this can be a massive undertaking and it can be challenging to cover special circumstances, [synthetic training data generation](https://docs.edgeimpulse.com/experts/readme/featured-machine-learning-projects/surgery-inventory-synthetic-data) has become a popular supplement. Even with real-time path-tracing and domain randomization methods capable of generating highly controlled masses of data, covering sufficient variation to create robust models can be labor intensive.

### Generating Synthetic Images with GPT-4 (DALL-E)
![](images/EI-synthetic-image.webp "Generating Synthetic Images with GPT-4 (DALL-E)")
Realistic training images can effortlessly be [generated using diffusion models such as GPT-4 Image/DALL-E](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/synthetic-data#generating-synthetic-images-with-gpt-4-dall-e).
These images needs to be labelled manually or by one of the many [AI assisted labelling methods](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/ai-labeling).

#### Note about AI labeling
> Models used for labeling are trained on vast collections of manually labeled objects. They usually work great for common objects. For labeling uncommon objects, or even objects that are new, e.g. new commercial products, these models fall short. In these cases **Cosmos Transfer** offer a novel architecture as label data from a 3D scene can be reused. This is due to the highly controllable nature of **Cosmos Transfer**, where certain aspects of the input signal data will be respected when a new variation of a Omniverse-created video sequence is generated. This can be achieved in a number of ways thanks to multimodal control signal input. In short - a new video clip is generated, for instance with a different background, but objects of interest stay at the same screen-space position as in the input clip. Bounding boxes therefore remain valid for both clips. **Cosmos Transfer** is not covered in this article, but this should be an important feature to consider when choosing among the different Cosmos models.

Another variation is to use video generation, such as **OpenAI Sora**, **Google Gemini Veo 3** or [any high performing video generators](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard). Text, image and video input can prompt generation of video sequences. The advantage of generating video clips over still images is that we get the objects of interest in many different angles with minimal effort. For object detection, these sequences can then be split into individual still frames, before labelling.

Training image generation with these types of models have the disadvantages of being hard to control. Objects of interest will tend to morph out of form and strict camera control is hard to achieve. Generating training images of novel objects, say a new commercial product, is also currently hard to accomplish. These models have been trained for generalization, from movie stars eating pasta to presidents promoting unconventional vacation resorts.

Some comparisons:

```
The video is a first-person perspective from the viewpoint of a large, humanoid robot navigating through a factory or industrial setting. The robot is equipped with a camera mounted on its head, providing a view of the surroundings. The environment is a spacious, well-lit industrial area with high ceilings and large windows, allowing natural light to flood the space. The floor is made of a smooth, reflective material, possibly concrete or polished metal, which enhances the robot's movement and reflections.
```

Sora:

[![Warehouse Robot Video](https://img.youtube.com/vi/NK6jpjv-des/0.jpg)](https://youtu.be/NK6jpjv-des)

Sora:

[![Warehouse Robot Video](https://img.youtube.com/vi/UAFlf6SF-_Q/0.jpg)](https://youtu.be/UAFlf6SF-_Q)

Cosmos Predict:

[![Warehouse Robot Video Cosmos-Predict1-7B-Text2World](https://img.youtube.com/vi/hWOfi2IGxbg/0.jpg)](https://youtu.be/hWOfi2IGxbg)


The Sora videos are of great fidelity, but notice how Cosmos Predict defaults to a setting suitable for edge AI scenarios. We could achieve the same results with Sora, but it would require a lot of trial and error with targeted prompting. Without API access to Sora, ability to change seed or negative prompt, methodically generating 10.000s of variations for model training is impractical.

The NVIDIA Cosmos WFMs on the other hand are trained on a curated set of driving and robotics data. The ability to supply multimodal control data and to post-train the models for custom scenarios makes them further suitable for tailored training data generation.

The Cosmos model's advantages do however come with a cost - they require a lot of compute and require a bit of insight to harness. This article will cover a few different options in getting to know the capabilities of **Cosmos Predict**.

### Method/model comparison
| Feature | Manual | Sora/Veo | Omniverse/Replicator | Cosmos Predict | Cosmos Transfer | Notes |
|---------|--------|----------|----------------------|----------------|-----------------|-------|
|Installation|N/A|N/A|Medium|Medium**|Medium**|**Requires GPU farm
|Initial domain customization|High - field work|Low|Medium|Low*|Medium*|*Cosmos needs post-training if out of foundation context, high effort|
|Iteration effort|High|High*|Medium|Low|Low|*API not generally available June 2025
|Variation effort|High|High*|Medium|Low|Low|*API not generally available June 2025, seed not accessible, negative prompt not available in Sora
|Photorealism|High|High|Medium|Medium*|Medium*|*14B models|
|Suitability for novel objects|High|Low|Medium|Low|Medium|
|Automatic labeling quality|Medium|Medium|High*|Medium|High*|*Perfect labeling from Replicator semantic tagging|

## Web demo
We can get an impression of the capabilities of Cosmos Predict with almost no effort. This approach does however severely limit options for synthesis customization. Be aware that the 20 request limit does not reset periodically. As of June 2025 **cosmos-predict2** is not available for testing at NVIDIA Build, only the previous **cosmos-predict1-7b**.
- Go to [build.nvidia.com](https://build.nvidia.com/nvidia/cosmos-predict1-7b), register or log on.
- Enter a prompt, such as:
```
A first person view from the perspective of a FPV quadcopter as it is flying over a harbor port. The FPV drone is slowly moving across the harbor, 20 meters above the ground, looking down at cargo containers, trucks and workers on the docks. The weather is clear and sunny. Photorealistic
```
- Wait about 60 seconds to see the results.

[![Cosmos-Predict1-7B-Text2World Docks sunny](https://img.youtube.com/vi/V2hm48HPlns/0.jpg)](https://youtu.be/V2hm48HPlns)

To download the generated video we can enter the following JavaScript into Console in the browser developer tools:

```javascript
// Grab the video element from DOM
const vid = document.getElementById('cosmos-web-component-result-video');

// Fetch the blob behind the blob: URL
fetch(vid.src)
  .then(res => res.blob())
  .then(blob => {
    // Create a real object URL for download
    const downloadUrl = URL.createObjectURL(blob);

    // Create a temporary <a> with download attribute
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = downloadUrl;

    // Choose a filename (you can change .webm to .mp4)
    a.download = 'video.webm';

    document.body.appendChild(a);
    a.click();

    // Cleanup
    URL.revokeObjectURL(downloadUrl);
    document.body.removeChild(a);
  })
  .catch(console.error);
```

Now try a slight variation to the text prompt to get instant weather variation. With a crafted 3D scene for synthetic image generation with domain randomization, achieving this would require a lot of work: 

>A first person view from the perspective of a FPV quadcopter as it is flying over a harbor port. The FPV drone is slowly moving across the harbor, 20 meters above the ground, looking down at cargo containers, trucks and workers on the docks. The weather is cloudy, it is snowing lightly, the ground is covered in a 1 cm layer of snow. Wheel tracks are visible in the snow. Photorealistic

[![Cosmos-Predict1-7B-Text2World Docks Snow](https://img.youtube.com/vi/kq9WroFR4X8/0.jpg)](https://youtu.be/kq9WroFR4X8)

With each result we get a refined prompt that might be useful for further enhancements:

>In a breathtaking aerial journey, we soar above a bustling harbor, captured through the lens of a cutting-edge first-person view (FPV) quadcopter. The camera glides smoothly at a steady 20 meters above the ground, revealing a sprawling landscape of vibrant cargo containers stacked neatly on the docks, their bright colors contrasting against the deep blue of the water. Below, a lively scene unfolds as workers in high-visibility vests coordinate the loading and unloading of containers, while trucks and forklifts crisscross the area, their movements a testament to the port's dynamic energy. The sun bathes the scene in a golden-hour glow, casting long shadows that dance across the ground, while the clear sky enhances the vivid hues of the containers and the shimmering water. This cinematic experience, enhanced by dynamic color grading and a steady, immersive perspective, invites viewers to feel the thrill of flight and the rhythm of the port's industrious life.

Another slight prompt change can dramatically change the output:

>View directly underneath a first-person quadcopter as it is flying over a harbor port. The view is slowly moving across the harbor, 10 meters above the ground, looking straight down at cargo containers, trucks and workers on the docks. The weather is cloudy, it is raining heavily, the ground is wet. Photorealistic

[![Cosmos-Predict1-7B-Text2World Docks rain](https://img.youtube.com/vi/68v7ypUYPfw/0.jpg)](https://youtu.be/68v7ypUYPfw)

We can easily achieve combined environmental factors that would require a lot of work if this was created as a 3D scene in Omniverse.

>View directly underneath a first-person quadcopter as it is flying over a harbor port. The view is slowly moving across the harbor, 10 meters above the ground, looking straight down at cargo containers, trucks and workers on the docks. The weather is clear, it is midnight and dark. The ground and objects are illuminated by a full moon and powerful site flood lights. It is raining heavily, the ground is wet. Photorealistic

[![Cosmos-Predict1-7B-Text2World Docks rain dark](https://img.youtube.com/vi/9oNFXRr7Pf4/0.jpg)](https://youtu.be/9oNFXRr7Pf4)

Extract stills
- Install FFmpeg (linux): 
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```
- Extract still images
  ```bash
  ffmpeg -i cosmos-predic-web-docks-sunny.mp4 ../docks-training-data/docks-sunny-frame_%03d.jpg
  ```
- Upload images to Edge Impulse Studio
- Use AI labelling
![](images/EI-AI-label.png "AI Labeling using OWL-ViT")
- (Optional) Validate labels with GPT-4o
- Train model.
![](images/EI-train-model.png "Train model")

## Self-hosting
Now that we have aquired a sense of the capabilities of Cosmos Predict the next natural step is to host the models ourself so that we can further explore capabilities. First we need to be aware of the hardware requirements of different features:

The following table shows the GPU memory requirements for different Cosmos-Predict2 models:

| Model | Required GPU VRAM |
|-------|-------------------|
| Cosmos-Predict2-2B-Text2Image | 26.02 GB |
| Cosmos-Predict2-14B-Text2Image | 48.93 GB |
| Cosmos-Predict2-2B-Video2World | 32.54 GB |
| Cosmos-Predict2-14B-Video2World | 56.38 GB |

For optimal performance
* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* At least 32GB of GPU VRAM for 2B models
* At least 64GB of GPU VRAM for 14B models

I am running locally on a NVIDIA RTX 5090 with 32 GB VRAM. I have been able to run Text2Image 2B, but not the rest. One might get lucky and be able to run Video2World 2B parameters. We'll start by running Text2Image 2B on the 5090 and then move on to a rented GPU farm.

### Installing cosmos-predict2
Follow the [repository instructions](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/setup.md), the Docker container route is recommended to try to avoid complicated issues with Blackwell GPUs, CUDA and Torch.

```bash
git clone git@github.com:nvidia-cosmos/cosmos-predict2.git
cd cosmos-predict2
```

Get a [NVIDIA Build API key](https://build.nvidia.com/settings/api-keys).
```bash
# Pull the Cosmos-Predict2 container
export NGC_API_KEY=[your_key]
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.0
```

1. Get a [Hugging Face](https://huggingface.co/settings/tokens) access token with `Read` permission
2. Login: `huggingface-cli login`
3. The [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B) must be accepted. Approval will be required before Llama Guard 3 can be downloaded.
4. Download models. Models for running Cosmos-Predict2-2B-Text2Image alone can run up near 200GB in checkpoint space.

| Models | Link | Download Command | Notes |
|--------|------|------------------|-------|
| Cosmos-Predict2-2B-Text2Image | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image) | `python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B` |
| Cosmos-Predict2-2B-Video2World | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) | `python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B` | Download 720P, 16FPS by default. Supports 480P and 720P resolution. Supports 10FPS and 16FPS |

### Running cosmos-predic2
With any luck you will now be able to spin up the container and have the checkpoints you need:

```bash
sudo docker run --gpus all -it --rm -v "$(pwd)":/workspace -v "$(pwd)/datasets":/workspace/datasets -v "$(pwd)/checkpoints":/workspace/checkpoints nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.0
```

This should land you in a shell in the container, to get a list of options for Text2Image run

```bash
python -m examples.text2image --help
```

Now we can run single frame generation with any prompt, optionally supplying `--NEGATIVE_PROMPT="${NEGATIVE_PROMPT}"`
```bash
python -m examples.text2image \
    --prompt "${PROMPT}" \
    --model_size 2B \
    --disable_guardrail \
    --save_path outputs/my_image01.jpg
```

```bash
PROMPT="A small parking lot with different types of european cars, some spaces empty. Parking spaces line the edges of the lot, each space are of equal size. Cars entering and leaving. People walking to and from cars. Sedans, hatchbacks, station wagons, pickup trucks, white vans. Dusk, low sun, street lights."
```
![](images/parking.webp "Parking lot Text2Image-2B")

```bash
PROMPT="Warehouse loading dock with workers and activity. Trucks offloading pallets of wares using forklifts, workers with high visibility work clothes."
```
![](images/warehouse_docking.webp "Warehouse docking Text2Image-2B")
```bash
PROMPT="Inside an endless water drainage pipe. Point of view of an inspection camera."
```
![](images/drain.webp "Drainage pipe Text2Image-2B")
```bash
PROMPT="Traveling along a winding road up a mountain in the arid Spanish Andalucian region. Point of view from the eyes of a road racing bicyclist facing forward in the direction of travel. The cyclist is not visible, only her hands on the steering wheel of the bike. Sun is setting from behind, her long shadow visible in front of her on the road. A few road racing bicyclists are leading in front of her traveling in the same direction, both male and female, we see their backs. Cyclists are dressed in bright color tops, black tight cycling shorts. Light traffic."
```
![](images/bike.webp "Bikes Text2Image-2B")
```bash
PROMPT="Point of view from the eyes of a micro-mobility e-scooter rider traveling on a reserved traffic lane for bikes. Right side of the road. A standing-position electric kick scooter with a low, flat deck, narrow T-bar handle and no seat, small 8- to 10-inch wheels, rider upright with one foot forward like a skateboard stance, urban backdrop (bike lane, scootershare style), modern matte aluminum frame, integrated LED headlight, no visible license plate, no Vespa/moped body panels or seat. Ryde type e-scooter. Downtown in Oslo, Norwegian traffic signs. The rider is not visible, only her hands on the steering wheel of the e-scooter. Two e-scooter riders are leading in front of her, a male and a female, we see their backs.  Riders are dressed in pastel colored clothes."

NEGATIVE_PROMPT="Camera, motorbike helmet, motorway"
```
![](images/e-scooter.webp "e-scooter Text2Image-2B")

Most of the computing effort is spent on loading the model checkpoints from disk and loading into VRAM. We can take advantage of batching and either increment or randomize inference seed to produce more training still images without having to load the models for each image:

`batch_can_factory.json:`

```json
[
  {
    "prompt": "A conveyor belt in a factory. Orange colored doda cans are lined up on the conveyor belt, moving towards a sorting station for packaging. The soda cans have no logo. There is space between the soda cans. The conveyor belt is colored green. The factory floor is well lit, workers are surveying the factory line. Perspective is from directly above, facing down to the conveyor belt.",
    "output_image": "outputs/factory/00001.jpg"
  },
  {
    "prompt": "A conveyor belt in a factory. Blue colored doda cans are lined up on the conveyor belt, moving towards a sorting station for packaging. The soda cans have no logo. There is space between the soda cans. The conveyor belt is colored white. The factory floor is dimly lit, warm temperature yellow fluorescent roof lights, workers are surveying the factory line. Perspective is 1 meter beside the conveyor belt, facing directly toward the belt.",
    "output_image": "outputs/factory/00002.jpg"
  },
  {
    "prompt": "A conveyor belt in a factory. Rainbow colored doda cans are lined up on the conveyor belt, moving towards a sorting station for packaging. The soda cans have no logo. There is space between the soda cans. The conveyor belt is colored black. The factory floor is well lit and deserted. Perspective is 1 meter to the left and 1 meter above the conveyor belt, facing directly toward the belt.",
    "output_image": "outputs/factory/00003.jpg"
  }
  ...
]
```

```bash
python -m examples.text2image --batch_input_json batch_can_factory.json --model_size 2B  --disable_guardrail --seed 0
```

The more variations you add to the batch file, the more efficient the execution will be.

We can put this into system to generate more variations by incrementing or randomizing the seed. Unfortunately the output file name is set in the json file, so we need to make a simple program to make sure we don't overwrite for each iteration.

1. Save duplicate `batch_can_factory.json` as `batch_can_factory_template.json`.
2. Create a python program like this:

```python
import json
import subprocess
import sys

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run can factory batches.")
    parser.add_argument("--n", type=int, default=10, help="Number of iterations (default: 10)")
    args = parser.parse_args()

    NUM_ITERATIONS = args.n
    TEMPLATE = "batch_can_factory_template.json"
    BATCH_JSON = "batch_can_factory.json"

    # Load the template once to count prompts
    with open(TEMPLATE) as f:
        template_prompts = json.load(f)
    IMAGES_PER_BATCH = len(template_prompts)

    for i in range(NUM_ITERATIONS):
        # Reload fresh for each batch
        with open(TEMPLATE) as f:
            prompts = json.load(f)

        start_img_num = i * IMAGES_PER_BATCH + 1
        for j, entry in enumerate(prompts):
            img_num = start_img_num + j
            entry["output_image"] = f"outputs/factory/{img_num:05d}.jpg"

        with open(BATCH_JSON, "w") as f:
            json.dump(prompts, f, indent=2)

        cmd = [
            "python", "-m", "examples.text2image",
            "--batch_input_json", BATCH_JSON,
            "--model_size", "2B",
            "--disable_guardrail",
            "--seed", str(i),
            #"--negative_prompt", "sharp focus, ultra‑high detail, DSLR, 4K, cinematic lighting, clean edges, HDR,no noise, no artefacts"
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
```

3. Run with `python run_can_factory_batches.py --n 3`

## Notes on prompting for realism
Diffusion models like the one Cosmos Predic uses are currently achieving increadible image fidelity. When generating images for training object detection models intended to run on constrained hardware, or any type of hardware for that matter, best results are achieved by generating images of a quality that closest resembles the quality the device itself produces. This should in theory be possible to achieve by prompting e.g. 
```
photographed with a cheap 2‑megapixel CMOS sensor, f/2.2 2.2 mm lens, 80‑degree FOV, slight barrel distortion, soft focus, ISO 800 with visible sensor noise, JPEG artefacts at 80 % quality, mild purple fringing, corner vignetting, clipped highlights and crushed shadows, white‑balance drift, overall low‑contrast snapshot
```

or by adding --negative_prompt e.g. 
```
sharp focus, ultra‑high detail, DSLR, 4K, cinematic lighting, clean edges, HDR, no noise, no artefacts
```

Testing show this has little desired effect in Cosmos Predict or even with Sora, probably due to the training data. With the Cosmos models however, we have the option to fine-tune to our needs.

Also note that adding "Photorealistic" is recommended in the documentation, but shows no significance in testing.

### Advanced: LLM-augmented prompt generation for Text2Image
To further bend the limitations of batch prompting and seed iteration we can use an LLM to augment and multiply prompts. Experimentation has shown that using a web-search capable LLM with a detailed description of intended use provides useful results. A prompt might look like so:

```
I am using NVIDIA Cosmos-predict2 to generate images to train object detection models. My model needs to see many images with variations of realistic situations with drink cans in a factory and workers. I am running a loop where I am incrementing seed and for each iteration generating an image for each prompt from a batch file. Provide a table of 30 new variations by following my examples. Feel add realistic objects that could appear in a drink can factory, as long as drink cans and workers are present. I will use a open vocabulary segmentation model for AI labeling cans and workers. Output the new variations as a json-formatted list. [
  {
    "prompt": "A conveyor belt in a factory. Orange colored drink cans are lined up on a conveyor belt, moving towards a sorting station for packaging. The drink cans have no logo. There is 2 cm of space between the drink cans. The conveyor belt is colored green. The factory floor is well lit, workers are surveying the factory line. Perspective is from directly above, facing down to the conveyor belt.",
    "output_image": ""
  },
  {
    "prompt": "A conveyor belt in a factory. Blue colored soda cans...
```

[batch_can_factory_LLM_enhanced_template.json](cosmos-predict2/batch_can_factory_LLM_enhanced_template.json)

This will produce 30 images with different settings for each iteration. Then the seed is incremented or randomized and a new set of 30 images is produced, with slight variations.

Notice how some images show larger or smaller cans than expected, this is a non-issue on training object detection models as the neural networks are to a minimal degree sensitive to scaling of visual features.

![](images/can_factory_enhanced.webp "LLM-augmented prompt generation for Text2Image")

### AI labeling with Grounded Segment Anything 2
In contrast to Cosmos-Transfer we have no way of producing bounding boxes or labels of our objects of interest with these image or video clip generators. Without labels our images are useless for machine learning. Manually drawing bounding boxes and classifying tens of objects per image requires a large amount of manual labor. Many AI segmentation models are available, but stand-alone they require some input on what objects we want to label. Manually selecting the objects of interest would still require a huge effort. Thankfully it is possible to combine segmentation models with multimodal Visual Language Models. This way we can use natural language to select only the objects of interest and discard the rest of the objects the segmentation model has identified. The following will walk through using one of many Open Vocabolary Object-Detection (OVD) pipelines, [**Grounded Segment Anything 2**](https://github.com/IDEA-Research/Grounded-SAM-2) with [**DINO 1.0**](https://github.com/IDEA-Research/GroundingDINO). This repo supports many different pipeline configurations, many grounding models and can be a bit overwhelming. Grounding DINO 1.0 might not be the best performing alternative but it's open source and works for common objects. For niche objects [**DINO 1.5**](https://github.com/IDEA-Research/Grounding-DINO-1.5-API), [**DINO-X**](https://github.com/IDEA-Research/DINO-X-API) might reduce false positives by 30-40%, but these models require API-access and might be rate limited.

Note: SAM2 video predictor wants jpg-files as input.

Clone the repo.
```bash
git clone git@github.com/IDEA-Research/Grounded-SAM-2
cd Grounded-SAM-2
```

Download model checkpoints.
```bash
cd checkpoints
~/repos/Grounded-SAM-2/checkpoints$ bash download_ckpts.sh

cd ../gdino_checkpoints/
~/repos/Grounded-SAM-2/gdino_checkpoints$ bash download_ckpts.sh
```

If running on a newer Blackwell GPU you might run into the following.
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```
This can be tricky to fix, but aim for CUDA toolkit version 12.6 or newer and nightly pytorch builds before running the install script.
```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

python -m pip install --no-build-isolation -e grounding_dino
```

Included in this project repository are some examples on how to produce labeled object segmentation data, bounding boxes and images with visual annotations for quality control and tuning. 

For image input, [grounded_sam2_batch_image.py](labelling-grounded-sam2/grounded_sam2_batch_image.py)
```bash
python grounded_sam2_batch_image.py   --input_folder "can_factory/input_frames"   --output_folder "can_factory/tracking_results"   --json_output_folder "can_factory/labels"   --text_prompt "can. person. forklift." --box_threshold 0.15 --text_threshold 0.15
```

![](images/can_factory_annotated.webp "Labeled images with segmentation masks and bounding boxes")

### A note on thresholds
Grounded‑SAM‑2 inherits two key filtering knobs from Grounding DINO `--box_threshold` and `--text_threshold`. Both are cosine‑similarity cut‑offs in the range 0‑1, but they act at different stages of the pipeline:
| Stage                | What is scored?                                                                                   | Kept when score ≥ threshold              | Typical default                 |
| -------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------- |
| **`box_threshold`**  | The highest similarity between an image **region (bounding box)** and **any token** in the prompt | The whole box                            |  0.25–0.30 (case‑study configs) |
| **`text_threshold`** | The similarity between that region and **each individual token**                                  | The token becomes the label for that box |  0.25 (Hugging Face default)    |

#### How they work together
- **Box screening** The model predicts up to 900 candidate boxes. Any whose best token similarity is below box_threshold are discarded outright.

- **Label assignment** For every surviving box, each prompt token is compared to the box feature; tokens scoring below `text_threshold` are dropped, so they never appear in the final phrase.

#### Practical effect
- Raise thresholds → higher precision, lower recall. You get fewer false detections and fewer stray sub‑word labels such as `##lift` or `canperson`, due to tokenization of `can`, `person` and `forklift`, but you may miss faint objects.

- Lower thresholds → higher recall, lower precision. More boxes and more tokens survive, which can help in cluttered scenes but increases the need for post‑processing. A common starting point is `box_threshold 0.3`, `text_threshold 0.25` as recommended in the Grounded‑SAM demo scripts.

In short, `--box_threshold` decides whether a region is worth keeping at all, while `--text_threshold` decides which words (if any) are attached to that region. Tune them together to balance missed objects against noisy labels.

To be able to upload label data to Edge Impulse Studio we need to convert the output to one of the [supported formats](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/uploader#understanding-image-dataset-annotation-formats), in this case Pascal VOC XML.

[json_to_pascal_voc.py](labelling-grounded-sam2/json_to_pascal_voc.py)
```bash
python json_to_pascal_voc.py --json_dir can_factory/labels --xml_dir can_factory/pascal_voc_annotations
```

Now the dataset is ready for uploading to Edge Impulse Studio. Practically the images and label files should be located in the same directory and can be uploaded with one action in the web UI Data acquisition page, or by CLI.

Once in Edge Impulse Studio we can design an object detection neural network and evalutate the results. We can keep generating more data until classification results stop improving.
![](images/EI-evaluate-model.png "Classification results")

## 

python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py --video_path ../edgeai-synthetic-cosmos-predict1/videos/cosmos-predic-web-docks-rain.mp4 --text_prompt "drone. container. forklift. semitrailer." --OUTPUT_VIDEO_PATH ../edgeai-synthetic-cosmos-predict1/labelling-grounded-sam2/rain/rain.mp4 --SOURCE_VIDEO_FRAME_DIR ../edgeai-synthetic-cosmos-predict1/labelling-grounded-sam2/rain/custom_video_frames --SAVE_TRACKING_RESULTS_DIR ../edgeai-synthetic-cosmos-predict1/labelling-grounded-sam2/rain/tracking_results

python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py --video_path ../edgeai-synthetic-cosmos-predict1/videos/cosmos-predic-web-docks-rain.mp4 --text_prompt "drone. container. forklift. semitrailer." --OUTPUT_VIDEO_PATH ../edgeai-synthetic-cosmos-predict1/labelling-grounded-sam2/rain/rain.mp4 --SOURCE_VIDEO_FRAME_DIR ../edgeai-synthetic-cosmos-predict1/labelling-grounded-sam2/rain/custom_video_frames --SAVE_TRACKING_RESULTS_DIR ../edgeai-synthetic-cosmos-predict1/labelling-grounded-sam2/rain/tracking_results


