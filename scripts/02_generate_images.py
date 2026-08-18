import torch
from diffusers import FluxPipeline
from PIL import Image
import os
import json
from pathlib import Path
import random
from tqdm import tqdm
from contextlib import redirect_stdout

import transformers
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="The following part of your input was truncated because CLIP can *")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

IMAGE_SIZE = 512
BATCH_SIZE = 4

print("Redirecting output to log_image_gen.txt...")

with open("log_image_gen.txt", "w") as log_file, redirect_stdout(log_file):
    print("Attempting to load FLUX.1-dev with 8-bit quantization...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True
    )
    print("FLUX.1-dev loaded with 8-bit quantization.")

    print("Enabling model CPU offload...")
    pipe.enable_model_cpu_offload()

    print("Enabling VAE tiling and slicing...")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    def select_random_color(colors):
        return random.choice(colors)

    def select_random_sex():
        different_genders = ['man', 'woman']
        return random.choice(different_genders)

    def create_images_for(color, negative_prompt, scene_descriptions, output_folder, batch_size):
        style_prompt = f"Ultra-realistic photo of a {select_random_sex()} wearing a {color} T-shirt,"
        prompts = [style_prompt + desc for desc in scene_descriptions]
        negative_prompts = [negative_prompt] * len(prompts)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_negative_prompts = negative_prompts[i:i + batch_size]

            images = pipe(
                prompt=batch_prompts,
                negative_prompt=batch_negative_prompts,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE
            ).images

            for j, img in enumerate(images):
                img_path = f"{output_folder}/{i + j + 1}.png"
                img.save(img_path)
                print(f"Saved: {img_path}")

    colors = [
        "dark green", "gray", "navy blue", "light blue", "blue", "red",
        "dark red", "green", "maroon", "white", "black", "orange", "yellow"
    ]
    negative_prompt = (
        "empty room, no humans, lack of people, absence of persons, deserted, blurry, low quality, "
        "unusual face, bad posture, bad anatomy, animated face, bad face, bad legs, "
        "unusual posture, dark room, darkness"
    )

    input_folder = "data/enhanced_text"
    output_parent = "data/images"
    Path(output_parent).mkdir(parents=True, exist_ok=True)

    input_files = list(Path(input_folder).glob('*.json'))
    print(f"\nGenerating images for {len(input_files)} files")
    counter = 0
    for json_file in tqdm(input_files, desc="Processing episodes"):
        counter += 1
        print(f"\nProcessing file: {json_file.name}")
        with open(json_file, 'r') as f:
            data = json.load(f)

        scene_descriptions = [
            scene['scene_description']
            for scene in data['episode_scenes']
        ]

        output_subdir = Path(output_parent) / json_file.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        create_images_for(
            color=select_random_color(colors),
            negative_prompt=negative_prompt,
            scene_descriptions=scene_descriptions,
            output_folder=output_subdir,
            batch_size=BATCH_SIZE
        )
        print(f"done with {counter} files!")

    print(f"\nProcessed {len(input_files)} files")
