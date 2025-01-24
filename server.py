from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime
import torch
from PIL import Image
from io import BytesIO
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from flux_infer import inference

# Load model and related components
def initialize_model():
    base_model_path = "booksforcharlie/stable-diffusion-inpainting"
    resume_path = "zhengchong/CatVTON"

    # Download checkpoint
    repo_path = snapshot_download(repo_id=resume_path)

    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device="cuda",
    )

    return None, mask_processor, automasker

pipeline, mask_processor, automasker = initialize_model()

def infer(person_image_file,cloth_image_file):
    try:

        # Read images from file data
        person_image = Image.open(person_image_file).convert("RGB")
        cloth_image = Image.open(cloth_image_file).convert("RGB")

        # Extract input parameters
        cloth_type ='upper'
        width = 768
        height = 1024
        num_inference_steps = 50
        guidance_scale = 3.0
        seed = 42

        # Resize images
        person_image = resize_and_crop(person_image, (width, height))
        cloth_image = resize_and_padding(cloth_image, (width, height))

        # Generate mask
        mask = automasker(person_image, cloth_type)["mask"]
        mask = mask_processor.blur(mask, blur_factor=9)

        # Generate random seed
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed != -1 else None

        # Run pipeline
        result_image =inference(
        person_image, 
        cloth_image, 
        cloth_type="upper",
        num_steps=50, 
        guidance_scale=30.0, 
        seed=42, 
        width=768,
        height=1024
        )

        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return result_image

    except Exception as e:
        print(e)
        return e#jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    infer(person_image_file="yo.png",cloth_image_file="sweater.png")
