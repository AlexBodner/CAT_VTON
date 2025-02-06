import argparse
import os
from datetime import datetime

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CatVTON pipeline.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",
        help="Path to the base model for evaluation.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help="Path to the checkpoint of the trained try-on model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width to resize the input images.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height to resize the input images.",
    )
    parser.add_argument(
        "--person_image_path",
        type=str,
        required=True,
        help="Path to the person image.",
    )
    parser.add_argument(
        "--cloth_image_path",
        type=str,
        required=True,
        help="Path to the cloth image.",
    )
    parser.add_argument(
        "--cloth_type",
        type=str,
        default="upper",
        choices=["upper", "lower", "overall"],
        help="Type of the cloth.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for the pipeline.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="Guidance scale for the pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility (-1 for random seed).",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Allow TensorFloat32 for Ampere GPUs.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision type.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Download checkpoint
    repo_path = snapshot_download(repo_id=args.resume_path)

    # Initialize pipeline and mask processor
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model_path,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(args.mixed_precision),
        use_tf32=False,
        device="cuda",
        skip_safety_check = True

    )
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device="cuda",
    )

    # Read images
    person_image = Image.open(args.person_image_path).convert("RGB")
    cloth_image = Image.open(args.cloth_image_path).convert("RGB")

    # Resize images
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    # Generate mask
    mask = automasker(person_image, args.cloth_type)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    # Generate random seed
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed != -1 else None

    # Run pipeline
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )[0]

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = os.path.join(args.output_dir, f"result_{timestamp}.png")
    result_image.save(result_path)

    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
