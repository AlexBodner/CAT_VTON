import torch
import os
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor

from huggingface_hub import snapshot_download
from model.cloth_masker import AutoMasker
from tryon_inference import run_inference
import tempfile

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"


def inference(
    image_data, 
    garment, 
    cloth_type="upper",
    num_steps=50, 
    guidance_scale=30.0, 
    seed=-1, 
    width=768,
    height=1024
):
    """Wrapper function for Gradio interface"""
    # Check if mask has been drawn
    #if image_data is None or "layers" not in image_data or not image_data["layers"]:
    #    print("Please draw a mask over the clothing area before generating!")
    
    # Check if mask is empty (all black)
    resume_path = "zhengchong/CatVTON"

    # Download checkpoint
    repo_path = snapshot_download(repo_id=resume_path)
    #mask = #image_data["layers"][0]
    mask_array = np.array(mask)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device="cuda",
    )
    mask = automasker(image_data, cloth_type)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)
    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save inputs to temp directory
        temp_image = os.path.join(tmp_dir, "image.png")
        temp_mask = os.path.join(tmp_dir, "mask.png")
        temp_garment = os.path.join(tmp_dir, "garment.png")
        
        # Extract image and mask from ImageEditor data
        #image = image_data["background"]
        #mask = image_data["layers"][0]  # First layer contains the mask
        
        # Convert to numpy array and process mask
        mask_array = np.array(mask)
        is_black = np.all(mask_array < 10, axis=2)
        mask = Image.fromarray(((~is_black) * 255).astype(np.uint8))
        
        # Save files to temp directory
        image_data.save(temp_image)
        mask.save(temp_mask)
        garment.save(temp_garment)
        
        try:
            # Run inference
            _, tryon_result = run_inference(
                pipe=None,
                image_path=temp_image,
                mask_path=temp_mask,
                garment_path=temp_garment,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                size=(width, height)
            )
            return tryon_result
        except Exception as e:
            print(f"Error during inference: {str(e)}")
