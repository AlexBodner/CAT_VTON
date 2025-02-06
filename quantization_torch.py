from ov_catvton_helper import download_models
import torch
import torch.quantization
from pathlib import Path
from PIL import Image
import os
import time
from utils import resize_and_crop, resize_and_padding

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download models
pipeline, mask_processor, automasker = download_models()
vae_scaling_factor = pipeline.vae.config.scaling_factor

# Move models to GPU
pipeline.vae = pipeline.vae.to(device)
pipeline.unet = pipeline.unet.to(device)
automasker.densepose_processor.predictor.model = automasker.densepose_processor.predictor.model.to(device)
automasker.schp_processor_atr.model = automasker.schp_processor_atr.model.to(device)
automasker.schp_processor_lip.model = automasker.schp_processor_lip.model.to(device)

# Quantize models
print("Quantizing models...")

def quantize_model(model):
    model.eval()  # Ensure model is in evaluation mode
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model

pipeline.unet = quantize_model(pipeline.unet)
pipeline.vae = quantize_model(pipeline.vae)
automasker.densepose_processor.predictor.model = quantize_model(
    automasker.densepose_processor.predictor.model
)
automasker.schp_processor_atr.model = quantize_model(
    automasker.schp_processor_atr.model
)
automasker.schp_processor_lip.model = quantize_model(
    automasker.schp_processor_lip.model
)

print("Quantization complete!")

# Example dataset
dataset = [
    (
        Path("resource/demo/example/person/men/model_5.png"),
        Path("resource/demo/example/condition/upper/24083449_54173465_2048.jpg"),
    ),
    (
        Path("resource/demo/example/person/women/2-model_4.png"),
        Path("resource/demo/example/condition/overall/21744571_51588794_1000.jpg"),
    ),
]

# Process images
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Read and preprocess images
person_image = Image.open("yo.png").convert("RGB")
cloth_image = Image.open("buzo.png").convert("RGB")

person_image = resize_and_crop(person_image, (768, 1024))
cloth_image = resize_and_padding(cloth_image, (768, 1024))

# Generate mask
start = time.perf_counter()
mask = automasker(person_image, "upper")["mask"]
mask = mask_processor.blur(mask, blur_factor=9)
print("Mask creation time:", time.perf_counter() - start)

# Generate random seed for reproducibility
seed = 42
generator = torch.Generator(device=device).manual_seed(seed) if seed != -1 else None

# Run pipeline
start = time.perf_counter()
result_image = pipeline(
    image=person_image,
    condition_image=cloth_image,
    mask=mask,
    num_inference_steps=50,
    guidance_scale=30.0,
    generator=generator,
)[0]
print("Inference time:", time.perf_counter() - start)

# Save results
result_path = os.path.join(output_dir, "result.png")
result_image.save(result_path)
print(f"Result saved to: {result_path}")
