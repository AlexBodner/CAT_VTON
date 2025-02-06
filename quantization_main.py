from ov_catvton_helper import download_models, convert_pipeline_models, convert_automasker_models

pipeline, mask_processor, automasker = download_models()
vae_scaling_factor = pipeline.vae.config.scaling_factor
convert_pipeline_models(pipeline)
convert_automasker_models(automasker)


import openvino as ov



core = ov.Core()

device = device_widget()

print(device)
from ov_catvton_helper import (
    get_compiled_pipeline,
    get_compiled_automasker,
    VAE_ENCODER_PATH,
    VAE_DECODER_PATH,
    UNET_PATH,
    DENSEPOSE_PROCESSOR_PATH,
    SCHP_PROCESSOR_ATR,
    SCHP_PROCESSOR_LIP,
)

pipeline = get_compiled_pipeline(pipeline, core, device, VAE_ENCODER_PATH, VAE_DECODER_PATH, UNET_PATH, vae_scaling_factor)
automasker = get_compiled_automasker(automasker, core, device, DENSEPOSE_PROCESSOR_PATH, SCHP_PROCESSOR_ATR, SCHP_PROCESSOR_LIP)
from pathlib import Path
from catvton_quantization_helper import collect_calibration_data, UNET_INT8_PATH

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

if not UNET_INT8_PATH.exists():
    subset_size = 100
    calibration_data = collect_calibration_data(pipeline, automasker, mask_processor, dataset, subset_size)

import gc
import nncf
from ov_catvton_helper import UNET_PATH

# cleanup before quantization to free memory
del pipeline
del automasker
gc.collect()


if not UNET_INT8_PATH.exists():
    unet = core.read_model(UNET_PATH)
    quantized_model = nncf.quantize(
        model=unet,
        calibration_dataset=nncf.Dataset(calibration_data),
        subset_size=subset_size,
        model_type=nncf.ModelType.TRANSFORMER,
    )
    ov.save_model(quantized_model, UNET_INT8_PATH)
    del quantized_model
    gc.collect()

from catvton_quantization_helper import compress_models

compress_models(core)

is_optimized_pipe_available = True

from catvton_quantization_helper import compare_models_size

compare_models_size()


from ov_catvton_helper import get_pipeline_selection_option

use_quantized_models = get_pipeline_selection_option(is_optimized_pipe_available)




from catvton_quantization_helper import (
    VAE_ENCODER_INT4_PATH,
    VAE_DECODER_INT4_PATH,
    DENSEPOSE_PROCESSOR_INT4_PATH,
    SCHP_PROCESSOR_ATR_INT4,
    SCHP_PROCESSOR_LIP_INT4,
    UNET_INT8_PATH,
)

pipeline, mask_processor, automasker = download_models()
pipeline = get_compiled_pipeline(pipeline, core, device, VAE_ENCODER_INT4_PATH, VAE_DECODER_INT4_PATH, UNET_INT8_PATH, vae_scaling_factor)
automasker = get_compiled_automasker(automasker, core, device, DENSEPOSE_PROCESSOR_INT4_PATH, SCHP_PROCESSOR_ATR_INT4, SCHP_PROCESSOR_LIP_INT4)


output_dir = "output"
from PIL import Image
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# Read images
person_image = Image.open("yo.png").convert("RGB")
cloth_image = Image.open("buzo.png").convert("RGB")

# Resize images
person_image = resize_and_crop(person_image, (768, 1024))
cloth_image = resize_and_padding(cloth_image, (768, 1024))
import time
start = time.perf_counter()
# Generate mask
mask = automasker(person_image, "upper")["mask"]
mask = mask_processor.blur(mask, blur_factor=9)

print("mask creation", time.perf_counter()-start)
import torch
# Generate random seed
seed = 42
generator = torch.Generator(device="cuda").manual_seed(seed) if seed != -1 else None

# Run pipeline
result_image = pipeline(
    image=person_image,
    condition_image=cloth_image,
    mask=mask,
    num_inference_steps=50,
    guidance_scale=30.0,
    generator=generator,
)[0]

print("whole pipe",time.perf_counter()-start)
import os
# Save results
os.makedirs(output_dir, exist_ok=True)
result_path = os.path.join(output_dir, f"result.png")
result_image.save(result_path)

print(f"Result saved to: {result_path}")
