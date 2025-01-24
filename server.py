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
app = Flask(__name__)

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

@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Check if the post request has the file part
        if 'person_image' not in request.files or 'cloth_image' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        person_image_file = request.files['person_image']
        cloth_image_file = request.files['cloth_image']

        # If the user does not select a file, the browser submits an empty part without filename
        if person_image_file.filename == '' or cloth_image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read images from file data
        person_image = Image.open(person_image_file).convert("RGB")
        cloth_image = Image.open(cloth_image_file).convert("RGB")

        # Extract input parameters
        cloth_type = request.form.get('cloth_type', 'upper')
        width = int(request.form.get('width', 768))
        height = int(request.form.get('height', 1024))
        num_inference_steps = int(request.form.get('num_inference_steps', 50))
        guidance_scale = float(request.form.get('guidance_scale', 2.5))
        seed = int(request.form.get('seed', -1))

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

        # Return the image as response
        return send_file(
            BytesIO(img_byte_arr),
            mimetype='image/png',
            as_attachment=False,
            download_name='result.png'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Default to 8080
    app.run(host="0.0.0.0", port=port)