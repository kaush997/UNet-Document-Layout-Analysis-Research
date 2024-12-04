import os
from preprocessing.noise_removal import remove_noise
from preprocessing.binarization import binarize_image
from preprocessing.skew_correction import correct_skew
from preprocessing.enhancement import enhance_contrast
from preprocessing.normalization import normalize_image
from preprocessing.augmentation import augment_image
from preprocessing.utils import save_image

# Update input and output directories to match your structure
INPUT_DIR = "data/project-7-at-2024-11-28-20-06-811bd479"
OUTPUT_DIR = "data/project-7-at-2024-11-28-20-06-811bd479-processed"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Process each label directory
for label in os.listdir(INPUT_DIR):
    label_path = os.path.join(INPUT_DIR, label)
    if os.path.isdir(label_path):  # Ensure it's a directory
        print(f"Processing label: {label}")

        # Define input and output subdirectories
        input_images_dir = os.path.join(label_path, "Images")
        output_images_dir = os.path.join(OUTPUT_DIR, label, "Images")

        # Create output directories for this label
        os.makedirs(output_images_dir, exist_ok=True)

        # Iterate over images in the label's "Images" folder
        for image_name in os.listdir(input_images_dir):
            input_image_path = os.path.join(input_images_dir, image_name)
            output_image_path = os.path.join(output_images_dir, image_name)

            # Preprocess the image step-by-step
            denoised = remove_noise(input_image_path)
            binarized = binarize_image(denoised)
            deskewed = correct_skew(binarized)
            enhanced = enhance_contrast(deskewed)
            normalized = normalize_image(enhanced)
            augmented = augment_image(normalized)

            # Save the processed image
            save_image(augmented, output_image_path)

        print(f"Completed processing for label: {label}")
