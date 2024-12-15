import os

# Define the root folder containing category folders
root_folder = r"D:\Python_projects\UNet-Document-Layout-Analysis-Research\data\project-7-at-2024-11-28-20-06-811bd479"

# List of categories in the root folder
categories = os.listdir(root_folder)

# Initialize total count
total_images = 0

# Iterate through each category and count images
for category in categories:
    category_path = os.path.join(root_folder, category)
    augmented_image_folder = os.path.join(category_path, "Augmented_Images")

    # Check if the Images folder exists
    if os.path.exists(augmented_image_folder):
        # Count the number of image files
        image_count = len([file for file in os.listdir(augmented_image_folder) if os.path.isfile(os.path.join(augmented_image_folder, file))])
        total_images += image_count  # Add to total count
        print(f"Category '{category}' contains {image_count} image(s) in the 'Images' folder.")
    else:
        print(f"Category '{category}' does not have an 'Images' folder.")

# Print the total count of images
print(f"\nTotal number of images across all categories: {total_images}")
