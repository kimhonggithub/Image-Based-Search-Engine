import os
import shutil
import random

def generate_selected_image_list(original_dataset_root, max_images_per_category=500):
    selected_image_list = []
    for class_name in os.listdir(original_dataset_root):
        class_path = os.path.join(original_dataset_root, class_name)
        images_in_category = sum(len(files) for _, _, files in os.walk(class_path))
        if images_in_category <= max_images_per_category:
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        selected_image_list.append((class_name, os.path.join(root, file)))
        else:
            images_per_subfolder = max_images_per_category // len(os.listdir(class_path))
            for root, _, files in os.walk(class_path):
                random.shuffle(files)
                selected_files = files[:images_per_subfolder]
                for file in selected_files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        selected_image_list.append((class_name, os.path.join(root, file)))
    return selected_image_list

def generate_dataset_with_selected_images(original_dataset_root, new_dataset_root, selected_image_list):
    for class_name, image_path in selected_image_list:
        dest_folder = os.path.join(new_dataset_root, class_name)
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, os.path.basename(image_path))
        shutil.copy(image_path, dest_path)

original_dataset_root = 'Fashion_Items_Dataset/'  
new_dataset_root = 'New_Fashion_Items_Dataset/'  

selected_image_list = generate_selected_image_list(original_dataset_root)

generate_dataset_with_selected_images(original_dataset_root, new_dataset_root, selected_image_list)
