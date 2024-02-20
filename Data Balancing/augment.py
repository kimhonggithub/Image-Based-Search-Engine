import os
import random
from torchvision import transforms
from PIL import Image
import numpy as np

# default_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomRotation(degrees=30),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def get_all_image_files(folder):
#     image_files = []
#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_files.append(os.path.join(root, file))
#     return image_files

# print(len(get_all_image_files('New_Fashion_Items_Dataset/Caps')))


# def calculate_images_to_augment(subfolder_input_path):
#     image_files = get_all_image_files(subfolder_input_path)
#     target_images = 500
#     num_images_to_augment = max(target_images - len(image_files), 0)
#     return num_images_to_augment

# print(calculate_images_to_augment('New_Fashion_Items_Dataset/Caps'))

# def augment_and_save_images(
#     input_folder: str,
#     output_folder: str,
#     num_images_to_augment: int = 10,
# ):
#     subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

#     augmented_filenames = set()

#     for subfolder in subfolders:
#         subfolder_input_path = os.path.join(input_folder, subfolder)
#         subfolder_output_path = os.path.join(output_folder, subfolder)

#         os.makedirs(subfolder_output_path, exist_ok=True)

#         num_images_to_augment = calculate_images_to_augment(subfolder_input_path)

#         for _ in range(num_images_to_augment):
#             image_files = get_all_image_files(subfolder_input_path)
#             available_files = set(image_files) - augmented_filenames
#             if not available_files:
#                 print(f"No more image files to augment in subfolder {subfolder_input_path}.")
#                 break
#             random_image_file = random.choice(list(available_files))
#             img = Image.open(random_image_file)
#             augmented_img = default_transform(img)
#             augmented_img_array = (np.array(augmented_img) * 255).astype(np.uint8)
#             # # Save the original image
#             # original_img_path = os.path.join(subfolder_output_path, f"original_{os.path.basename(random_image_file)}")
#             # img.save(original_img_path)
#             augmented_img_path = os.path.join(subfolder_output_path, f"augmented_{os.path.basename(random_image_file)}")
#             Image.fromarray(augmented_img_array.transpose((1, 2, 0))).save(augmented_img_path)
#             augmented_filenames.add(random_image_file)

#     print("Augmentation and saving complete.")

import os
import random
from torchvision import transforms
from PIL import Image
import numpy as np

# Your provided default transform
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_all_image_files(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return image_files

def calculate_images_to_augment(subfolder_input_path):
    image_files = get_all_image_files(subfolder_input_path)
    target_images = 500
    num_images_to_augment = max(target_images - len(image_files), 0)
    return num_images_to_augment

def augment_and_save_images(
    input_folder: str,
    output_folder: str,
    num_images_to_augment: int = 10,
):
    image_files = get_all_image_files(input_folder)
    augmented_filenames = set()

    os.makedirs(output_folder, exist_ok=True)

    num_images_to_augment = calculate_images_to_augment(input_folder)

    for _ in range(num_images_to_augment):
        available_files = set(image_files) - augmented_filenames
        if not available_files:
            print(f"No more image files to augment in folder {input_folder}.")
            break
        random_image_file = random.choice(list(available_files))
        img = Image.open(os.path.join(input_folder, random_image_file))
        augmented_img = default_transform(img)
        augmented_img_array = (np.array(augmented_img) * 255).astype(np.uint8)
        augmented_img_path = os.path.join(output_folder, f"augmented_{random_image_file}")
        Image.fromarray(augmented_img_array.transpose((1, 2, 0))).save(augmented_img_path)
        augmented_filenames.add(random_image_file)

    print("Augmentation and saving complete.")


input_folder_path = "New_Fashion_Items_Dataset/Wallets"  
output_folder_path = "New_Fashion_Items_Dataset/Wallets"
augment_and_save_images(input_folder_path, output_folder_path)
