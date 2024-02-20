# streamlit_app.py
import streamlit as st
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors
from model import CustomImageDataset  # assuming you have the CustomImageDataset 
from vector import get_vector  # and get_vector functions in a model.py file
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

# Load the pretrained model
resnet50_model_path = "resnet50_model.pth"  # Replace with the actual path to your saved model
resnet50_model = torch.load(resnet50_model_path)

# Load the model's state dictionary
layer_of_resnet50 = resnet50_model._modules.get('avgpool')

resnet50_model_state_dict_path = "resnet50_model_state_dict.pth"
resnet50_model.load_state_dict(torch.load(resnet50_model_state_dict_path))
resnet50_model.eval()


# Define Simple Image Transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the dataset
image_directory = 'New_Fashion_Items_Dataset/'  # Will Be Replace with the actual directory path of FULL DATASET
dataset = CustomImageDataset(root_dir=image_directory, transform=image_transform, include_feature_vector=True)

# Streamlit UI
st.title("Fashion Items Similarity Search")

# Upload a query image
query_image = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])


# Display query image and similar images when a query image is uploaded
if query_image is not None:
    # Process the query image
    query_image = Image.open(query_image)
    st.image(query_image, caption="Query Image", width=200)

    # Define the DataLoader
    batch_size = 500
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        
    # Get a batch from the DataLoader (assuming batch size is 1)
    for labels, image_names, images, feature_vectors in data_loader:
   
        break  # Break after showing the first batch

    # Compute similar images
    output_size = 2048
    

    query_vector = get_vector(query_image, resnet50_model, layer_of_resnet50, output_size=output_size)

    knn = NearestNeighbors(n_neighbors=5, metric="cosine")

    knn.fit(feature_vectors)

    query_vector = query_vector.reshape(1, -1)

    distances, indices = knn.kneighbors(query_vector)

    distances_list = distances.tolist()

    indices_list = indices.tolist()


    # Display similar images in a horizontal line
    st.subheader("Similar Images")
    for i in range(5):
        neighbor_index = indices_list[0][i]
        similarity_percentage = (1 - distances_list[0][i]) * 100

        # Convert torch tensor to numpy array and then to PIL Image
        neighbor_image_np = images[neighbor_index].permute(1, 2, 0).numpy()
        neighbor_image_np = np.clip(neighbor_image_np, 0, 1) * 255  # Ensure pixel values are in the valid range
        neighbor_image_pil = Image.fromarray(neighbor_image_np.astype(np.uint8))

        # Display image and title in the same line
        st.image(neighbor_image_pil, caption=f"Similarity: {similarity_percentage:.2f}%", width=150)
        st.text(f"Neighbor {i + 1}: {image_names[neighbor_index]}")
