import torchvision.transforms as transforms
import torch
from torch.autograd import Variable


# Additional transformations for data augmentation
random_rotation = transforms.RandomRotation(degrees=30)
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
horizontal_flip = transforms.RandomHorizontalFlip()

# Image transforms
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image, model, layer, output_size, augment=True):
    # 2. Create a PyTorch Variable with the transformed image (data augment)
    if augment:
        image = random_rotation(image)
        image = horizontal_flip(image)
        image = color_jitter(image)

    # 3. Data Preprocessing
    transformed_image = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))
    
    # 4. Create a vector of zeros that will hold our feature vector
    # Number of the output size of the 'avgpool' layer 
    my_embedding = torch.zeros(output_size)
    
    # 5. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
        
    # 6. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 7. Run the model on our transformed image
    model(transformed_image)
    
    # 8. Detach our copy function from the layer
    h.remove()
    
    # 9. Return the feature vector
    return my_embedding.numpy()

    

