import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Define the SpectraVGG16 model with activation hooks
class SpectraVGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(SpectraVGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        
        # Freeze the convolutional base
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        num_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # Dictionary to store activations
        self.activations = {}

    def forward(self, x):
        return self.model(x)
    
    # Hook to capture the activations
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    # Method to register hooks for specific layers
    def register_hooks(self, layer_names):
        hooks = []
        for name, layer in self.model.features.named_children():
            if name in layer_names:
                hook = layer.register_forward_hook(self.get_activation(name))
                hooks.append(hook)
        return hooks

# Function to define the transformation for input images
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to visualize activations
def visualize_activations(model, image_path, transform, layer_names, device, save_dir):
    # Prepare the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Create directory to save activations
    image_name = os.path.basename(image_path).split('.')[0]
    os.makedirs(save_dir, exist_ok=True)

    # Register hooks
    hooks = model.register_hooks(layer_names)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Plot activations
    activations = model.activations
    for layer in layer_names:
        act = activations.get(layer)
        if act is None:
            print(f"No activation found for layer: {layer}")
            continue
        
        act = act.squeeze(0)  # Remove batch dimension
        num_channels = act.size(0)
        
        # Determine grid size
        grid_size = int(np.sqrt(num_channels))
        if grid_size ** 2 < num_channels:
            grid_size += 1
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'Activations from layer: {layer} for {image_name}', fontsize=16)
        
        for i in range(grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            ax = axes[row, col]
            if i < num_channels:
                feature_map = act[i].cpu().numpy()
                feature_map -= feature_map.mean()
                feature_map /= (feature_map.std() + 1e-5)
                feature_map = np.clip(feature_map, 0, 1)
                ax.imshow(feature_map, cmap='viridis')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer}_activations.png'))
        plt.close()

    # Remove hooks to prevent memory leaks
    for hook in hooks:
        hook.remove()

# Main function
def main():
    # Configuration
    base_dir = 'path/to/barack_obama_threshold_images'
    save_base_dir_deepfake = 'path/to/saved_activations/VGG16/deepfake'
    save_base_dir_original = 'path/to/saved_activations/VGG16/original'
    
    # Layers to visualize
    layers_to_visualize = ['0', '5', '10', '14', '19', '21']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold_values = [0, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    for threshold in threshold_values:
        threshold_dir = os.path.join(base_dir, f'thresh_{threshold}/test')
        print(f"threshold_dir: {threshold_dir}")
        if not os.path.exists(threshold_dir):
            print(f"No directory found for {threshold_dir}")
            continue
        
        # Get images from deepfake and original folders
        deepfake_folder = os.path.join(threshold_dir, 'deepfake')
        original_folder = os.path.join(threshold_dir, 'original')
        
        deepfake_images = [os.path.join(deepfake_folder, f) for f in os.listdir(deepfake_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        original_images = [os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

        # Randomly select one image from each folder
        if deepfake_images and original_images:
            selected_deepfake_image = random.choice(deepfake_images)
            selected_original_image = random.choice(original_images)

            # Load the model
            model_path = f'path/to/models/spectra_vgg16_model_barack_obama_{threshold}.pth'
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue

            model = SpectraVGG16(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            # Get the transformation
            transform = get_transform()

            # Create save directory for activations
            save_dir_deepfake = os.path.join(save_base_dir_deepfake, str(threshold))
            save_dir_original = os.path.join(save_base_dir_original, str(threshold))
            os.makedirs(save_dir_deepfake, exist_ok=True)
            os.makedirs(save_dir_original, exist_ok=True)

            # Visualize activations for both selected images
            visualize_activations(model, selected_deepfake_image, transform, layers_to_visualize, device, save_dir_deepfake)
            visualize_activations(model, selected_original_image, transform, layers_to_visualize, device, save_dir_original)
        else:
            print(f"No images found in {threshold_dir}")

if __name__ == "__main__":
    main()
