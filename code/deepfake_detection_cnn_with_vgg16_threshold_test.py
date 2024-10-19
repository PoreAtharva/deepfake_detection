import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Define the custom dataset class
class SpectraDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the SpectraVGG16 model
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
    
    def forward(self, x):
        return self.model(x)

# Function to load test data
def load_test_data(deepfake_dir, original_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Gather image paths and labels
    deepfake_images = [os.path.join(deepfake_dir, img) for img in os.listdir(deepfake_dir) if os.path.isfile(os.path.join(deepfake_dir, img))]
    original_images = [os.path.join(original_dir, img) for img in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, img))]
    
    image_paths = deepfake_images + original_images
    labels = [0] * len(deepfake_images) + [1] * len(original_images)  # 0: Deepfake, 1: Original
    
    # Create dataset
    test_dataset = SpectraDataset(image_paths, labels, transform=transform)
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

# Function to evaluate the model
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    return accuracy

# Main execution
if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set speaker name
    speaker_name = "Barack_Obama"  # Change this to the desired speaker name

    # Define the list of threshold values
    threshold_values = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]

    # Load the trained model once and reuse it for evaluation on each threshold
    model = SpectraVGG16(num_classes=2).to(device)
    
    # Iterate over each threshold value
    for threshold_value in threshold_values:
        model_path = f'path/to/models/spectra_vgg16_model_{speaker_name.lower()}_{threshold_value}.pth'
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Skipping this threshold.")
            continue
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'\nModel loaded for threshold value: {threshold_value}')
        
        # Define test datasets dynamically based on speaker name and threshold value
        test_sets = {
            "Original Test Set": {
                "deepfake_dir": f'path/to/data/{speaker_name}/test/deepfake_img',
                "original_dir": f'path/to/data/{speaker_name}/test/original_img'
            },
            "Thresholded Test Set": {
                "deepfake_dir": f'path/to/data/{speaker_name.lower()}_threshold_images/thresh_{threshold_value}/test/deepfake',
                "original_dir": f'path/to/
                data/{speaker_name.lower()}_threshold_images/thresh_{threshold_value}/test/original'
            }
        }
        
        # Iterate over each test set, evaluate, and print accuracy
        for test_set_name, paths in test_sets.items():
            deepfake_dir = paths["deepfake_dir"]
            original_dir = paths["original_dir"]
            
            # Check if directories exist
            if not os.path.isdir(deepfake_dir):
                print(f"Deepfake directory not found: {deepfake_dir}")
                continue
            if not os.path.isdir(original_dir):
                print(f"Original directory not found: {original_dir}")
                continue
            
            print(f'\nEvaluating on {test_set_name} for threshold {threshold_value}...')
            test_loader = load_test_data(deepfake_dir, original_dir, batch_size=32)
            accuracy = evaluate_model(model, test_loader, device=device)
            print(f'{test_set_name} Accuracy for threshold {threshold_value}: {accuracy:.2f}%')
            print('-' * 50)
