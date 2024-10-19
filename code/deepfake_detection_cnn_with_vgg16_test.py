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
def load_test_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Paths for test dataset
    deepfake_dir = os.path.join(data_dir, 'Barack_Obama', 'test', 'deepfake_img')
    original_dir = os.path.join(data_dir, 'Barack_Obama', 'test', 'original_img')

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
    
    # Load the trained model
    model = SpectraVGG16(num_classes=2).to(device)
    model.load_state_dict(torch.load('path/to/models/spectra_vgg16_model_updated.pth'))
    
    # Load test data
    data_dir = 'path/to/data'
    test_loader = load_test_data(data_dir, batch_size=32)
    
    # Evaluate the model
    accuracy = evaluate_model(model, test_loader, device=device)
    print(f'Test Accuracy: {accuracy:.2f}%')
