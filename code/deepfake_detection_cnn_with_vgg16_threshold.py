import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import copy

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

# Define the SpectraVGG16 model without activation hooks
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

# Function to load data with train, val, test split
def load_data(speaker_name, thresh_value, batch_size=32, val_split=0.2, test_split=0.1):
    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Paths for speaker threshold images
    deepfake_dir = f'path/to/data/{speaker_name}_threshold_images/thresh_{thresh_value}/train/deepfake'
    original_dir = f'path/to/data/{speaker_name}_threshold_images/thresh_{thresh_value}/train/original'
    
    # Gather image paths and labels
    deepfake_images = [os.path.join(deepfake_dir, img) for img in os.listdir(deepfake_dir) if os.path.isfile(os.path.join(deepfake_dir, img))]
    original_images = [os.path.join(original_dir, img) for img in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, img))]
    
    image_paths = deepfake_images + original_images
    labels = [0] * len(deepfake_images) + [1] * len(original_images)  # 0: Deepfake, 1: Original
    
    # Split into train, validation, and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_split, random_state=42, stratify=labels)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split/(1 - test_split), random_state=42, stratify=y_train_val)
    
    # Create datasets with different transforms
    train_dataset = SpectraDataset(X_train, y_train, transform=transform_train)
    val_dataset = SpectraDataset(X_val, y_val, transform=transform_val_test)
    test_dataset = SpectraDataset(X_test, y_test, transform=transform_val_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Training function with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    min_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            epoch_loss = running_loss / total
            epoch_acc = 100 * correct / total
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
            
            # Deep copy the model if validation loss improves
            if phase == 'val':
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print('Early stopping!')
                        model.load_state_dict(best_model_wts)
                        return model
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Function to save the trained model
def save_model(model, speaker_name, thresh_value):
    model_path = f'path/to/models/spectra_vgg16_model_{speaker_name}_{thresh_value}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    test_loss = running_loss / total
    test_acc = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%')

# Main execution
if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize the model
    model = SpectraVGG16(num_classes=2).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only parameters of the classifier are being optimized
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Specify the speaker name
    speaker_name = 'barack_obama'  # Modify this based on your requirement
    
    # Iterate over a range of thresholds from 60 to 240 with a step of 10
    #threshold_values = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    threshold_values = [230, 240]
    for thresh_value in threshold_values:
        print(f"\nProcessing threshold value: {thresh_value}")
        
        # Load your data for the current threshold
        train_loader, val_loader, test_loader = load_data(speaker_name, thresh_value, batch_size=32)
        
        # Train the model
        model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5, device=device)
        
        # Save the trained model
        save_model(model, speaker_name, thresh_value)
        
        # Evaluate the model
        evaluate_model(model, test_loader, criterion, device=device)
