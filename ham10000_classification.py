import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# HAM10000 Classes
CLASS_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma', 
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES.keys())}

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        # Filter to only include images that actually exist
        available_images = set([f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.metadata = self.metadata[self.metadata['image_id'].isin(available_images)]
        
        # Remove duplicates, keep first occurrence
        self.metadata = self.metadata.drop_duplicates(subset=['lesion_id'], keep='first')
        
        # Split data
        if is_train:
            _, self.metadata = train_test_split(
                self.metadata, test_size=0.2, stratify=self.metadata['dx'], random_state=42
            )
        else:
            self.metadata, _ = train_test_split(
                self.metadata, test_size=0.8, stratify=self.metadata['dx'], random_state=42
            )
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = class_to_idx[self.metadata.iloc[idx]['dx']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Reduced size for faster processing
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Reduced size for faster processing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model():
    # Load pretrained EfficientNet-B0 (smaller and faster than B3)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace final classifier for 7 classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 7)
    
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx):
        output = self.model(x)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, device='cuda'):
    model = model.to(device)
    best_acc = 0.0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 20 == 0:  # More frequent updates for CPU
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss_epoch = val_loss / len(val_loader)
        val_acc_epoch = 100 * correct / total
        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%')
        
        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model, train_losses, val_losses, train_accs, val_accs, all_preds, all_labels

def evaluate_model(y_true, y_pred):
    # Convert back to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    y_true_names = [idx_to_class[idx] for idx in y_true]
    y_pred_names = [idx_to_class[idx] for idx in y_pred]
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    print(f'\n=== EVALUATION METRICS ===')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    # Classification report
    print(f'\n=== CLASSIFICATION REPORT ===')
    print(classification_report(y_true_names, y_pred_names, target_names=list(CLASS_NAMES.values())))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(CLASS_NAMES.values()), 
                yticklabels=list(CLASS_NAMES.values()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, cm

def visualize_grad_cam(model, dataset, grad_cam, device='cuda', num_samples=3):
    model.eval()
    
    plt.figure(figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Get a sample
        image, label = dataset[i]
        img_tensor = image.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output).item()
            confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
        
        # Generate Grad-CAM
        cam = grad_cam(img_tensor, pred_idx)
        
        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = image * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Resize CAM to match image
        cam_resized = torch.nn.functional.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0), 
            size=(256, 256),  # Match reduced image size
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        
        # Create heatmap
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        
        # Overlay
        overlay = 0.6 * img_denorm.permute(1, 2, 0).numpy() + 0.4 * heatmap
        
        # Plot
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_denorm.permute(1, 2, 0))
        plt.title(f'Original\nTrue: {list(CLASS_NAMES.keys())[label]}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(cam_resized, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(overlay)
        plt.title(f'Overlay\nPred: {list(CLASS_NAMES.keys())[pred_idx]} ({confidence:.2f})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('grad_cam_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    CSV_FILE = "HAM10000_metadata.csv"
    IMG_DIR = "."
    BATCH_SIZE = 16  # Reduced for CPU
    NUM_EPOCHS = 2  # Reduced for faster training
    DEVICE = 'cpu'  # Force CPU usage
    
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {CSV_FILE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    # Create datasets and dataloaders
    train_transform, val_transform = get_transforms()
    
    train_dataset = HAM10000Dataset(CSV_FILE, IMG_DIR, train_transform, is_train=True)
    val_dataset = HAM10000Dataset(CSV_FILE, IMG_DIR, val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    trained_model, train_losses, val_losses, train_accs, val_accs, y_pred, y_true = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE
    )
    
    # Evaluate
    accuracy, precision, recall, cm = evaluate_model(y_true, y_pred)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Grad-CAM visualization
    print("\nGenerating Grad-CAM visualizations...")
    grad_cam = GradCAM(trained_model, trained_model.features[-1])  # Final convolution layer
    visualize_grad_cam(trained_model, val_dataset, grad_cam, DEVICE, num_samples=2)  # Reduced samples
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Precision: {precision:.4f}")
    print(f"Final Test Recall: {recall:.4f}")
    print("\nFiles saved:")
    print("- best_model.pth (best model weights)")
    print("- confusion_matrix.png")
    print("- grad_cam_results.png")
    print("- training_curves.png")

if __name__ == "__main__":
    main()
