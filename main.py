import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Define data paths
base_dir = '/kaggle/input/leapgestrecog/leapGestRecog'

# Parameters
img_height, img_width = 64, 64
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(base_dir, transform=train_transform)
num_classes = len(dataset.classes)
print(f"Found {len(dataset)} images belonging to {num_classes} classes.")
print(f"Classes: {dataset.classes}")

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Apply validation transform to validation dataset
val_dataset.dataset = datasets.ImageFolder(base_dir, transform=val_transform)
val_indices = val_dataset.indices
val_dataset = torch.utils.data.Subset(val_dataset.dataset, val_indices)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

# Define CNN model
class GestureRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureRecognitionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        # Calculate the size for the first linear layer
        # After 3 conv layers with pooling: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First conv block
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)

        # Second conv block
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        # Third conv block
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GestureRecognitionCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct_train/total_train:.2f}%'})

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct_val/total_val:.2f}%'})

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    return train_losses, train_accuracies, val_losses, val_accuracies

# Train the model
print("Starting training...")
train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
)

# Load best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))
torch.save(model.state_dict(), 'final_model.pth')

# Evaluation function
def evaluate_model(model, val_loader, class_names):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print evaluation metrics
    print('\nConfusion Matrix:')
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    print('\nClassification Report:')
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print(report)

    return all_predictions, all_labels

# Evaluate the model
print("\nEvaluating model...")
predictions, true_labels = evaluate_model(model, val_loader, dataset.classes)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Function to visualize model predictions
def visualize_predictions(model, val_loader, class_names, num_images=8):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, num_images * 1.5))

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_shown == num_images:
                    return
                images_shown += 1

                plt.subplot(num_images // 4, 4, images_shown)
                plt.imshow(images[i].cpu().permute(1, 2, 0))
                plt.title(f'Pred: {class_names[predicted[i]]}\nActual: {class_names[labels[i]]}')
                plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize predictions
print("Visualizing predictions...")
visualize_predictions(model, val_loader, dataset.classes)
print("Prediction visualization saved as 'predictions_visualization.png'")

print("\nTraining completed! Best model saved as 'best_model.pth' and final model as 'final_model.pth'")
print(f"Training history plot saved as 'training_history.png'")
