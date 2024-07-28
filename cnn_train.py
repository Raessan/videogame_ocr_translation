import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, random_split
import cv2
from params import *
import os

# Define the letterbox function
def letterbox(img, size_characters=(64, 64)):
    """Return updated labels and image with added border."""
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(size_characters[0] / shape[0], size_characters[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = size_characters[1] - new_unpad[0], size_characters[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[0] != new_unpad[0] or shape[1] != new_unpad[1]:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    return img

# Define a custom transform to include letterbox
class LetterboxTransform:
    def __init__(self, size_characters):
        self.size_characters = size_characters

    def __call__(self, img):
        # Convert PIL image to numpy array
        img = np.array(img)
        img = letterbox(img, self.size_characters)
        # Convert back to PIL image
        img = transforms.functional.to_pil_image(img)
        return img
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_size=(64, 64)):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Define a function to train the model
def train_model(model, train_loader, test_loader, device, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # img = images[0, 0, :, :].squeeze().numpy()*255
            # print(np.max(img))
            # print(np.min(img))
            #cv2.imshow("image", img)
            #cv2.waitKey(0)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = correct / total * 100

        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = correct / total * 100

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
    # Save the model after training
    if not os.path.exists(save_cnn_folder):
        os.makedirs(save_cnn_folder)
    torch.save(model.state_dict(), os.path.join(save_cnn_folder, save_cnn_file))
    print("Model saved to ", os.path.join(save_cnn_folder, save_cnn_file))

# Define a function to evaluate the model and collect misclassified images
def evaluate_model(model, dataset, label_list):
    model.eval()
    misclassified = {}

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if predicted != labels:
                img_path = dataset.samples[i][0]
                misclassified[img_path] = label_list[predicted.item()]

    return misclassified

if __name__ == '__main__':

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dataset = CharDataset(folder_chars, size_characters)
    # Define transformations (only to tensor and normalize without augmentation)
    # Define transformations (only to tensor and normalize without augmentation)
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure images are in grayscale (since binary)
        LetterboxTransform(size_characters),  # Apply letterbox transformation
        transforms.ToTensor(),   # Convert images to PyTorch tensors
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(root=folder_chars, transform=transform)

    # Split the dataset into train and test sets (80% train, 20% test)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders for the train and test sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Print some information about the datasets
    print(f'Total images: {len(full_dataset)}')
    print(f'Training images: {len(train_dataset)}')
    print(f'Testing images: {len(test_dataset)}')

    # Example: Iterate over the train_loader and print the shape of the images and labels
    for images, labels in train_loader:
        print(f'Image batch dimensions: {images.shape}')
        print(f'Image label dimensions: {labels.shape}')
        print(f"Label: {labels}")
        break

    # Get the class-to-index mapping
    class_to_idx = full_dataset.class_to_idx
    print("Class to index mapping:")
    for class_name, idx in class_to_idx.items():
        print(f"{class_name}: {idx}")

    # Count the number of classes (subfolders)
    classes = os.listdir(folder_chars)
    num_classes = len(classes)

    # Initialize the model and move it to the appropriate device
    model = SimpleCNN(num_classes, size_characters).to(device)

    # Print the model architecture
    print(model)

    # Train the model
    train_model(model, train_loader, test_loader, device, num_epochs=10, learning_rate=0.001)

    # Evaluate the model and get misclassified images
    misclassified_images = evaluate_model(model, full_dataset, list(class_to_idx.keys()))

    # Print the misclassified images and their predicted labels
    for img_path, predicted_label in misclassified_images.items():
        print(f'Image: {img_path}, Predicted Label: {predicted_label}')
