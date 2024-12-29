import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
from module import ViViT

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 60
INPUT_SHAPE = (1, 28, 28, 28)  # (C, D, H, W)
PATCH_SIZE = (8, 8, 8)
NUM_CLASSES = 11
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

# Dataset preparation using MedMNIST
def prepare_medmnist_data():
    info = INFO["organmnist3d"]
    DataClass = getattr(medmnist, info["python_class"])
    train_dataset = DataClass(split="train", download=True)
    val_dataset = DataClass(split="val", download=True)
    test_dataset = DataClass(split="test", download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

# Training Loop
def train_model():
    train_loader, val_loader, test_loader = prepare_medmnist_data()

    model = ViViT(input_shape=INPUT_SHAPE, patch_size=PATCH_SIZE, embed_dim=PROJECTION_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch, (data, labels) in enumerate(train_loader):
            data = data.float()  # Ensure data is in float format
            labels = labels[:, 0].long()  # Select only the first column if labels are multi-target
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.float()  # Ensure data is in float format
                labels = labels[:, 0].long()  # Select only the first column if labels are multi-target
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Testing
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float()  # Ensure data is in float format
            labels = labels[:, 0].long()  # Select only the first column if labels are multi-target
            outputs = model(data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    return model

# Train the model
if __name__ == "__main__":
    trained_model = train_model()