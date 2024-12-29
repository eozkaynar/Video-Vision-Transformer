import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from EchoSet import EchoSet
import sklearn.metrics
import tqdm
import pandas as pd

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 30
INPUT_SHAPE = (3, 128, 112, 112)
PATCH_SIZE = (16, 16, 16)
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

# Tubelet Embedding
class TubeletEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super(TubeletEmbedding, self).__init__()
        self.projection = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)

# Positional Encoder
class PositionalEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        position = torch.arange(0, num_patches, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32, device=x.device) * 
                             -(torch.log(torch.tensor(10000.0)) / embed_dim))

        pos_encoding = torch.zeros((num_patches, embed_dim), device=x.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        return x + pos_encoding

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

# Video Vision Transformer
class ViViT(nn.Module):
    def __init__(self, input_shape, patch_size, embed_dim, num_heads, num_layers):
        super(ViViT, self).__init__()
        num_patches = (
            (input_shape[1] // patch_size[0]) * 
            (input_shape[2] // patch_size[1]) * 
            (input_shape[3] // patch_size[2])
        )
        self.embedding = TubeletEmbedding(embed_dim, patch_size)
        self.pos_encoder = PositionalEncoder(num_patches, embed_dim)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.transformer:
            x = layer(x)
        x = x.mean(dim=1)
        return self.regressor(x).squeeze()

# Training, Validation, and Test Function
def run_experiment(data_dir, model, criterion, optimizer, train_loader, val_loader, test_loader, epochs=10):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} - Training") as pbar:
            for (filename, video, label, ejection, repeat, fps) in train_loader:
                video, ejection = video.float().to("cuda"), ejection.float().to("cuda")
                video = video.permute(0, 2, 1, 3, 4)  # [Batch, Channel, Depth, Height, Width]
                optimizer.zero_grad()
                outputs = model(video)
                loss = criterion(outputs, ejection)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * video.size(0)
                pbar.update(1)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            with tqdm.tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{epochs} - Validation") as pbar:
                for (filename, video, label, ejection, repeat, fps) in val_loader:
                    video, ejection = video.float().to("cuda"), ejection.float().to("cuda")
                    video = video.permute(0, 2, 1, 3, 4)  # [Batch, Channel, Depth, Height, Width]
                    outputs = model(video)
                    loss = criterion(outputs, ejection)
                    val_loss += loss.item() * video.size(0)
                    y_true.extend(ejection.cpu().numpy())
                    y_pred.extend(outputs.cpu().numpy())
                    pbar.update(1)
        val_loss /= len(val_loader.dataset)
        val_r2_score = sklearn.metrics.r2_score(y_true, y_pred)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, R2 Score: {val_r2_score:.4f}")

    # Test Phase
    model.eval()
    test_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_loader), desc="Testing") as pbar:
            for (filename, video, label, ejection, repeat, fps) in test_loader:
                video, ejection = video.float().to("cuda"), ejection.float().to("cuda")
                video = video.permute(0, 2, 1, 3, 4)  # [Batch, Channel, Depth, Height, Width]
                outputs = model(video)
                loss = criterion(outputs, ejection)
                test_loss += loss.item() * video.size(0)
                y_true.extend(ejection.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
                pbar.update(1)
    test_loss /= len(test_loader.dataset)
    test_r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    print(f"Test Loss: {test_loss:.4f}, R2 Score: {test_r2_score:.4f}")

# Main execution
if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    output_dir = "output"

    # Initialize model and criterion
    model = ViViT(INPUT_SHAPE, PATCH_SIZE, PROJECTION_DIM, NUM_HEADS, NUM_LAYERS).to("cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load datasets
    train_dataset = EchoSet(root=data_dir, split="train", random_clip=True)
    val_dataset = EchoSet(root=data_dir, split="val", random_clip=False)
    test_dataset = EchoSet(root=data_dir, split="test", random_clip=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Run experiment
    run_experiment(
        data_dir=data_dir,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=10,
    )
