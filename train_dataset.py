import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)

class PlateDataset(Dataset):
    def __init__(self, data_dir, img_size=(160, 160), is_training=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_training = is_training
        self.samples = []

        for img_file in os.listdir(data_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(data_dir, img_file)
                label_path = os.path.join(data_dir, os.path.splitext(img_file)[0] + '.txt')

                if os.path.exists(label_path):
                    self.samples.append((img_path, label_path))

    def augment_image(self, image):
        # Hızlı augmentasyon teknikleri
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)

        # Basit parlaklık değişimi
        if np.random.random() < 0.5:
            image = image * np.random.uniform(0.8, 1.2)
            image = np.clip(image, 0, 1)

        return image

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0

        if self.is_training:
            img = self.augment_image(img)

        img = torch.FloatTensor(img).permute(2, 0, 1)

        with open(label_path, 'r') as f:
            coords = list(map(float, f.read().strip().split(',')))
        label = torch.FloatTensor(coords)

        return img, label

    def __len__(self):
        return len(self.samples)

class OptimizedPlateDetectionModel(nn.Module):
    def __init__(self):
        super(OptimizedPlateDetectionModel, self).__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            # Second block
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            # Third block
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),

            # Fifth block
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 5 * 5, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - labels)).item()

        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_mae += torch.mean(torch.abs(outputs - labels)).item()

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start

        logging.info(f"Epoch {epoch+1}/{epochs}:")
        logging.info(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")
        logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            logging.info("New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

def main():
    logging.info("Starting training process...")

    # Optimized configuration
    img_size = (160, 160)  # Küçültülmüş görüntü boyutu
    batch_size = 32        # Artırılmış batch size
    epochs = 50           # Azaltılmış epoch sayısı
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_dataset = PlateDataset("prepared_dataset/train", img_size, is_training=True)
    val_dataset = PlateDataset("prepared_dataset/val", img_size, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    logging.info(f"Device: {device}")
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    logging.info(f"Image size: {img_size}")
    logging.info(f"Batch size: {batch_size}")

    model = OptimizedPlateDetectionModel().to(device)
    train_model(model, train_loader, val_loader, device, epochs)

    logging.info("Training completed!")

if __name__ == "__main__":
    main()
