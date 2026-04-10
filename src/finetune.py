import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from torchvision import transforms

# =====================
# CONFIG
# =====================
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = 224

DATA_PATH = "/content/clean_data/train"  # 🔥 CLEAN DATA ONLY

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =====================
# TRANSFORMS (WITH NORMALIZATION)
# =====================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# DATASET
# =====================
train_dataset = datasets.ImageFolder(DATA_PATH, transform=train_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,   # 🔥 FIXED (your system warning)
    pin_memory=True
)

print("\nFinal Classes:")
for c in train_dataset.classes:
    print(c)

# =====================
# MODEL
# =====================
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))

# 🔥 Unfreeze last layers for finetuning
for param in model.features[-6:].parameters():
    param.requires_grad = True

model = model.to(device)

# =====================
# LOSS & OPTIMIZER
# =====================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

# Mixed precision (new API)
scaler = torch.amp.GradScaler('cuda')

# =====================
# TRAINING LOOP
# =====================
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

    model.train()
    running_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch Loss: {epoch_loss:.4f}")

# =====================
# SAVE MODEL
# =====================
os.makedirs("models", exist_ok=True)

torch.save({
    "model_state": model.state_dict(),
    "classes": train_dataset.classes
}, "models/model_final.pth")

print("\n✅ Final model saved: models/model_final.pth")