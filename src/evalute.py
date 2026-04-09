import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import get_transforms, get_device

device = get_device()

# Load model
checkpoint = torch.load("models/model.pth")
classes = checkpoint["classes"]

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# Load test data
test_dataset = datasets.ImageFolder("data/test", transform=get_transforms(False))
test_loader = DataLoader(test_dataset, batch_size=32)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")