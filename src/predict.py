import torch
import cv2
import numpy as np
from torchvision import models
import torch.nn as nn
from utils import get_device

device = get_device()

# Load model
checkpoint = torch.load("models/model.pth")
classes = checkpoint["classes"]

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# Load image
img_path = "test.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0

img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)

print("Prediction:", classes[pred.item()])