import torch
import cv2
import numpy as np
from torchvision.models import mobilenet_v2
import torch.nn as nn
from utils import get_device

device = get_device()

# =====================
# LOAD MODEL
# =====================
checkpoint = torch.load("models/model_final.pth",map_location=torch.device('cpu'))
classes = checkpoint["classes"]

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# =====================
# LOAD IMAGE
# =====================
img_path = "test_image/test3.jpg"

img = cv2.imread(img_path)

# ❗ Fix BGR → RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (224, 224))
img = img / 255.0

# =====================
# NORMALIZATION (CRITICAL)
# =====================
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

img = (img - mean) / std

# =====================
# TO TENSOR
# =====================
img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

# =====================
# PREDICTION
# =====================
with torch.no_grad():
    output = model(img)

    probs = torch.nn.functional.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)

print("Prediction:", classes[pred.item()])
print("Confidence:", f"{confidence.item():.2f}")