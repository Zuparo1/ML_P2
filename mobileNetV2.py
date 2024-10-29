import time

import torch
from sympy.physics.mechanics import System
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import os
from utils import get_data_loaders, train_model, save_model, predict_image, device

data_dir = "split_data"
train_loader, valid_loader, class_names = get_data_loaders(data_dir)

#####################################################################################################
model = models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_classes = len(class_names)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)
#####################################################################################################

train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10)
save_model(model, "mushroom_mobilenetV2.pth.pth")

#pred
image_path = "split_data/test/butter_cap/5.png"
prediction = predict_image(model, image_path, train_loader.dataset.transform, class_names)
print("Predicted class:", prediction)