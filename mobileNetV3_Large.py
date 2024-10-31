import time
import torch
from sympy.physics.mechanics import System
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import os
from utils import get_data_loaders, train_model, save_model, predict_image, device, evaluate_model

data_dir = "data"
train_loader ,valid_loader, test_loader, class_names = get_data_loaders(data_dir)

#####################################################################################################
model = models.mobilenet_v3_large(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_classes = len(class_names)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[3].parameters(), lr=0.001) #Lower learing rate
#####################################################################################################

train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10)

evaluate_model(model, test_loader)

save_model(model, "models/mushroom_mobilenetV3_Large.pth")
#pred


