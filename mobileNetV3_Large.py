import time

import numpy as np
import torch
from lion_pytorch import Lion
from sympy.physics.mechanics import System
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import os
from sklearn.utils import resample
from torch.utils.data import DataLoader, Subset


from utils import get_data_loaders, train_model, save_model, predict_image, device, evaluate_model

data_dir = "data"
train_loader ,valid_loader, test_loader, class_names = get_data_loaders(data_dir)

#Model
model = models.mobilenet_v3_large(pretrained=True)
for param in model.parameters():
    param.requires_grad = False #

num_classes = len(class_names)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()


#optimizers
optimizer = optim.Adam(model.classifier[3].parameters(), lr=0.001)
#optimizer = Lion(model.parameters(), lr=0.001, betas=(0.9, 0.99))


#Schedulers
scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.1, patience=10, threshold=1e-1, min_lr=1e-6)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=scheduler ,epochs=100)

# Start Confidence intervals
print("Evaluating with resampled test subsets...")
test_dataset = test_loader.dataset
num_samples = 100
test_accuracies = []

for i in range(num_samples):
    resampled_indices = resample(range(len(test_dataset)), n_samples=len(test_dataset) // 2, replace=False)
    resampled_test_loader = DataLoader(Subset(test_dataset, resampled_indices), batch_size=32, shuffle=False)

    test_accuracy = evaluate_model(model, resampled_test_loader)
    test_accuracies.append(test_accuracy)
    print(f"Sample {i + 1}/{num_samples}: Test Accuracy = {test_accuracy:.4f}")

mean_accuracy = np.mean(test_accuracies)
lower_bound = np.percentile(test_accuracies, 2.5)
upper_bound = np.percentile(test_accuracies, 97.5)

print("\nFinal Results:")
print(f"Mean Test Accuracy: {mean_accuracy:.4f}")
print(f"95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
# End Confidence intervals


#evaluate_model(model, test_loader)
save_model(model, "models/mushroom_mobilenetV3_Large.pth")



