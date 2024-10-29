import time
import torch
from torchvision import models
import torch.optim as optim
from PIL import Image
from utils import get_data_loaders, train_model, save_model, predict_image, device

data_dir = "split_data"
train_loader, valid_loader, class_names = get_data_loaders(data_dir)

#####################################################################################################
model = models.mobilenet_v3_small(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_classes = len(class_names)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[3].parameters(), lr=0.001) #Lower learing rate
#####################################################################################################

train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10)
save_model(model, "mushroom_mobilenetV3_Small.pth")
#pred
image_path = "split_data/test/butter_cap/5.png"
prediction = predict_image(model, image_path, train_loader.dataset.transform, class_names)
print("Predicted class:", prediction)