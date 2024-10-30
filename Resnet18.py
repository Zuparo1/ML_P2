
import torch
from torchvision import models
import torch.optim as optim
from utils import get_data_loaders, train_model, save_model, predict_image, device

data_dir = "split_data"
train_loader, valid_loader, class_names = get_data_loaders(data_dir)

#####################################################################################################
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_classes = len(class_names)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

#Loss and Optimization
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
#####################################################################################################

train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10)
save_model(model, "models/mushroom_resnet18.pth")
#pred
image_path = "split_data/test/butter_cap/5.png"
prediction = predict_image(model, image_path, train_loader.dataset.transform, class_names)
print("Predicted class:", prediction)