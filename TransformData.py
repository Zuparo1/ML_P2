import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#step 1
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#step 2
data_dir = "C:/Users/Zupar/PycharmProjects/Mushroom/split_data/"
train_data = datasets.ImageFolder(root=data_dir + 'train', transform=transform)
valid_data = datasets.ImageFolder(root=data_dir + 'test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

class_names = train_data.classes

print(f"Number of training images: {len(train_data)}")
print(f"Number of validation images: {len(valid_data)}")
print(f"Class names: {class_names}")

images, labels = next(iter(train_loader))

############################## step 3
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_classes = len(train_data.classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

#step 4
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

#Step 5

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation Phase
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            accuracy += torch.sum(preds == labels).item()

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Training loss: {running_loss / len(train_loader):.4f}, "
          f"Validation accuracy: {accuracy / len(valid_data):.4f}")

#step 6
torch.save(model.state_dict(), "mushroom_resnet18.pth")

#7
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return train_data.classes[predicted.item()]

#pref
print(predict("C:/Users/Zupar/PycharmProjects/Mushroom/split_data/test/butter_cap/5.png"))