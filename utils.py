import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    valid_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    class_names = train_data.classes
    print(f"Number of training images: {len(train_data)}")
    print(f"Number of validation images: {len(valid_data)}")
    print(f"Class names: {class_names}")

    return train_loader, valid_loader, class_names


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    print("Training started")
    start_time = time.perf_counter()

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
              f"Validation accuracy: {accuracy / len(valid_loader.dataset):.4f}")

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time} seconds")


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]