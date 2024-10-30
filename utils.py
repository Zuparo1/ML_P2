import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(data_dir, batch_size=32):
    transformNormalized = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformTraining = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        #NEW
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #NEW
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ##train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transformTraining)
   ## valid_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transformNormalized)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transformTraining)

    # Calculate lengths for each subset (80/10/10 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    class_names = full_dataset.classes
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Class names: {class_names}")

    return train_loader, val_loader,test_loader, class_names


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