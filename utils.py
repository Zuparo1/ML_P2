import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

def get_data_loaders(data_dir, batch_size=32):
    transformTraining = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    full_dataset = datasets.ImageFolder(root=data_dir, transform=transformTraining)

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = full_dataset.classes
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Class names: {class_names}")

    return train_loader, val_loader,test_loader, class_names


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, epochs=10):
    print("Training started")
    start_time = time.perf_counter()
    prev_lr = optimizer.param_groups[0]['lr']

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
        val_loss = 0 #
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0) #
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == labels).item()


        val_loss /= len(valid_loader.dataset) #
        val_accuracy = accuracy / len(valid_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Training loss: {running_loss / len(train_loader):.4f}, "
              #f"Validation loss: {val_loss:.4f}, "  #
              f"Validation accuracy: {val_accuracy:.4f}")

        if scheduler:
           # scheduler.step() #
            scheduler.step(val_loss) #

            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                print(f"Learning rate reduced from {prev_lr:.6f} to {current_lr:.6f}")
                prev_lr = current_lr  # Update previous learning rate

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time} seconds")

def evaluate_model(model, test_loader):
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_accuracy += torch.sum(preds == labels).item()
    accuracy = test_accuracy / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

#DOES NOT WORK
def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]