import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Path to the base data directory
data_dir = "C:/Users/Zupar/PycharmProjects/Mushroom/.venv/Lib/data/mushroom/data/data"

# Load the entire dataset from the directory, assuming each folder represents a class
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# Define split ratio for train and validation
train_size = int(0.8 * len(full_dataset))  # 80% for training
valid_size = len(full_dataset) - train_size

# Split the dataset
train_data, valid_data = random_split(full_dataset, [train_size, valid_size])

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

# Classes (derived from folder names)
class_names = full_dataset.classes
print("Class names:", class_names)