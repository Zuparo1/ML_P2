import os
import shutil
import random

# Define the main directory and target directories
data_dir = "C:/Users/Zupar/PycharmProjects/Mushroom/data/mushroom/data/data"
train_dir = "C:/Users/Zupar/PycharmProjects/Mushroom/split_data/train/"
valid_dir = "C:/Users/Zupar/PycharmProjects/Mushroom/split_data/test/"

# Create train and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Split ratio
split_ratio = 0.8  # 80% train, 20% validation

# Iterate through each class folder and split the images
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        # Make train/valid subdirectories for each class
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

        # Get all images in the class directory
        images = os.listdir(class_path)
        random.shuffle(images)

        # Determine split index
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        valid_images = images[split_idx:]

        # Move images to train and valid directories
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

        for img in valid_images:
            shutil.move(os.path.join(class_path, img), os.path.join(valid_dir, class_name, img))

print("Dataset split completed successfully.")