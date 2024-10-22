# Loading a Dataset
print("Loading a Dataset")

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data",  # Location on disk where the data is located
    train = True,   # Training set, if False, it returns the test set
    download = True,    # Downloads the data from the internet if it's not available at root
    transform = ToTensor()  # Transfer an image (May from Numpy [0 - 255]) to Tensor [0 - 1])
)

test_data = datasets.FashionMNIST(
    root = "data",  # Location on disk where the data is located
    train = False,   # Test set
    download = True,    # Downloads the data from the internet if it's not available at root
    transform = ToTensor()  # Transfer an image (May from Numpy [0 - 255]) to Tensor [0 - 1])
)

# Iterating and Visualizing the Dataset
print("Iterating and Visualizing the Dataset")

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(5, 5))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Creating a Custom Dataset for your files
    # Optional for this tutorial, since datasets.FashionMNIST is built-in
print("Creating a Custom Dataset for your files")

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    # Initialize the directory containing the images, the annotations file, and both transforms
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # Returns the number of samples in our dataset
    def __len__(self):
        return len(self.img_labels)
    
    # Loads and returns a sample from the dataset at the given index “idx”
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)    # Read an image from a file path into a tensor
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Preparing your data for training with DataLoaders
print("Preparing your data for training with DataLoaders")

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

# Iterate through the DataLoader
print("Iterate through the DataLoader")

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")