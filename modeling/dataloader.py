# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 
import torch.nn.functional as F

# For all Optimization algorithms, SGD, Adam, etc.
import torch.optim as optim

# Loading and Performing transformations on dataset
import torchvision
import torchvision.transforms as transforms 
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import pandas as pd
from PIL import Image

class MyJP2Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        hmi = Image.open(img_path)

        if self.transform:
            image = self.transform(hmi)
            
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        return (image, y_label)

    def __len__(self):
        return len(self.annotations)
    
class NFDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        iter_csv = pd.read_csv(csv_file, iterator=True)
        df = pd.concat([chunk[chunk['goes_class'] == 0] for chunk in iter_csv])
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        hmi = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        #print(y_label)

        if self.transform:
            image = self.transform(hmi)
            
        return (image, y_label)

    def __len__(self):
        return len(self.annotations)

class FLDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        iter_csv = pd.read_csv(csv_file, iterator=True)
        df = pd.concat([chunk[chunk['goes_class'] == 1] for chunk in iter_csv])
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        hmi = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        #print(y_label)

        if self.transform:
            image = self.transform(hmi)
            
        return (image, y_label)

    def __len__(self):
        return len(self.annotations)
    

    
# class Balancer(Dataset):
#     def __init__(self, csv_file, root_dir, len, transform=None):
#         iter_csv = pd.read_csv(csv_file, iterator=True)
#         df = pd.concat([chunk[chunk['goes_class'] == 1] for chunk in iter_csv])
#         self.annotations = df
#         self.root_dir = root_dir
#         self.transform = transform
#         self.len = len

#     def __getitem__(self, index):
#         img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
#         hmi = Image.open(img_path)
#         y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
#         #print(y_label)

#         if self.transform:
#             image = self.transform(hmi)
            
#         return (image, y_label)

#     def __len__(self):
#         return self.len

class Balancer(Dataset):
    def __init__(self, csv_file, root_dir, length, transform=None):
        # Read the CSV file containing image paths and labels.
        df = pd.read_csv(csv_file)

        # Keep only flare samples where goes_class is 1.
        self.annotations = df[df["goes_class"] == 1].reset_index(drop=True)

        # Directory containing the actual magnetogram images.
        self.root_dir = root_dir

        # Image transformation, such as resize or ToTensor.
        self.transform = transform

        # Number of additional flare samples needed for balancing.
        # This can be larger than the number of real flare images.
        self.length = length

    def __getitem__(self, index):
        # Convert the requested index into a valid flare-image index.
        #
        # Example:
        # If there are 3 flare images, indices 3, 4 and 5 become 0, 1 and 2.
        # This allows the existing flare images to be repeated safely.
        actual_index = index % len(self.annotations)

        # Get the image path of the selected flare sample.
        relative_path = self.annotations.iloc[actual_index, 0]

        # Combine the image directory with the relative image path.
        img_path = os.path.join(self.root_dir, relative_path)

        # Open the magnetogram image.
        image = Image.open(img_path)

        # Read its label. This will be 1 because Balancer contains only flares.
        y_label = torch.tensor(
            int(self.annotations.iloc[actual_index, 1])
        )

        # Apply transformations when provided.
        if self.transform:
            image = self.transform(image)

        # Return the image and its label.
        return image, y_label

    def __len__(self):
        # Tell DataLoader how many additional flare samples to request.
        return self.length