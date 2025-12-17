import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class FaceDataset(Dataset):
    """Base class for face recognition datasets"""
    
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def load_dataset(self):
        """To be implemented by subclasses"""
        raise NotImplementedError
