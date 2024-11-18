import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T

import os
import rasterio
import numpy as np

# Clase para cargar datos
class My_dataloader(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = sorted(os.listdir(os.path.join(data_dir, "images")))
        self.mask_dir = sorted(os.listdir(os.path.join(data_dir, "masks")))
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        img_path=os.path.join(self.data_dir, 'images', self.image_dir[idx])
        mask_path=os.path.join(self.data_dir, 'masks', self.mask_dir[idx])
        #image_name = img_path[idx].split('/')[-1]  # Obtiene el nombre de la imagen
        with rasterio.open(img_path) as img_dataset:
            # Leemos la imagen multiespectral
            img = img_dataset.read()
            # Normalizamos la imagen para que se encuentre en el rango de [0,1]
            img = img.astype(np.float32) / np.max(img)
            # Converción a tensor de pytorch
            img = torch.tensor(img)
        
        with rasterio.open(mask_path) as mask_dataset:
            # Leemos la mascara
            mask = mask_dataset.read()
            # Normalizamos la mascara para que se encuentre en el rango de [0,1]
            #mask = mask.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / np.max(mask)
            # Converción a tensor de pytorch
            mask = torch.tensor(mask)
        
        return img, mask
    

    
        
"""
class My_dataloader2(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = sorted(os.listdir(os.path.join(data_dir, "images")))
        self.mask_dir = sorted(os.listdir(os.path.join(data_dir, "masks")))
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        img_path=os.path.join(self.data_dir, 'images', self.image_dir[idx])
        mask_path=os.path.join(self.data_dir, 'masks', self.mask_dir[idx])
        image_name = img_path[idx].split('/')[-1]  # Obtiene el nombre de la imagen
        with rasterio.open(img_path) as img_dataset:
            # Leemos la imagen multiespectral
            img = img_dataset.read()
            # Normalizamos la imagen para que se encuentre en el rango de [0,1]
            img = img.astype(np.float32) / 65535.0
            # Converción a tensor de pytorch
            img = torch.tensor(img)
        
        with rasterio.open(mask_path) as mask_dataset:
            # Leemos la mascara
            mask = mask_dataset.read()
            # Normalizamos la mascara para que se encuentre en el rango de [0,1]
            mask = mask.astype(np.float32) / 255.0
            # Converción a tensor de pytorch
            mask = torch.tensor(mask)
        
        return img, mask, image_name
        #return img, mask
"""