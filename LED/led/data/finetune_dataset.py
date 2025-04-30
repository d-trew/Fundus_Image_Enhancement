import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class FineTuneDataset(Dataset):
    def __init__(self, good_dir, bad_dir, image_size=512):
        self.good_images = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.bad_images = [os.path.join(bad_dir, f) for f in os.listdir(bad_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Ensure we have matching pairs
        assert len(self.good_images) == len(self.bad_images), "Mismatched number of good and bad images"
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.good_images)
    
    def __getitem__(self, idx):
        # Load good and bad images
        good_img = Image.open(self.good_images[idx]).convert('RGB')
        bad_img = Image.open(self.bad_images[idx]).convert('RGB')
        
        # Apply transformations
        good_img = self.transform(good_img)
        bad_img = self.transform(bad_img)
        
        # Concatenate along channel dimension to create 6-channel input
        combined = torch.cat([bad_img, good_img], dim=0)
        
        return combined, good_img  # (6-channel input, 3-channel target)