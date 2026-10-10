import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class KYCDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Returns training and validation transforms according to Phase 1 spec (Section 3.5)."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def load_kyc_data(split="train", max_samples=None):
    """
    Loads the Identity-Isolated Deepfake dataset.
    Label mapping: 0 = Fake, 1 = Real
    """
    print(f"Loading KYC dataset ({split} split)...")
    dataset = load_dataset(
        "ThinothW/Deepfake-Identity-Isolated-Dataset-PreP", 
        split=split,
        trust_remote_code=True
    )
    
    if max_samples and max_samples < len(dataset):
        # Using a fixed seed for reproducible subset
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        
    return dataset

def get_dataloaders(batch_size=32, num_workers=0, train_samples=10000, val_samples=2000):
    """
    Returns train and val DataLoaders.
    Uses reproducible subsets as per the plan.
    """
    train_transform, val_transform = get_transforms()
    
    train_hf = load_kyc_data("train", max_samples=train_samples)
    val_hf = load_kyc_data("validation", max_samples=val_samples)
    
    train_dataset = KYCDataset(train_hf, transform=train_transform)
    val_dataset = KYCDataset(val_hf, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
