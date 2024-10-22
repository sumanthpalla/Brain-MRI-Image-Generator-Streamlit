import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from utils import calculate_ssim, calculate_psnr

class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train(config):
    # Load configuration
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    # Set up dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = BrainMRIDataset(config['data_dir'], transform=transform)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    # Load model
    model = StableDiffusionPipeline.from_pretrained(config['model_id'], torch_dtype=torch.float16)
    model.to(config['device'])

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(batch, return_dict=False)[0]
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        ssim_scores = []
        psnr_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Generate images
                generated_images = model(batch, return_dict=False)[0]
                
                # Calculate metrics
                for original, generated in zip(batch, generated_images):
                    ssim_scores.append(calculate_ssim(original, generated))
                    psnr_scores.append(calculate_psnr(original, generated))
                
                val_loss += model(batch, return_dict=False)[0].item()

        # Print epoch results
        print(f"Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"SSIM: {np.mean(ssim_scores):.4f}")
        print(f"PSNR: {np.mean(psnr_scores):.4f}")

    # Save the finetuned model
    model.save_pretrained("finetuned_model")

if __name__ == "__main__":
    train("config.yaml")
