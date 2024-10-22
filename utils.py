import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM) between two images."""
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    return ssim(img1, img2, multichannel=True)

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    return psnr(img1, img2)
