import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LivenessDetection:
    """Liveness Detection Defense for Face Recognition"""
    
    def __init__(self, model=None, threshold=0.5, device='cuda'):
        self.model = model
        self.threshold = threshold
        self.device = device
        
    def check_liveness(self, images):
        """Check if face is live or spoofed"""
        images = images.clone().detach().to(self.device)
        
        # Simple texture analysis
        # Real faces have more texture variation than adversarial examples
        
        # Calculate local binary patterns (simplified)
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        
        # Calculate variance as texture measure
        variance = torch.var(gray_images, dim=[1, 2])
        
        # Check if variance is above threshold
        is_live = variance > self.threshold
        
        return is_live
    
    def defend(self, images):
        """Apply liveness detection"""
        is_live = self.check_liveness(images)
        
        # Return original if live, otherwise return blank
        defended_images = images.clone()
        for i in range(len(images)):
            if not is_live[i]:
                defended_images[i] = torch.zeros_like(defended_images[i])
        
        return defended_images
