import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdversarialDetector:
    """Adversarial Example Detector Defense"""
    
    def __init__(self, model=None, threshold=0.15, device='cuda'):
        self.model = model
        self.threshold = threshold
        self.device = device
        if self.model:
            self.model.eval()
        
    def detect_adversarial(self, images):
        """Detect if images are adversarial examples"""
        images = images.clone().detach().to(self.device)
        
        # Method 1: Check prediction confidence
        if self.model:
            with torch.no_grad():
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                
                # Low confidence indicates potential adversarial
                is_adversarial = max_probs < (1.0 - self.threshold)
        else:
            # Method 2: Statistical analysis
            # Check for unusual patterns in pixel distributions
            batch_size = images.size(0)
            is_adversarial = torch.zeros(batch_size, dtype=torch.bool)
            
            for i in range(batch_size):
                img = images[i]
                
                # Calculate local variance
                local_var = []
                for c in range(3):
                    channel = img[c]
                    # Compute variance in small patches
                    patches = channel.unfold(0, 8, 8).unfold(1, 8, 8)
                    patch_var = patches.reshape(-1, 64).var(dim=1).mean()
                    local_var.append(patch_var.item())
                
                avg_var = np.mean(local_var)
                
                # If variance is too high or too low, might be adversarial
                if avg_var > self.threshold or avg_var < 0.001:
                    is_adversarial[i] = True
        
        return is_adversarial
    
    def defend(self, images):
        """Apply adversarial detection"""
        is_adversarial = self.detect_adversarial(images)
        
        # Return original if clean, otherwise apply denoising
        defended_images = images.clone()
        for i in range(len(images)):
            if is_adversarial[i]:
                # Apply Gaussian smoothing to potentially adversarial images
                img = defended_images[i:i+1]
                img = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode='replicate'), 
                                  kernel_size=3, stride=1)
                defended_images[i] = img[0]
        
        defended_images = torch.clamp(defended_images, 0, 1)
        return defended_images

