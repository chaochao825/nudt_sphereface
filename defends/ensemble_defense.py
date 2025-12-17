import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleDefense:
    """Ensemble Defense combining multiple defense methods"""
    
    def __init__(self, defenses=None, device='cuda'):
        self.defenses = defenses if defenses else []
        self.device = device
        
    def add_defense(self, defense):
        """Add a defense method to the ensemble"""
        self.defenses.append(defense)
        
    def defend(self, images):
        """Apply ensemble defense"""
        images = images.clone().detach().to(self.device)
        
        if not self.defenses:
            # If no defenses, apply simple preprocessing
            defended_images = self.simple_preprocess(images)
        else:
            # Apply all defenses and average results
            defended_images = torch.zeros_like(images)
            for defense in self.defenses:
                defended_images += defense.defend(images)
            defended_images /= len(self.defenses)
        
        defended_images = torch.clamp(defended_images, 0, 1)
        return defended_images
    
    def simple_preprocess(self, images):
        """Simple preprocessing as fallback"""
        # Apply median filter approximation
        defended = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        return defended
