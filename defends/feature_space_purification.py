import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureSpacePurification:
    """Feature Space Purification Defense"""
    
    def __init__(self, model, purification_strength=0.1, device='cuda'):
        self.model = model
        self.purification_strength = purification_strength
        self.device = device
        self.model.eval()
        
    def extract_features(self, images):
        """Extract features from model"""
        with torch.no_grad():
            # Get features from model
            if hasattr(self.model, 'features'):
                features = self.model.features(images)
            elif hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(images)
            else:
                # Use model output as features
                features = self.model(images)
        return features
    
    def purify_features(self, features):
        """Purify features by removing outliers"""
        # Calculate mean and std
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        
        # Clip features to within n standard deviations
        purified = torch.clamp(features, 
                              mean - self.purification_strength * std,
                              mean + self.purification_strength * std)
        
        return purified
    
    def defend(self, images):
        """Apply feature space purification"""
        images = images.clone().detach().to(self.device)
        
        # For now, apply simple denoising
        # In practice, this would involve feature extraction and purification
        defended_images = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        defended_images = torch.clamp(defended_images, 0, 1)
        
        return defended_images
