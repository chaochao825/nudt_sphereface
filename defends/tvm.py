import torch
import torch.nn as nn
import torch.nn.functional as F

class TVMDefense:
    """Total Variation Minimization (TVM) Defense"""
    
    def __init__(self, weight=0.03, iterations=10, learning_rate=0.01, device='cuda'):
        self.weight = weight
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.device = device
        
    def total_variation(self, images):
        """Calculate total variation"""
        diff_i = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        diff_j = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        tv = diff_i.sum() + diff_j.sum()
        return tv
    
    def defend(self, images):
        """Apply TVM defense to images"""
        images = images.clone().detach().to(self.device)
        defended_images = images.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([defended_images], lr=self.learning_rate)
        
        for _ in range(self.iterations):
            optimizer.zero_grad()
            
            # L2 loss to keep close to original
            l2_loss = torch.norm(defended_images - images, p=2)
            
            # Total variation loss
            tv_loss = self.total_variation(defended_images)
            
            # Combined loss
            loss = l2_loss + self.weight * tv_loss
            
            loss.backward()
            optimizer.step()
            
            # Clamp values
            defended_images.data = torch.clamp(defended_images.data, 0, 1)
        
        return defended_images.detach()
