import torch
import torch.nn as nn
import torch.nn.functional as F

class HGDDefense:
    """High-level Guided Denoising (HGD) Defense"""
    
    def __init__(self, model, iterations=5, step_size=0.01, device='cuda'):
        self.model = model
        self.iterations = iterations
        self.step_size = step_size
        self.device = device
        self.model.eval()
        
    def defend(self, images):
        """Apply HGD defense to images"""
        images = images.clone().detach().to(self.device)
        defended_images = images.clone().detach()
        
        for _ in range(self.iterations):
            defended_images.requires_grad = True
            outputs = self.model(defended_images)
            
            # Get predicted class
            pred_class = outputs.argmax(dim=1)
            
            # Maximize prediction confidence
            loss = -outputs[0, pred_class]
            
            # Calculate gradient
            grad = torch.autograd.grad(loss, defended_images)[0]
            
            # Update images
            defended_images = defended_images.detach() - self.step_size * grad.sign()
            defended_images = torch.clamp(defended_images, 0, 1).detach()
        
        return defended_images
