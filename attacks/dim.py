import torch
import torch.nn as nn
import torch.nn.functional as F

class DIMAttack:
    """Diverse Input Method (DIM) Attack"""
    
    def __init__(self, model, epsilon=0.031, alpha=0.008, iterations=10, diversity_prob=0.5, device='cuda'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.diversity_prob = diversity_prob
        self.device = device
        self.model.eval()
        
    def input_diversity(self, x):
        """Apply input diversity transformation"""
        if torch.rand(1) < self.diversity_prob:
            img_size = x.shape[-1]
            img_resize = int(img_size * torch.rand(1) * 0.2 + img_size * 0.8)
            x = F.interpolate(x, size=(img_resize, img_resize), mode='bilinear', align_corners=False)
            pad_left = torch.randint(0, img_size - img_resize + 1, (1,)).item()
            pad_top = torch.randint(0, img_size - img_resize + 1, (1,)).item()
            x = F.pad(x, (pad_left, img_size - img_resize - pad_left, 
                         pad_top, img_size - img_resize - pad_top))
        return x
    
    def attack(self, images, labels=None):
        """Generate adversarial examples using DIM"""
        images = images.clone().detach().to(self.device)
        if labels is not None:
            labels = labels.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()
        
        for _ in range(self.iterations):
            adv_images.requires_grad = True
            diverse_images = self.input_diversity(adv_images)
            outputs = self.model(diverse_images)
            
            if labels is not None:
                loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                loss = -outputs.max()
            
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images
