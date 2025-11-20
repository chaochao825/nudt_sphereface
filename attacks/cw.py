import torch
import torch.nn as nn
import torch.optim as optim

class CWAttack:
    """Carlini & Wagner (C&W) Attack"""
    
    def __init__(self, model, c=1e-4, kappa=0, max_iterations=100, learning_rate=0.01, device='cuda'):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.device = device
        self.model.eval()
        
    def attack(self, images, labels=None):
        """Generate adversarial examples using C&W"""
        images = images.clone().detach().to(self.device)
        if labels is not None:
            labels = labels.clone().detach().to(self.device)
        
        batch_size = images.shape[0]
        w = torch.zeros_like(images, requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        
        for i in range(self.max_iterations):
            adv_images = torch.tanh(w) * 0.5 + 0.5
            
            outputs = self.model(adv_images)
            
            if labels is not None:
                real = outputs.gather(1, labels.unsqueeze(1)).squeeze()
                other = outputs.clone()
                other.scatter_(1, labels.unsqueeze(1), -float('inf'))
                other_max = other.max(1)[0]
                f_loss = torch.clamp(real - other_max + self.kappa, min=0).sum()
            else:
                f_loss = -outputs.max()
            
            l2_loss = torch.norm(adv_images - images, p=2)
            loss = l2_loss + self.c * f_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        adv_images = torch.tanh(w) * 0.5 + 0.5
        return adv_images.detach()
