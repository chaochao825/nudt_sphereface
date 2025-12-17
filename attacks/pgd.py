import torch
import torch.nn as nn

class PGDAttack:
    """Projected Gradient Descent (PGD) Attack"""
    
    def __init__(self, model, epsilon=0.031, alpha=0.008, iterations=10, random_start=True, device='cuda'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.random_start = random_start
        self.device = device
        self.model.eval()
        
    def attack(self, images, labels=None):
        """Generate adversarial examples using PGD"""
        images = images.clone().detach().to(self.device)
        if labels is not None:
            labels = labels.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()
        
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        for _ in range(self.iterations):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            if labels is not None:
                loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                loss = -outputs.max()
            
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images
