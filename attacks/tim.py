import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as st

class TIMAttack:
    """Translation-Invariant Method (TIM) Attack"""
    
    def __init__(self, model, epsilon=0.031, alpha=0.008, iterations=10, kernel_size=5, device='cuda'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.device = device
        self.model.eval()
        
        # Create Gaussian kernel
        self.kernel = self.gkern(kernel_size).to(device)
        
    def gkern(self, kernlen=15, nsig=3):
        """Generate Gaussian kernel"""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kern2d = np.outer(kern1d, kern1d)
        kernel = kern2d / kern2d.sum()
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel
    
    def attack(self, images, labels=None):
        """Generate adversarial examples using TIM"""
        images = images.clone().detach().to(self.device)
        if labels is not None:
            labels = labels.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()
        
        for _ in range(self.iterations):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            if labels is not None:
                loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                loss = -outputs.max()
            
            grad = torch.autograd.grad(loss, adv_images)[0]
            
            # Apply Gaussian smoothing
            grad = F.conv2d(grad, self.kernel.repeat(grad.shape[1], 1, 1, 1), 
                          groups=grad.shape[1], padding=self.kernel_size//2)
            
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images
