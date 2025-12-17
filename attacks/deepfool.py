import torch
import torch.nn as nn

class DeepFoolAttack:
    """DeepFool Attack"""
    
    def __init__(self, model, max_iterations=50, overshoot=0.02, device='cuda'):
        self.model = model
        self.max_iterations = max_iterations
        self.overshoot = overshoot
        self.device = device
        self.model.eval()
        
    def attack(self, images, labels=None):
        """Generate adversarial examples using DeepFool"""
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        
        for i in range(self.max_iterations):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            pred = outputs.argmax(dim=1)
            
            if labels is not None and pred.item() != labels.item():
                break
            
            outputs[0, pred].backward(retain_graph=True)
            grad_orig = adv_images.grad.data.clone()
            
            min_val = float('inf')
            min_grad = None
            
            for k in range(outputs.size(1)):
                if k == pred:
                    continue
                
                adv_images.grad.zero_()
                outputs[0, k].backward(retain_graph=True)
                grad_k = adv_images.grad.data.clone()
                
                w_k = grad_k - grad_orig
                f_k = outputs[0, k] - outputs[0, pred]
                value = abs(f_k) / (torch.norm(w_k) + 1e-8)
                
                if value < min_val:
                    min_val = value
                    min_grad = w_k
            
            if min_grad is not None:
                r = (min_val + 1e-4) * min_grad / (torch.norm(min_grad) + 1e-8)
                adv_images = adv_images.detach() + (1 + self.overshoot) * r
                adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
