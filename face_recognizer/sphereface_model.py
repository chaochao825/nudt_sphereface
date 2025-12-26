import torch
import torch.nn as nn
import torchvision.models as models

class SphereFaceModel(nn.Module):
    """SphereFaceModel for Face Recognition"""
    
    def __init__(self, num_classes=1000, pretrained_path=None):
        super(SphereFaceModel, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=False)
        
        # Replace last layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add embedding layer
        self.embedding = nn.Linear(num_features, 512)
        
        # Classification layer
        self.fc = nn.Linear(512, num_classes)
        
        if pretrained_path:
            try:
                self.load_state_dict(torch.load(pretrained_path))
            except:
                pass
                # print(f"Warning: Could not load pretrained weights from {pretrained_path}")
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.fc(x)
        return x
    
    def forward_features(self, x):
        """Extract features without classification"""
        x = self.backbone(x)
        x = self.embedding(x)
        return x
