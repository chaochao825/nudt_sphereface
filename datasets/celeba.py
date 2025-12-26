import os
import glob
from .base_dataset import FaceDataset

class CelebADataset(FaceDataset):
    """CelebA Dataset"""
    
    def __init__(self, root_dir, transform=None, mode='train'):
        super().__init__(root_dir, transform, mode)
        self.load_dataset()
        
    def load_dataset(self):
        """Load CelebA dataset"""
        data_dir = os.path.join(self.root_dir, 'celeba', 'img_align_celeba')
        
        if not os.path.exists(data_dir):
            # print(f"Warning: CelebA data directory not found: {data_dir}")
            return
        
        # Get all images
        images = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
        
        # Split into train/test
        split_idx = int(len(images) * 0.8)
        
        if self.mode == 'train':
            images = images[:split_idx]
        else:
            images = images[split_idx:]
        
        for img_path in images:
            self.image_paths.append(img_path)
            # Use filename as label (identity)
            label = int(os.path.basename(img_path).split('.')[0]) % 1000
            self.labels.append(label)
