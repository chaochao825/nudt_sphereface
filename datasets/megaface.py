import os
import glob
from .base_dataset import FaceDataset

class MegaFaceDataset(FaceDataset):
    """MegaFace Dataset"""
    
    def __init__(self, root_dir, transform=None, mode='train'):
        super().__init__(root_dir, transform, mode)
        self.load_dataset()
        
    def load_dataset(self):
        """Load MegaFace dataset"""
        data_dir = os.path.join(self.root_dir, 'megaface', 'FlickrFinal2')
        
        if not os.path.exists(data_dir):
            # print(f"Warning: MegaFace data directory not found: {data_dir}")
            return
        
        # Get all images recursively
        images = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True) + \
                glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
        
        for img_path in images:
            self.image_paths.append(img_path)
            # Use directory structure for label
            rel_path = os.path.relpath(img_path, data_dir)
            label = hash(os.path.dirname(rel_path)) % 10000
            self.labels.append(label)
