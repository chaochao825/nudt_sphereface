import os
import glob
from .base_dataset import FaceDataset

class YaleBDataset(FaceDataset):
    """Yale Face Database B Dataset"""
    
    def __init__(self, root_dir, transform=None, mode='train'):
        super().__init__(root_dir, transform, mode)
        self.load_dataset()
        
    def load_dataset(self):
        """Load YaleB dataset"""
        data_dir = os.path.join(self.root_dir, 'yaleb', 'CroppedYale')
        
        if not os.path.exists(data_dir):
            # print(f"Warning: YaleB data directory not found: {data_dir}")
            return
        
        # Get all person directories
        person_dirs = sorted(glob.glob(os.path.join(data_dir, 'yaleB*')))
        
        for label, person_dir in enumerate(person_dirs):
            if os.path.isdir(person_dir):
                images = glob.glob(os.path.join(person_dir, '*.pgm')) + \
                        glob.glob(os.path.join(person_dir, '*.jpg'))
                
                for img_path in images:
                    self.image_paths.append(img_path)
                    self.labels.append(label)
