import os
import glob
from .base_dataset import FaceDataset

class VGGFace2Dataset(FaceDataset):
    """VGGFace2 Dataset"""
    
    def __init__(self, root_dir, transform=None, mode='train'):
        super().__init__(root_dir, transform, mode)
        self.load_dataset()
        
    def load_dataset(self):
        """Load VGGFace2 dataset"""
        data_dir = os.path.join(self.root_dir, 'vggface2', self.mode)
        
        if not os.path.exists(data_dir):
            # print(f"Warning: VGGFace2 data directory not found: {data_dir}")
            return
        
        # Get all subdirectories (each is a person)
        person_dirs = sorted(glob.glob(os.path.join(data_dir, '*')))
        
        for label, person_dir in enumerate(person_dirs):
            if os.path.isdir(person_dir):
                # Get all images for this person
                images = glob.glob(os.path.join(person_dir, '*.jpg')) + \
                        glob.glob(os.path.join(person_dir, '*.png'))
                
                for img_path in images:
                    self.image_paths.append(img_path)
                    self.labels.append(label)
