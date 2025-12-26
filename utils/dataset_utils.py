import os
import zipfile
import tarfile
import shutil
import sys
from pathlib import Path
from PIL import Image
import numpy as np

def extract_archive(archive_path, extract_dir):
    """Extract zip or tar archive"""
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar') or archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    return extract_dir

def get_images_from_dir(directory, limit=100):
    """Recursively find images in directory up to limit"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm')
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                if len(image_paths) >= limit:
                    return image_paths
    return image_paths

def prepare_dataset(data_path, limit=100):
    """Prepare dataset: handle archives, directories, and limit images"""
    # Convert to absolute path if needed
    data_path = Path(data_path).absolute()
    
    # Check if data_path exists
    if not data_path.exists():
        # Try relative to project root
        proj_root = Path(__file__).parent.parent
        data_path = (proj_root / data_path).absolute()
        
    if not data_path.exists():
        # Try to find it in common locations
        possible_paths = [
            Path("/data6/user23215430/nudt/input/data/LFW/lfw"),
            Path("/data6/user23215430/nudt/input/data/LFW"),
            Path("/data6/user23215430/nudt/input/data/web_face"),
            Path("/data6/user23215430/nudt/input/data/VGGFace2"),
            Path("/project/input/data"),
        ]
        for p in possible_paths:
            if p.exists():
                # If we are looking for a specific dataset, check if it matches
                if any(name.lower() in p.name.lower() for name in ["lfw", "webface", "vggface2"]):
                    data_path = p
                    break
                # Default to the first existing one if no match
                if not data_path.exists():
                    data_path = p

    # Check if there are zip files in the directory and extract them if needed
    if data_path.is_dir():
        zip_files = list(data_path.glob("*.zip"))
        for zf in zip_files:
            extract_dir = data_path / zf.stem
            if not extract_dir.exists():
                os.makedirs(extract_dir, exist_ok=True)
                extract_archive(str(zf), str(extract_dir))
    
    # Check if it's an archive itself
    if data_path.is_file() and data_path.suffix in ['.zip', '.tar', '.gz']:
        extract_dir = data_path.parent / (data_path.stem + "_extracted")
        if not extract_dir.exists():
            os.makedirs(extract_dir, exist_ok=True)
            extract_archive(str(data_path), str(extract_dir))
        data_path = extract_dir

    # Find images
    image_paths = get_images_from_dir(str(data_path), limit=limit)
    
    # If no images found, check if there's a subfolder with images
    if not image_paths and data_path.is_dir():
        for sub in data_path.iterdir():
            if sub.is_dir():
                image_paths.extend(get_images_from_dir(str(sub), limit=limit - len(image_paths)))
                if len(image_paths) >= limit:
                    break
                    
    return image_paths

def calculate_metrics(orig_img, adv_img):
    """Calculate PSNR, SSIM, L2 and L-inf norms between two images (numpy arrays, 0-1 range)"""
    mse = np.mean((orig_img - adv_img) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Simple SSIM approximation
    mu1 = orig_img.mean()
    mu2 = adv_img.mean()
    sigma1 = orig_img.var()
    sigma2 = adv_img.var()
    c1 = (0.01 * 1.0)**2
    c2 = (0.03 * 1.0)**2
    ssim = (2 * mu1 * mu2 + c1) * (2 * np.sqrt(sigma1 * sigma2) + c2) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    # Perturbation metrics
    diff = orig_img - adv_img
    l2_norm = np.linalg.norm(diff)
    linf_norm = np.max(np.abs(diff))
           
    return {
        "psnr": float(psnr), 
        "ssim": float(ssim),
        "l2_norm": float(l2_norm),
        "linf_norm": float(linf_norm),
        "combined_perturbation": float(l2_norm * 0.7 + linf_norm * 0.3)
    }

