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

def prepare_dataset(data_path, limit=100, dataset_name=None):
    """Prepare dataset: handle archives, directories, and robustly search for specific datasets"""
    data_path = Path(data_path).absolute()
    
    # 1. Search for datasets in common locations
    search_roots = [
        data_path, 
        data_path.parent, 
        Path("/project/input/data"),
        Path("/data6/user23215430/nudt/input/data")
    ]
    
    found_path = None
    
    # Priority search for the specific dataset name if provided
    if dataset_name:
        for root in search_roots:
            if not root.exists() or not root.is_dir(): continue
            # Check for exact or fuzzy match in subdirectories
            for item in root.iterdir():
                if item.is_dir() and (dataset_name.lower() in item.name.lower()):
                    found_path = item
                    break
            if found_path: break
            
    # If not found yet, just try to find ANY of the known datasets
    if not found_path:
        known_names = ["lfw", "webface", "web_face", "yaleb", "celeba", "megaface", "vggface2"]
        for root in search_roots:
            if not root.exists() or not root.is_dir(): continue
            for item in root.iterdir():
                if item.is_dir() and any(k in item.name.lower() for k in known_names):
                    found_path = item
                    break
            if found_path: break

    # Use the found path if available, else stick with data_path
    final_path = found_path if found_path else data_path
    
    # 2. Check for archives in the final path and extract them
    if final_path.is_dir():
        for arch_ext in ['*.zip', '*.tar', '*.tar.gz', '*.tgz']:
            for arch in final_path.glob(arch_ext):
                ext_dir = final_path / (arch.stem + "_extracted")
                if not ext_dir.exists():
                    os.makedirs(ext_dir, exist_ok=True)
                    try:
                        extract_archive(str(arch), str(ext_dir))
                    except: pass
    
    # 3. Handle if final_path itself is an archive
    if final_path.is_file() and final_path.suffix in ['.zip', '.tar', '.gz']:
        ext_dir = final_path.parent / (final_path.stem + "_extracted")
        if not ext_dir.exists():
            os.makedirs(ext_dir, exist_ok=True)
            try:
                extract_archive(str(final_path), str(ext_dir))
            except: pass
        final_path = ext_dir

    # 4. Find images recursively up to limit
    image_paths = get_images_from_dir(str(final_path), limit=limit)
    
    # 5. If still no images, look one level deeper in subdirectories
    if not image_paths and final_path.is_dir():
        for sub in final_path.iterdir():
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

