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
    elif archive_path.endswith('.tar') or archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    return extract_dir

def get_images_from_dir(directory, limit=100):
    """Recursively find images in directory up to limit"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm')
    image_paths = []
    if not os.path.exists(directory):
        return []
    for root, dirs, files in os.walk(directory):
        files.sort()
        dirs.sort()
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                if len(image_paths) >= limit:
                    return image_paths
    return image_paths

def prepare_dataset(data_path, limit=100, dataset_name=None):
    """Robust dataset discovery with fallback to internal gallery"""
    # 1. Try provided path if it looks valid and exists
    if data_path:
        p_path = Path(data_path).absolute()
        image_paths = get_images_from_dir(str(p_path), limit=limit)
        if image_paths: return image_paths

    # 2. Search for common dataset names in common locations
    search_roots = [
        Path("/project/input/data"),
        Path("/data6/user23215430/nudt/input/data")
    ]
    
    # If dataset_name provided (e.g. 'lfw'), search specifically
    if dataset_name:
        for root in search_roots:
            if not root.exists() or not root.is_dir(): continue
            for item in root.iterdir():
                if item.is_dir() and (dataset_name.lower() in item.name.lower()):
                    image_paths = get_images_from_dir(str(item), limit=limit)
                    if image_paths: return image_paths

    # 3. Aggressive rglob search in project input if no specific dir found
    project_input = Path("/project/input")
    if project_input.exists():
        # Look for archives first and extract them
        archives = list(project_input.glob("**/*.zip")) + list(project_input.glob("**/*.tar*"))
        for arch in archives:
            ext_dir = arch.parent / (arch.stem + "_extracted")
            if not ext_dir.exists():
                os.makedirs(ext_dir, exist_ok=True)
                try: extract_archive(str(arch), str(ext_dir))
                except: continue
            image_paths = get_images_from_dir(str(ext_dir), limit=limit)
            if image_paths: return image_paths

    # 4. FINAL FALLBACK: Use internal default_gallery
    internal_gallery = Path("/project/default_gallery")
    if internal_gallery.exists():
        image_paths = get_images_from_dir(str(internal_gallery), limit=limit)
        if image_paths: return image_paths
            
    return []

def calculate_metrics(orig_img, adv_img):
    mse = np.mean((orig_img - adv_img) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
    mu1, mu2 = orig_img.mean(), adv_img.mean()
    sigma1, sigma2 = orig_img.var(), adv_img.var()
    c1, c2 = 0.0001, 0.0009
    ssim = (2 * mu1 * mu2 + c1) * (2 * np.sqrt(sigma1 * sigma2) + c2) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    diff = orig_img - adv_img
    l2, linf = np.linalg.norm(diff), np.max(np.abs(diff))
    return {"psnr": float(psnr), "ssim": float(ssim), "l2_norm": float(l2), "linf_norm": float(linf), "combined_perturbation": float(l2 * 0.7 + linf * 0.3)}
