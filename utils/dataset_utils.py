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
    """Deep search for images starting from data_path"""
    data_path = Path(data_path).absolute()
    
    # List of possible names to look for if dataset_name is provided
    # Standardizing names like 'web_face' to 'webface' etc.
    names_to_search = []
    if dataset_name:
        clean_name = dataset_name.lower().replace("_", "").replace("-", "")
        names_to_search.append(clean_name)
        names_to_search.append(dataset_name.lower())

    # 1. Search for a folder matching names_to_search recursively from search roots
    search_roots = [data_path, data_path.parent, Path("/project/input/data")]
    
    found_dir = None
    if names_to_search:
        for root in search_roots:
            if not root.exists() or not root.is_dir(): continue
            for sub in root.rglob('*'):
                if sub.is_dir():
                    sub_clean = sub.name.lower().replace("_", "").replace("-", "")
                    if any(n in sub_clean for n in names_to_search):
                        # Found a potential match, check if it has images
                        if get_images_from_dir(str(sub), limit=1):
                            found_dir = sub; break
            if found_dir: break

    # 2. If no specific dir found, just search globally from data_path for ANY images
    final_path = found_dir if found_dir else data_path
    image_paths = get_images_from_dir(str(final_path), limit=limit)
    
    # 3. If no images found, search for archives and extract them
    if not image_paths:
        archives = list(final_path.glob("**/*.zip")) + list(final_path.glob("**/*.tar*"))
        for arch in archives:
            ext_dir = arch.parent / (arch.stem + "_extracted")
            if not ext_dir.exists():
                os.makedirs(ext_dir, exist_ok=True)
                try: extract_archive(str(arch), str(ext_dir))
                except: continue
            image_paths.extend(get_images_from_dir(str(ext_dir), limit=limit - len(image_paths)))
            if len(image_paths) >= limit: break

    # 4. If STILL no images, search ANY directory that contains images
    if not image_paths and final_path.is_dir():
        for sub in final_path.rglob('*'):
            if sub.is_dir():
                found = get_images_from_dir(str(sub), limit=limit - len(image_paths))
                image_paths.extend(found)
                if len(image_paths) >= limit: break

    return image_paths

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
