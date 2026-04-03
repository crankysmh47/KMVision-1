import os
import cv2
import glob
import random
import albumentations as A
import multiprocessing as mp
from tqdm import tqdm

# Globals to heavily speed up multiprocessing
global_clinical_transform = None
global_anchor_transform = None

def init_worker():
    """Initializes the Albumentations pipelines ONCE per CPU worker to save massive overhead."""
    global global_clinical_transform
    global global_anchor_transform
    
    # STRICTLY pixel-level noise ONLY 
    global_clinical_transform = A.Compose([
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(max_holes=3, max_height=25, max_width=25, fill_value=0, p=0.3)
    ])
    
    # Pixel-level noise PLUS Safe Spatial Deformations
    global_anchor_transform = A.Compose([
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(max_holes=3, max_height=25, max_width=25, fill_value=0, p=0.3),
        A.SafeRotate(limit=4, border_mode=cv2.BORDER_CONSTANT, fill_value=255, p=0.5),
        A.Perspective(scale=(0.01, 0.05), fill=255, p=0.5),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, fill_value=255, p=0.5)
    ])

def process_image(img_path):
    try:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: return False
        
        if "_anchor" in filename:
            augmented = global_anchor_transform(image=image)['image']
        elif any(t in filename for t in ["_km", "_forest", "_waterfall"]):
            augmented = global_clinical_transform(image=image)['image']
        else:
            augmented = global_clinical_transform(image=image)['image']
            
        cv2.imwrite(img_path, augmented)
        return True
    except Exception as e:
        # Silently fail on locked files or corrupt writes
        return False

def augment_images(image_dir, ratio=0.2):
    image_paths = glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
    n_augment = int(len(image_paths) * ratio)
    if n_augment == 0: 
        print("No images found to augment.")
        return
    
    augment_paths = random.sample(image_paths, n_augment)
    print(f"Bifurcated Augmentation Pipeline -> Modifying {n_augment} images ({ratio*100}% sample)")
    
    # We restrict processes to 10 to prevent Windows from locking up under intense I/O
    with mp.Pool(processes=min(10, os.cpu_count()), initializer=init_worker) as pool:
        # Chunksize=50 reduces IPC overhead massively
        list(tqdm(pool.imap_unordered(process_image, augment_paths, chunksize=50), total=len(augment_paths)))

if __name__ == "__main__":
    # Allow passing ratio as CLI parameter for easy canary testing
    import sys
    ratio = 0.2
    if len(sys.argv) > 1:
        ratio = float(sys.argv[1])
    augment_images(r"C:\sem4\KMVision-1 Data\dataset\images", ratio=ratio)
