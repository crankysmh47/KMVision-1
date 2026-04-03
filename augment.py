import os
import cv2
import glob
import random
import albumentations as A
import multiprocessing as mp
from tqdm import tqdm

def get_clinical_augmenter():
    # STRICTLY pixel-level noise ONLY 
    # Must preserve spatial step-function and forest properties
    return A.Compose([
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.5), # In newer versions quality_lower/upper instead of quality_range
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # var_limit is robust
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(max_holes=5, max_height=40, max_width=100, min_holes=1, min_height=10, min_width=10, fill_value=0, p=0.3) 
    ])

def get_anchor_augmenter():
    # Pixel-level noise PLUS Safe Spatial Deformations
    return A.Compose([
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(max_holes=5, max_height=40, max_width=100, min_holes=1, min_height=10, min_width=10, fill_value=0, p=0.3),
        A.SafeRotate(limit=4, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.5),
        A.Perspective(scale=(0.01, 0.05), pad_val=(255, 255, 255), p=0.5),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), p=0.5)
    ])

# Handle albumentations version differences safely
def safe_get_clinical_augmenter():
    try:
        return get_clinical_augmenter()
    except Exception:
        # Fallback to older albumentations syntax
        return A.Compose([
            A.ImageCompression(quality_range=(30, 70), p=0.5),
            A.GaussNoise(std_range=(0.2, 0.44), p=0.5),
            A.Blur(blur_limit=5, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(10, 40), hole_width_range=(10, 100), fill_value=0, p=0.3)
        ])

def safe_get_anchor_augmenter():
    try:
        return get_anchor_augmenter()
    except Exception:
        return A.Compose([
            A.ImageCompression(quality_range=(30, 70), p=0.5),
            A.GaussNoise(std_range=(0.2, 0.44), p=0.5),
            A.Blur(blur_limit=5, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(10, 40), hole_width_range=(10, 100), fill_value=0, p=0.3),
            A.SafeRotate(limit=4, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.5),
            A.Perspective(scale=(0.01, 0.05), p=0.5),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), p=0.5)
        ])


def process_image(img_path):
    try:
        # Determine pipeline based on filename
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: return False
        
        if "_anchor" in filename:
            transform = safe_get_anchor_augmenter()
        elif any(t in filename for t in ["_km", "_forest", "_waterfall"]):
            transform = safe_get_clinical_augmenter()
        else:
            transform = safe_get_clinical_augmenter()
            
        augmented = transform(image=image)['image']
        cv2.imwrite(img_path, augmented)
        return True
    except Exception as e:
        print(f"Error augmenting {filename}: {e}")
        return False

def augment_images(image_dir, ratio=0.2):
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    n_augment = int(len(image_paths) * ratio)
    if n_augment == 0: 
        print("No images found to augment.")
        return
    
    augment_paths = random.sample(image_paths, n_augment)
    print(f"Bifurcated Augmentation Pipeline -> Modifying {n_augment} images (20% sample)")
    
    with mp.Pool(processes=min(os.cpu_count(), 16)) as pool:
        list(tqdm(pool.imap_unordered(process_image, augment_paths), total=len(augment_paths)))

if __name__ == "__main__":
    augment_images(r"C:\sem4\KMVision-1 Data\dataset\images", ratio=0.2)
