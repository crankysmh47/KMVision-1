import os
import cv2
import glob
import random
import albumentations as A

def get_augmenter():
    return A.Compose([
        A.ImageCompression(quality_range=(30, 70), p=0.5),
        A.GaussNoise(std_range=(0.2, 0.44), p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, fill_value=(255, 255, 255), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(10, 40), hole_width_range=(10, 100), fill_value=0, p=0.3)
    ])

def augment_images(image_dir, ratio=0.1):
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    n_augment = int(len(image_paths) * ratio)
    if n_augment == 0: return
    
    augment_paths = random.sample(image_paths, n_augment)
    transform = get_augmenter()
    
    for img_path in augment_paths:
        image = cv2.imread(img_path)
        if image is None: continue
        # OpenCV reads in BGR, albumentations applies transforms
        augmented = transform(image=image)['image']
        cv2.imwrite(img_path, augmented)

if __name__ == "__main__":
    augment_images("dataset/images", ratio=0.1)
