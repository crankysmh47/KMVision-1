"""5-crop image preprocessing (matches train_phase_b.ClinicalChartDataset)."""

from PIL import Image
import torch


def load_and_crop_five(image_path: str) -> list:
    image = Image.open(image_path)
    image.verify()
    image = Image.open(image_path).convert("RGB")

    global_img = image.resize((384, 384))
    img_768 = image.resize((768, 768))
    tl = img_768.crop((0, 0, 384, 384))
    tr = img_768.crop((384, 0, 768, 384))
    bl = img_768.crop((0, 384, 384, 768))
    br = img_768.crop((384, 384, 768, 768))
    return [global_img, tl, tr, bl, br]


def pixel_values_from_path(image_path: str, processor) -> torch.Tensor:
    """Returns tensor shape (5, C, H, W)."""
    crops = load_and_crop_five(image_path)
    batch = processor(images=crops, return_tensors="pt").pixel_values
    return batch.squeeze(0) if batch.dim() == 5 else batch
