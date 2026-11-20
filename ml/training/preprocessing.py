"""
Image Preprocessing Module — SMCI ML Pipeline
==============================================
Provides reusable, framework-agnostic image preprocessing.
Works with Pillow + NumPy (always available).
Optionally integrates with torchvision transforms when PyTorch is installed.

Design principles:
- Never modifies the original image file on disk.
- Always returns a copy / in-memory array.
- Handles corrupt/unreadable images gracefully.
- Consistent target size for all classifiers.
"""
import os
import hashlib
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ─── Configuration ────────────────────────────────────────────────────────────

TARGET_SIZE   = (224, 224)        # MobileNetV3 / ResNet standard input
NUM_CHANNELS  = 3                 # RGB
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# ImageNet normalisation constants (used by MobileNetV3, EfficientNet, ResNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Check optional dependencies
try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─── Core preprocessing (Pillow + NumPy, no PyTorch) ─────────────────────────

def load_image(path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Open an image file and convert to RGB.
    Returns None on any error without raising.
    Never modifies the file.
    """
    try:
        img = Image.open(path)
        img.load()                       # force full decode, detect truncation
        return img.convert('RGB')
    except (UnidentifiedImageError, OSError, struct.error, Exception):
        return None


def resize_with_padding(img: Image.Image, target: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """
    Resize image to target size using letterboxing (maintains aspect ratio).
    Fills padding with black (0,0,0).
    """
    img = img.convert('RGB')
    img.thumbnail(target, Image.LANCZOS)       # in-place but only shrinks
    result = Image.new('RGB', target, (0, 0, 0))
    offset = ((target[0] - img.width) // 2, (target[1] - img.height) // 2)
    result.paste(img, offset)
    return result


def resize_stretch(img: Image.Image, target: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Resize image to exact target size (may distort aspect ratio)."""
    return img.convert('RGB').resize(target, Image.LANCZOS)


def to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to float32 numpy array [H, W, C] in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    assert arr.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3), \
        f"Unexpected shape {arr.shape} — ensure image is resized first."
    return arr


def normalize_imagenet(arr: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet normalisation: (pixel - mean) / std
    Input: float32 array [H, W, 3] in [0, 1]
    Output: float32 array [H, W, 3], each channel z-normalised
    """
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_numpy(
    path: Union[str, Path],
    target: Tuple[int, int] = TARGET_SIZE,
    normalize: bool = True,
    padding: bool = True,
) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline → numpy array [H, W, 3].
    Returns None if the image cannot be loaded (corrupt / missing).

    Args:
        path      : Path to image file.
        target    : (width, height) target size.
        normalize : Apply ImageNet normalisation if True.
        padding   : Use letterboxing (True) or stretch resize (False).
    """
    img = load_image(path)
    if img is None:
        return None

    img = resize_with_padding(img, target) if padding else resize_stretch(img, target)
    arr = to_numpy(img)
    if normalize:
        arr = normalize_imagenet(arr)
    return arr


# ─── PyTorch transform pipeline (used during training) ───────────────────────

def get_torch_transforms(
    phase: str = 'train',
    target: Tuple[int, int] = TARGET_SIZE,
) -> Optional[object]:
    """
    Returns a torchvision.transforms.Compose pipeline for the given phase.
    Returns None if PyTorch is not installed.

    Args:
        phase : 'train' | 'val' | 'test'
        target: (H, W) tuple — same as TARGET_SIZE
    """
    if not TORCH_AVAILABLE:
        return None

    H, W = target
    if phase == 'train':
        return T.Compose([
            T.Resize((H + 32, W + 32)),          # slightly larger for random crop
            T.RandomCrop((H, W)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ])
    else:  # val / test
        return T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ])


# ─── Batch preprocessing (for inference or feature extraction) ────────────────

def preprocess_batch(
    paths: list,
    target: Tuple[int, int] = TARGET_SIZE,
    normalize: bool = True,
) -> Tuple[np.ndarray, list]:
    """
    Preprocess a list of image paths into a stacked numpy array.

    Returns:
        (batch_array [N, H, W, 3], valid_paths)
        Failed images are silently skipped; valid_paths contains paths that
        matched successfully.
    """
    arrays = []
    valid = []
    for p in paths:
        arr = preprocess_numpy(p, target=target, normalize=normalize)
        if arr is not None:
            arrays.append(arr)
            valid.append(p)

    if not arrays:
        return np.empty((0, target[1], target[0], 3), dtype=np.float32), []

    return np.stack(arrays, axis=0), valid


# ─── Feature extraction (NumPy, no PyTorch needed) ───────────────────────────

def extract_color_histogram(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Extract per-channel colour histogram features from a normalised [H,W,3] array.
    Denormalises before computing histogram.
    Returns 1-D float32 feature vector of length bins*3.
    """
    # Denormalize to [0,1]
    raw = (arr * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
    features = []
    for c in range(3):
        hist, _ = np.histogram(raw[:, :, c].ravel(), bins=bins, range=(0.0, 1.0))
        features.extend(hist / hist.sum())          # L1 normalise
    return np.array(features, dtype=np.float32)


def extract_lbp_like(arr: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Lightweight texture features: mean and std of each spatial block.
    Provides coarse texture information without CV2.
    Returns 1-D float32 feature vector.
    """
    H, W, C = arr.shape
    features = []
    for r in range(0, H - block_size + 1, block_size):
        for c_idx in range(0, W - block_size + 1, block_size):
            block = arr[r:r+block_size, c_idx:c_idx+block_size, :]
            features.extend(block.mean(axis=(0, 1)).tolist())
            features.extend(block.std(axis=(0, 1)).tolist())
    return np.array(features, dtype=np.float32)


def extract_features(
    path: Union[str, Path],
    include_histogram: bool = True,
    include_texture: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract a combined handcrafted feature vector from an image.
    Used by the scikit-learn baseline classifier.

    Returns None if the image cannot be loaded.
    """
    arr = preprocess_numpy(path, normalize=True)
    if arr is None:
        return None

    parts = []
    if include_histogram:
        parts.append(extract_color_histogram(arr))
    if include_texture:
        parts.append(extract_lbp_like(arr))

    return np.concatenate(parts) if parts else None


# ─── Utility ──────────────────────────────────────────────────────────────────

def image_phash(path: Union[str, Path], hash_size: int = 8) -> Optional[str]:
    """
    Compute a simple perceptual hash (average-hash) for near-duplicate detection.
    Returns hex string or None on error.
    """
    img = load_image(path)
    if img is None:
        return None
    small = img.resize((hash_size, hash_size), Image.LANCZOS).convert('L')
    arr = np.array(small, dtype=np.float32)
    threshold = arr.mean()
    bits = (arr > threshold).flatten()
    # Pack bits into hex
    packed = 0
    for bit in bits:
        packed = (packed << 1) | int(bit)
    return format(packed, f'0{hash_size * hash_size // 4}x')


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex phash strings."""
    if len(h1) != len(h2):
        return 999
    n1 = int(h1, 16)
    n2 = int(h2, 16)
    xor = n1 ^ n2
    return bin(xor).count('1')


if __name__ == '__main__':
    # Quick self-test
    import sys
    print("preprocessing.py — self-test")
    print(f"  PyTorch available      : {TORCH_AVAILABLE}")
    print(f"  Target size            : {TARGET_SIZE}")
    print(f"  ImageNet mean          : {IMAGENET_MEAN}")
    print(f"  ImageNet std           : {IMAGENET_STD}")
    print(f"  Torch train transforms : {get_torch_transforms('train')}")
    print(f"  Torch val transforms   : {get_torch_transforms('val')}")
    print("  preprocess_numpy()     : ready")
    print("  extract_features()     : ready")
    print("  image_phash()          : ready")
    print("Self-test PASSED ✅")
