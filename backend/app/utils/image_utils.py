"""
Utility functions - Haversine distance, image processing, etc.
"""
import math
import os
import hashlib
import struct
from PIL import Image
import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in kilometers."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def compute_image_hash(image_path):
    """Compute perceptual hash of image using average hash algorithm."""
    try:
        img = Image.open(image_path).convert('L').resize((8, 8), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p >= avg else '0' for p in pixels)
        # Convert to hex
        hex_hash = ''
        for i in range(0, len(bits), 4):
            hex_hash += hex(int(bits[i:i+4], 2))[2:]
        return hex_hash
    except Exception:
        return None


def compute_image_features(image_path):
    """
    Extract image feature vector for similarity comparison.
    Uses color histogram + spatial features as a lightweight alternative to deep features.
    Returns a list of floats.
    """
    try:
        img = Image.open(image_path).convert('RGB').resize((64, 64), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

        features = []

        # Color histograms per channel (16 bins each)
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1))
            hist = hist / (hist.sum() + 1e-7)
            features.extend(hist.tolist())

        # Spatial average in grid (4x4 grid, 3 channels = 48 features)
        for row in range(4):
            for col in range(4):
                block = arr[row*16:(row+1)*16, col*16:(col+1)*16]
                features.extend(block.mean(axis=(0, 1)).tolist())

        # Brightness and contrast
        gray = arr.mean(axis=2)
        features.append(float(gray.mean()))
        features.append(float(gray.std()))

        return features
    except Exception:
        return None


def cosine_similarity(a, b):
    """Compute cosine similarity between two feature vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two hex hashes."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64  # max distance
    bits1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    bits2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))


def hash_similarity(hash1, hash2):
    """Convert hash Hamming distance to similarity score 0-1."""
    dist = hamming_distance(hash1, hash2)
    # 64 bits total; normalize
    return max(0.0, 1.0 - dist / 64.0)


def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_image(file, upload_folder):
    """Save uploaded image and return filename."""
    import uuid
    import os
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filename, filepath

