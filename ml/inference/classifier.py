"""
Unified Classifier Interface — SMCI ML Pipeline
================================================
Single entry point for ML inference used by Flask.

Priority chain:
  1. PyTorch MobileNetV3 6-class   (multiclass_mobilenetv3.pt)
  2. Sklearn SVM 6-class baseline   (multiclass_sklearn.pkl)
  3. Heuristic fallback             (colour + keyword rules, always available)

The Flask app imports only this module; it never imports training scripts.

DATASET LABEL → SMCI CANONICAL CATEGORY mapping:
  potholes           → Potholes
  cracks             → Damaged Roads
  broken_edges       → Damaged Roads
  waterlogged_roads  → Waterlogging
  open_manholes      → Drainage Problems
  construction_zones → Other
"""
import os
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

# ─── Path setup ───────────────────────────────────────────────────────────────
# ml/inference/classifier.py lives at:  <repo>/ml/inference/classifier.py
# Models are saved at:                  <repo>/ml/models/
ML_DIR     = Path(__file__).resolve().parent.parent  # <repo>/ml/
MODELS_DIR = ML_DIR / 'models'                       # <repo>/ml/models/

# ─── Category mappings ────────────────────────────────────────────────────────

# 6-class dataset labels (must match training CATEGORIES list in order)
DATASET_CATEGORIES = [
    'broken_edges',
    'construction_zones',
    'cracks',
    'open_manholes',
    'potholes',
    'waterlogged_roads',
]

# Map dataset label → SMCI canonical category displayed in the app
DATASET_TO_SMCI = {
    'potholes':           'Potholes',
    'cracks':             'Damaged Roads',
    'broken_edges':       'Damaged Roads',
    'waterlogged_roads':  'Waterlogging',
    'open_manholes':      'Drainage Problems',
    'construction_zones': 'Other',
}

# All canonical SMCI categories (for heuristic fallback)
SMCI_CATEGORIES = [
    'Potholes',
    'Garbage Accumulation',
    'Waterlogging',
    'Damaged Roads',
    'Drainage Problems',
    'Street Lights',
    'Other',
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE   = (224, 224)

# ─── Backend availability ─────────────────────────────────────────────────────

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.pipeline import Pipeline as _SklearnPipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─── State (populated on first call) ─────────────────────────────────────────

_state: Dict[str, Any] = {
    'pytorch_multiclass': None,   # dict with model + metadata
    'sklearn_multiclass': None,   # dict with pipeline + label_encoder
    'loaded': False,
}


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_pytorch_multiclass() -> bool:
    """Load multiclass_mobilenetv3.pt if available."""
    if not TORCH_AVAILABLE:
        return False
    model_file = MODELS_DIR / 'multiclass_mobilenetv3.pt'
    if not model_file.exists():
        return False
    try:
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small

        ckpt        = torch.load(model_file, map_location='cpu', weights_only=False)
        num_classes = ckpt.get('num_classes', 6)
        categories  = ckpt.get('categories', DATASET_CATEGORIES)

        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        _state['pytorch_multiclass'] = {
            'model':      model,
            'categories': categories,
            'metrics':    ckpt.get('test_metrics', {}),
            'num_classes': num_classes,
        }
        print('[Classifier] pytorch_multiclass: loaded')
        return True
    except Exception as e:
        print(f'[Classifier] PyTorch multiclass load failed: {e}')
        return False


def _load_sklearn_multiclass() -> bool:
    """Load multiclass_sklearn.pkl if available."""
    if not SKLEARN_AVAILABLE:
        return False
    model_file = MODELS_DIR / 'multiclass_sklearn.pkl'
    if not model_file.exists():
        return False
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        _state['sklearn_multiclass'] = data
        print('[Classifier] sklearn_multiclass: loaded')
        return True
    except Exception as e:
        print(f'[Classifier] sklearn multiclass load failed: {e}')
        return False


def initialize_models():
    """
    Attempt to load all available trained models.
    Called once at startup; safe to call multiple times.
    """
    if _state['loaded']:
        return

    pt_ok = _load_pytorch_multiclass()
    sk_ok = _load_sklearn_multiclass()

    if not pt_ok:
        print('[Classifier] pytorch_multiclass: not available')
    if not sk_ok:
        print('[Classifier] sklearn_multiclass: not available')

    _state['loaded'] = True


# ─── Image preprocessing ──────────────────────────────────────────────────────

def _preprocess_for_sklearn(image_path: str) -> Optional[np.ndarray]:
    """Extract the same 1,224-dim handcrafted feature vector used during training."""
    try:
        img = Image.open(image_path).convert('RGB').resize(TARGET_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

        # Colour histogram (16 bins × 3 channels = 48 features)
        raw = (arr * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
        hist_feats: List[float] = []
        for c in range(3):
            hist, _ = np.histogram(raw[:, :, c].ravel(), bins=16, range=(0.0, 1.0))
            hist_feats.extend((hist / hist.sum()).tolist())

        # Block mean/std texture (16×16 blocks, 2 stats × 3 channels = 1176 features)
        H, W, _ = arr.shape
        bs = 16
        tex_feats: List[float] = []
        for r in range(0, H - bs + 1, bs):
            for c_idx in range(0, W - bs + 1, bs):
                block = arr[r:r+bs, c_idx:c_idx+bs, :]
                tex_feats.extend(block.mean(axis=(0, 1)).tolist())
                tex_feats.extend(block.std(axis=(0, 1)).tolist())

        return np.array(hist_feats + tex_feats, dtype=np.float32).reshape(1, -1)
    except Exception:
        return None


def _preprocess_for_pytorch(image_path: str):
    """Preprocess image to [1, 3, 224, 224] tensor for MobileNetV3."""
    if not TORCH_AVAILABLE:
        return None
    try:
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize(TARGET_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ])
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0)   # [1, 3, H, W]
    except Exception:
        return None


# ─── Multiclass inference ─────────────────────────────────────────────────────

def _infer_pytorch_multiclass(image_path: str) -> Optional[Dict]:
    """6-class MobileNetV3 inference. Returns dataset label + confidence."""
    entry = _state.get('pytorch_multiclass')
    if not entry:
        return None
    try:
        import torch.nn.functional as F

        tensor = _preprocess_for_pytorch(image_path)
        if tensor is None:
            return None

        model      = entry['model']
        categories = entry['categories']

        with torch.no_grad():
            outputs    = model(tensor)
            probs      = F.softmax(outputs, dim=1)[0]
            pred_idx   = int(probs.argmax())
            confidence = float(probs[pred_idx])

        dataset_label = categories[pred_idx]
        return {
            'dataset_label': dataset_label,
            'confidence':    round(confidence, 4),
            'backend':       'pytorch_mobilenetv3',
            'all_probs':     {categories[i]: round(float(probs[i]), 4)
                              for i in range(len(categories))},
        }
    except Exception as e:
        print(f'[Classifier] PyTorch multiclass inference error: {e}')
        return None


def _infer_sklearn_multiclass(image_path: str) -> Optional[Dict]:
    """6-class sklearn SVM inference. Returns dataset label + confidence."""
    entry = _state.get('sklearn_multiclass')
    if not entry:
        return None
    try:
        feats = _preprocess_for_sklearn(image_path)
        if feats is None:
            return None

        pipeline = entry['pipeline']
        le       = entry['label_encoder']

        pred_encoded = int(pipeline.predict(feats)[0])
        proba        = pipeline.predict_proba(feats)[0]
        confidence   = float(proba[pred_encoded])
        dataset_label = str(le.inverse_transform([pred_encoded])[0])

        return {
            'dataset_label': dataset_label,
            'confidence':    round(confidence, 4),
            'backend':       'sklearn_' + entry.get('model_name', 'svm'),
            'all_probs':     {str(le.inverse_transform([i])[0]): round(float(proba[i]), 4)
                              for i in range(len(le.classes_))},
        }
    except Exception as e:
        print(f'[Classifier] sklearn multiclass inference error: {e}')
        return None


# ─── Heuristic fallback (always available, unchanged) ─────────────────────────

def _heuristic_classify(image_path: str, description: Optional[str] = None) -> Dict:
    """
    Rule-based classification using image colour statistics + description keywords.
    Confidence capped at 0.65. Labelled 'heuristic_fallback'.
    """
    try:
        img = Image.open(image_path).convert('RGB').resize((128, 128))
        arr = np.array(img, dtype=np.float32) / 255.0

        gray        = arr.mean(axis=2)
        dark_ratio  = float((gray < 0.3).mean())
        brown_ratio = float(((arr[:,:,0]>0.4) & (arr[:,:,1]<0.35) & (arr[:,:,2]<0.25)).mean())
        blue_ratio  = float(((arr[:,:,2]>arr[:,:,0]) & (arr[:,:,2]>arr[:,:,1])).mean())

        desc_lower = (description or '').lower()
        kw = {
            'Potholes':             sum(1 for k in ['pothole','hole','crater','road damage','broken road'] if k in desc_lower),
            'Garbage Accumulation': sum(1 for k in ['garbage','waste','trash','dump','litter','rubbish']   if k in desc_lower),
            'Waterlogging':         sum(1 for k in ['water','flood','waterlog','rain','stagnant']          if k in desc_lower),
            'Damaged Roads':        sum(1 for k in ['damaged','broken','road','surface','crack','collapse'] if k in desc_lower),
            'Drainage Problems':    sum(1 for k in ['drain','drainage','blocked','sewer','manhole','storm'] if k in desc_lower),
            'Street Lights':        sum(1 for k in ['light','lamp','dark','streetlight','pole','electric']  if k in desc_lower),
            'Other': 0,
        }
        vis = {
            'Potholes':             dark_ratio * 3.0 + brown_ratio * 2.0,
            'Garbage Accumulation': (1 - blue_ratio) * 2.0 + brown_ratio * 1.5,
            'Waterlogging':         blue_ratio * 4.0,
            'Damaged Roads':        dark_ratio * 2.0 + brown_ratio * 1.5,
            'Drainage Problems':    dark_ratio * 2.0 + blue_ratio * 1.5,
            'Street Lights':        (1 - dark_ratio) * 1.5,
            'Other':                1.0,
        }
        final = {cat: kw.get(cat, 0) * 2.0 + vis.get(cat, 0) for cat in SMCI_CATEGORIES}
        best  = max(final, key=final.get)
        total = sum(final.values())
        conf  = final[best] / total if total > 0 else 1.0 / len(SMCI_CATEGORIES)
        conf  = round(min(0.65, max(0.3, conf)), 4)

        return {
            'category':   best,
            'confidence': conf,
            'model_type': 'heuristic_fallback',
            'status':     'heuristic',
            'message': (
                'No trained model available. Result is a heuristic estimate '
                'based on image colour statistics and description keywords. '
                'Manual review is recommended.'
            ),
        }
    except Exception as e:
        print(f'[Classifier] Heuristic error: {e}')
        return {
            'category': 'Other', 'confidence': 0.3,
            'model_type': 'heuristic_fallback', 'status': 'heuristic',
            'message': 'Classification failed; manual review required.',
        }


# ─── Unified classify_image ───────────────────────────────────────────────────

def classify_image(image_path: str, description: Optional[str] = None) -> Dict:
    """
    Main entry point for the Flask complaint API.

    Priority:
      1. PyTorch MobileNetV3 6-class
      2. Sklearn SVM 6-class baseline
      3. Heuristic fallback

    Returns dict:
      category   : SMCI canonical category string
      confidence : float [0,1]
      model_type : 'pytorch_mobilenetv3' | 'sklearn_<name>' | 'heuristic_fallback'
      status     : 'completed' | 'heuristic'
      message    : human-readable explanation
      dataset_label : raw dataset class name (for debugging/admin display)
    """
    if not os.path.exists(image_path):
        return {
            'category': 'Other', 'confidence': 0.0,
            'model_type': 'none', 'status': 'error',
            'message': f'Image file not found: {image_path}',
            'dataset_label': None,
        }

    initialize_models()

    # ── 1. Try PyTorch 6-class ────────────────────────────────────
    result = _infer_pytorch_multiclass(image_path)
    if result:
        dataset_label  = result['dataset_label']
        smci_category  = DATASET_TO_SMCI.get(dataset_label, 'Other')
        return {
            'category':     smci_category,
            'confidence':   result['confidence'],
            'model_type':   result['backend'],
            'status':       'completed',
            'message':      f'Classified by PyTorch MobileNetV3 ({dataset_label}).',
            'dataset_label': dataset_label,
            'all_probs':    result.get('all_probs', {}),
        }

    # ── 2. Try sklearn 6-class ────────────────────────────────────
    result = _infer_sklearn_multiclass(image_path)
    if result:
        dataset_label  = result['dataset_label']
        smci_category  = DATASET_TO_SMCI.get(dataset_label, 'Other')
        return {
            'category':     smci_category,
            'confidence':   result['confidence'],
            'model_type':   result['backend'],
            'status':       'completed',
            'message':      f'Classified by sklearn SVM ({dataset_label}).',
            'dataset_label': dataset_label,
            'all_probs':    result.get('all_probs', {}),
        }

    # ── 3. Heuristic fallback ─────────────────────────────────────
    h = _heuristic_classify(image_path, description)
    h['dataset_label'] = None
    return h


# ─── Status / diagnostics ────────────────────────────────────────────────────

def get_ml_status() -> Dict:
    """Return structured ML system status for /api/ml/status endpoint."""
    initialize_models()

    pt_entry = _state.get('pytorch_multiclass')
    sk_entry = _state.get('sklearn_multiclass')

    pt_loaded = pt_entry is not None
    sk_loaded = sk_entry is not None

    inference_mode = (
        'pytorch_mobilenetv3' if pt_loaded else
        'sklearn_multiclass'  if sk_loaded else
        'heuristic_fallback'
    )

    return {
        'torch_available':   TORCH_AVAILABLE,
        'sklearn_available': SKLEARN_AVAILABLE,
        'pytorch_multiclass_loaded': pt_loaded,
        'sklearn_multiclass_loaded': sk_loaded,
        'model_loaded':      pt_loaded or sk_loaded,
        'inference_mode':    inference_mode,
        'pytorch_metrics':   pt_entry.get('metrics', {}) if pt_entry else {},
        'sklearn_metrics':   {
            'val_metrics': sk_entry.get('val_metrics', {}),
            'test_metrics': sk_entry.get('test_metrics', {}),
        } if sk_entry else {},
        'dataset_categories': DATASET_CATEGORIES,
        'smci_categories':    SMCI_CATEGORIES,
        'label_mapping':      DATASET_TO_SMCI,
        'training_note': (
            'Train PyTorch : python ml/training/train_pytorch.py\n'
            'Train sklearn : python ml/training/train_multiclass_sklearn.py'
        ),
    }


def load_model():
    """Compatibility shim for existing Flask code that calls load_model()."""
    initialize_models()
    return bool(_state.get('pytorch_multiclass') or _state.get('sklearn_multiclass'))


# Initialise on import (non-blocking)
initialize_models()
