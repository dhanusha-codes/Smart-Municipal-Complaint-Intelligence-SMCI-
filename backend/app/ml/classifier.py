"""
Flask ML Classifier — routes inference to ml/inference/classifier.py
======================================================================
This file is the bridge between the Flask backend and the ML pipeline.
It preserves backward compatibility with all existing Flask routes
(complaints.py, ml_routes.py) while delegating to the unified classifier.
"""
import os
import sys
from pathlib import Path

# ─── Import the unified ML inference module ───────────────────────────────────
# Path: ml/inference/classifier.py (project-level, outside backend/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ML_INFERENCE = _REPO_ROOT / 'ml' / 'inference'

if str(_ML_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE))

try:
    from classifier import (
        classify_image as _classify_image,
        get_ml_status  as _get_ml_status,
        load_model     as _load_model,
        SMCI_CATEGORIES,
        DATASET_CATEGORIES,
        TORCH_AVAILABLE,
        SKLEARN_AVAILABLE,
        _state,
    )
    CATEGORIES = SMCI_CATEGORIES          # backward-compat alias
    _UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"[ML] Unified classifier not available: {e}")
    _UNIFIED_AVAILABLE = False

# ─── Backward-compat shims for existing Flask routes ─────────────────────────

if not _UNIFIED_AVAILABLE:
    # Minimal heuristic fallback (self-contained, no ML deps)
    import numpy as np
    from PIL import Image

    TORCH_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    CATEGORIES = [
        'Potholes', 'Garbage Accumulation', 'Waterlogging',
        'Damaged Roads', 'Drainage Problems', 'Street Lights', 'Other'
    ]
    MODEL_LOADED = False

    def classify_image(image_path, description=None):
        try:
            img = Image.open(image_path).convert('RGB').resize((128, 128))
            arr = np.array(img, dtype=np.float32) / 255.0
            gray       = arr.mean(axis=2)
            dark_ratio = float((gray < 0.3).mean())
            blue_ratio = float(((arr[:,:,2] > arr[:,:,0]) & (arr[:,:,2] > arr[:,:,1])).mean())
            brown_ratio = float(((arr[:,:,0]>0.4) & (arr[:,:,1]<0.35) & (arr[:,:,2]<0.25)).mean())

            desc_lower = (description or '').lower()
            kw = {
                'Potholes':             sum(1 for k in ['pothole','hole','crater'] if k in desc_lower),
                'Garbage Accumulation': sum(1 for k in ['garbage','waste','trash','dump'] if k in desc_lower),
                'Waterlogging':         sum(1 for k in ['water','flood','waterlog'] if k in desc_lower),
                'Damaged Roads':        sum(1 for k in ['damaged','broken','road','crack'] if k in desc_lower),
                'Drainage Problems':    sum(1 for k in ['drain','drainage','sewer','manhole'] if k in desc_lower),
                'Street Lights':        sum(1 for k in ['light','lamp','dark','streetlight'] if k in desc_lower),
                'Other': 0,
            }
            vis = {
                'Potholes': dark_ratio*3+brown_ratio*2,
                'Garbage Accumulation': (1-blue_ratio)*2+brown_ratio*1.5,
                'Waterlogging': blue_ratio*4,
                'Damaged Roads': dark_ratio*2+brown_ratio*1.5,
                'Drainage Problems': dark_ratio*2+blue_ratio*1.5,
                'Street Lights': (1-dark_ratio)*1.5,
                'Other': 1.0,
            }
            final = {c: kw.get(c,0)*2.0+vis.get(c,0) for c in CATEGORIES}
            best  = max(final, key=final.get)
            total = sum(final.values())
            conf  = round(min(0.65, max(0.3, final[best]/total if total>0 else 1/len(CATEGORIES))), 3)
        except Exception:
            best, conf = 'Other', 0.3
        return {
            'category': best, 'confidence': conf,
            'model_type': 'heuristic_fallback', 'status': 'heuristic',
            'message': 'Heuristic estimate — no trained model available.',
        }

    def get_ml_status():
        return {
            'torch_available':   False, 'sklearn_available': False,
            'model_loaded':      False, 'inference_mode': 'heuristic_fallback',
            'categories':        CATEGORIES,
            'training_note': 'Run ml/training/train_sklearn.py once dataset is ready.',
        }

    def load_model():
        return False

else:
    # Delegate to unified module
    MODEL_LOADED = bool(_state.get('pytorch_multiclass') or _state.get('sklearn_multiclass'))

    def classify_image(image_path, description=None):
        return _classify_image(image_path, description)

    def get_ml_status():
        return _get_ml_status()

    def load_model():
        return _load_model()
