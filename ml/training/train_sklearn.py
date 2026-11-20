"""
Scikit-Learn Baseline Classifier — SMCI ML Pipeline
====================================================
Trains binary classifiers for Pothole and Garbage detection using
handcrafted image features + RandomForest / SVM.

This baseline works with the currently installed packages
(scikit-learn, Pillow, NumPy) and does NOT require PyTorch.

Models trained here are saved to ml/models/ and are loaded by the
Flask classifier as a fallback when PyTorch is unavailable.

Usage:
    python ml/training/train_sklearn.py --classifier pothole
    python ml/training/train_sklearn.py --classifier garbage
    python ml/training/train_sklearn.py --classifier all
"""
import os
import sys
import json
import argparse
import pickle
import random
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score,
)

warnings.filterwarnings('ignore')

# ─── Path setup ───────────────────────────────────────────────────────────────
ML_DIR      = Path(__file__).resolve().parent.parent
REPO_ROOT   = ML_DIR.parent
DATASET_DIR = REPO_ROOT / 'dataset'
MODELS_DIR  = ML_DIR / 'models'
EVAL_DIR    = ML_DIR / 'evaluation'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Add ml/training to path for preprocessing import
sys.path.insert(0, str(ML_DIR / 'training'))
from preprocessing import (
    extract_features, image_phash, hamming_distance, SUPPORTED_EXTS
)
# SUPPORTED_EXTS may not be exported; define fallback
try:
    from preprocessing import SUPPORTED_EXTS
except ImportError:
    SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASSIFIERS_CONFIG = {
    'pothole': {
        'positive_dir': DATASET_DIR / 'pothole' / 'yes',
        'negative_dir': DATASET_DIR / 'pothole' / 'no',
        'labels': {1: 'pothole', 0: 'no_pothole'},
        'model_file': 'pothole_sklearn.pkl',
    },
    'garbage': {
        'positive_dir': DATASET_DIR / 'garbage' / 'yes',
        'negative_dir': DATASET_DIR / 'garbage' / 'no',
        'labels': {1: 'garbage', 0: 'no_garbage'},
        'model_file': 'garbage_sklearn.pkl',
    },
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_image_paths(folder: Path, label: int) -> List[Tuple[Path, int]]:
    """Return list of (path, label) for all supported images in folder."""
    items = []
    if not folder.exists():
        return items
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTS and f.is_file():
            items.append((f, label))
    return items


def remove_near_duplicates(
    items: List[Tuple[Path, int]],
    threshold: int = 5,
) -> List[Tuple[Path, int]]:
    """
    Remove near-duplicate images using perceptual hashing.
    Two images with Hamming distance <= threshold are considered duplicates.
    Keeps the first occurrence.
    """
    hashes = []
    kept = []
    removed = 0
    for path, label in items:
        h = image_phash(path)
        if h is None:
            continue  # skip unreadable
        is_dup = any(hamming_distance(h, existing) <= threshold for existing in hashes)
        if not is_dup:
            hashes.append(h)
            kept.append((path, label))
        else:
            removed += 1
    if removed:
        print(f"  [dedup] Removed {removed} near-duplicate images.")
    return kept


def extract_dataset_features(
    items: List[Tuple[Path, int]],
    desc: str = "Extracting",
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    """
    Extract feature vectors for all images.
    Returns (X, y, valid_paths).
    """
    X, y, paths = [], [], []
    total = len(items)
    for i, (path, label) in enumerate(items):
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"\r  {desc}: {i+1}/{total}", end='', flush=True)
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(label)
            paths.append(path)
    print()  # newline after progress
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


# ─── Model selection ──────────────────────────────────────────────────────────

def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """Return candidate sklearn Pipelines to evaluate."""
    return {
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=4,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        'gradient_boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=RANDOM_SEED,
            )),
        ]),
        'svm_rbf': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=RANDOM_SEED,
            )),
        ]),
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=500,
                random_state=RANDOM_SEED,
            )),
        ]),
    }


# ─── Training pipeline ────────────────────────────────────────────────────────

def train_binary_classifier(classifier_name: str) -> Optional[Dict]:
    """
    Full training pipeline for one binary classifier.
    Returns evaluation metrics dict or None if dataset is unavailable.
    """
    cfg = CLASSIFIERS_CONFIG[classifier_name]
    pos_dir = cfg['positive_dir']
    neg_dir = cfg['negative_dir']
    labels  = cfg['labels']

    print(f"\n{'='*60}")
    print(f"  TRAINING: {classifier_name.upper()} CLASSIFIER")
    print(f"{'='*60}")

    # ── 1. Load paths ────────────────────────────────────────────
    pos_items = load_image_paths(pos_dir, 1)
    neg_items = load_image_paths(neg_dir, 0)

    if not pos_items and not neg_items:
        print(f"  ❌ No images found in {pos_dir} or {neg_dir}")
        print(f"     Place images and re-run once the dataset is downloaded.")
        return None

    if len(pos_items) < 10:
        print(f"  ❌ Too few positive images ({len(pos_items)}). Need ≥10 to train.")
        return None
    if len(neg_items) < 10:
        print(f"  ❌ Too few negative images ({len(neg_items)}). Need ≥10 to train.")
        return None

    print(f"  Positive ({labels[1]}): {len(pos_items)} images")
    print(f"  Negative ({labels[0]}): {len(neg_items)} images")

    # ── 2. Near-duplicate removal ─────────────────────────────────
    print("  Removing near-duplicates...")
    pos_items = remove_near_duplicates(pos_items)
    neg_items = remove_near_duplicates(neg_items)
    all_items = pos_items + neg_items
    random.shuffle(all_items)

    # ── 3. Feature extraction ─────────────────────────────────────
    print(f"  Extracting features from {len(all_items)} images...")
    X, y, valid_paths = extract_dataset_features(all_items)

    if len(X) == 0:
        print("  ❌ Feature extraction failed for all images.")
        return None

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"  Valid features: {len(X)} total | {n_pos} positive | {n_neg} negative")
    print(f"  Feature vector length: {X.shape[1]}")

    # ── 4. Train / val / test split ───────────────────────────────
    # Stratified 70 / 15 / 15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15 / 0.85,
        random_state=RANDOM_SEED, stratify=y_trainval
    )

    print(f"  Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 5. Model selection via cross-validation ───────────────────
    print("  Cross-validating candidate models (5-fold)...")
    candidates = build_candidate_pipelines()
    cv_results = {}
    best_name  = None
    best_score = -1.0

    for name, pipe in candidates.items():
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            scoring='f1',
            n_jobs=-1,
        )
        mean_f1 = scores.mean()
        cv_results[name] = {
            'mean_f1': round(float(mean_f1), 4),
            'std_f1': round(float(scores.std()), 4),
        }
        status = "⬆ best" if mean_f1 > best_score else ""
        print(f"    {name:25s}: F1={mean_f1:.4f} ± {scores.std():.4f} {status}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_name  = name

    print(f"  Best model: {best_name} (F1={best_score:.4f})")

    # ── 6. Final training ─────────────────────────────────────────
    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)

    # ── 7. Validation metrics ─────────────────────────────────────
    y_val_pred = best_pipe.predict(X_val)
    val_metrics = {
        'accuracy':  round(float(accuracy_score(y_val, y_val_pred)), 4),
        'precision': round(float(precision_score(y_val, y_val_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_val, y_val_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_val, y_val_pred, zero_division=0)), 4),
    }
    print(f"\n  Val metrics: {val_metrics}")

    # ── 8. Test metrics (held-out set) ────────────────────────────
    y_test_pred  = best_pipe.predict(X_test)
    y_test_proba = best_pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_test_pred)
    test_metrics = {
        'accuracy':  round(float(accuracy_score(y_test, y_test_pred)), 4),
        'precision': round(float(precision_score(y_test, y_test_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_test_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_test_pred, zero_division=0)), 4),
        'roc_auc':   round(float(roc_auc_score(y_test, y_test_proba)), 4)
                     if len(np.unique(y_test)) > 1 else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_test, y_test_pred, target_names=[labels[0], labels[1]]
        ),
    }

    print(f"\n  {'─'*50}")
    print(f"  TEST RESULTS — {classifier_name.upper()}")
    print(f"  {'─'*50}")
    for k, v in test_metrics.items():
        if k not in ('confusion_matrix', 'classification_report'):
            print(f"  {k:12s}: {v}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              [{labels[0]}, {labels[1]}]")
    for row_label, row in zip([labels[0], labels[1]], cm):
        print(f"    {row_label:12s}: {row.tolist()}")
    print(f"\n{test_metrics['classification_report']}")

    # ── 9. Save model ─────────────────────────────────────────────
    model_data = {
        'classifier_name': classifier_name,
        'model_name': best_name,
        'pipeline': best_pipe,
        'labels': labels,
        'feature_dim': int(X.shape[1]),
        'train_size': len(X_train),
        'val_metrics': val_metrics,
        'test_metrics': {k: v for k, v in test_metrics.items()
                         if k != 'classification_report'},
        'cv_results': cv_results,
        'training_samples': {'positive': n_pos, 'negative': n_neg},
    }

    model_path = MODELS_DIR / cfg['model_file']
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  ✅ Model saved → {model_path}")

    # ── 10. Save metrics JSON ─────────────────────────────────────
    metrics_path = EVAL_DIR / f"{classifier_name}_sklearn_metrics.json"
    metrics_save = {k: v for k, v in test_metrics.items() if k != 'classification_report'}
    metrics_save['val_metrics'] = val_metrics
    metrics_save['cv_results']  = cv_results
    metrics_save['model']       = best_name
    metrics_save['samples']     = {'positive': n_pos, 'negative': n_neg}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)

    print(f"  ✅ Metrics saved → {metrics_path}")

    return metrics_save


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SMCI sklearn baseline classifiers.")
    parser.add_argument(
        '--classifier',
        choices=['pothole', 'garbage', 'all'],
        default='all',
        help='Which classifier to train',
    )
    args = parser.parse_args()

    targets = list(CLASSIFIERS_CONFIG.keys()) if args.classifier == 'all' else [args.classifier]

    all_results = {}
    for clf in targets:
        result = train_binary_classifier(clf)
        if result:
            all_results[clf] = result

    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    if not all_results:
        print("  ❌ No classifiers were trained — dataset not yet available.")
        print("  Run inspect_dataset.py to check readiness, then re-run.")
    else:
        for clf, metrics in all_results.items():
            print(f"  {clf:12s}: accuracy={metrics['accuracy']} f1={metrics['f1']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

