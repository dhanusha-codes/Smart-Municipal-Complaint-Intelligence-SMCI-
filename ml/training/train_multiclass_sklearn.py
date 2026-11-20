"""
train_multiclass_sklearn.py — SMCI ML Pipeline
================================================
6-class multiclass classifier for indian_road_issues/ dataset.

Categories:
  broken_edges | construction_zones | cracks |
  open_manholes | potholes | waterlogged_roads

Algorithm: handcrafted image features (colour histograms, texture,
           edge density, aspect ratio) → RandomForest / SVM / GBM
           selected by 5-fold cross-validation.

Does NOT require PyTorch or OpenCV. Works with Pillow + scikit-learn.

Dedup strategy: exact MD5 duplicates are detected and only the FIRST
copy of each unique hash is kept for training (--no-dedup to disable).

Usage:
    python ml/training/train_multiclass_sklearn.py
    python ml/training/train_multiclass_sklearn.py --no-dedup
    python ml/training/train_multiclass_sklearn.py --model random_forest
"""

import os
import sys
import json
import hashlib
import argparse
import pickle
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
ML_DIR      = Path(__file__).resolve().parent.parent
REPO_ROOT   = ML_DIR.parent
DATASET_DIR = REPO_ROOT / 'dataset' / 'indian_road_issues'
MODELS_DIR  = ML_DIR / 'models'
EVAL_DIR    = ML_DIR / 'evaluation'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ML_DIR / 'training'))
from preprocessing import extract_features, SUPPORTED_EXTS

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORIES = [
    'broken_edges',
    'construction_zones',
    'cracks',
    'open_manholes',
    'potholes',
    'waterlogged_roads',
]

MODEL_FILE   = MODELS_DIR / 'multiclass_sklearn.pkl'
METRICS_FILE = EVAL_DIR   / 'multiclass_sklearn_metrics.json'


# ─── Utilities ────────────────────────────────────────────────────────────────

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_images(dataset_dir: Path, dedup: bool = True) -> List[Tuple[Path, str]]:
    """
    Returns list of (image_path, category_label).
    If dedup=True, exact MD5 duplicates are removed (first occurrence kept).
    """
    items: List[Tuple[Path, str]] = []
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset not found: {dataset_dir}")
        return items

    seen_hashes: set = set()
    skipped = 0

    for cat in CATEGORIES:
        cat_dir = dataset_dir / cat
        if not cat_dir.exists():
            print(f"  [WARN] Category folder missing: {cat_dir}")
            continue
        cat_count = 0
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            if not img_path.is_file():
                continue
            if dedup:
                h = md5(img_path)
                if h in seen_hashes:
                    skipped += 1
                    continue
                seen_hashes.add(h)
            items.append((img_path, cat))
            cat_count += 1
        print(f"  {cat:<25}: {cat_count} images loaded")

    if dedup and skipped:
        print(f"\n  [dedup] Skipped {skipped} exact duplicate files.")
    return items


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_all_features(
    items: List[Tuple[Path, str]],
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Extract features for every image.
    Returns (X, y_encoded, label_encoder).
    """
    le = LabelEncoder()
    le.fit(CATEGORIES)

    X, y_raw, paths = [], [], []
    total = len(items)
    failed = 0

    for i, (img_path, label) in enumerate(items):
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"\r  Feature extraction: {i+1}/{total}", end='', flush=True)
        feat = extract_features(img_path)
        if feat is None:
            failed += 1
            continue
        X.append(feat)
        y_raw.append(label)
        paths.append(img_path)

    print()  # newline after progress
    if failed:
        print(f"  [WARN] {failed} images could not be read and were skipped.")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = le.transform(y_raw)
    return X_arr, y_arr, le


# ─── Model candidates ─────────────────────────────────────────────────────────

def build_pipelines(force_model: Optional[str] = None) -> Dict[str, Pipeline]:
    all_pipelines = {
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1,
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
        'gradient_boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.07,
                max_depth=4,
                random_state=RANDOM_SEED,
            )),
        ]),
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_SEED,
            )),
        ]),
    }
    if force_model and force_model in all_pipelines:
        return {force_model: all_pipelines[force_model]}
    return all_pipelines


# ─── Training pipeline ────────────────────────────────────────────────────────

def train(dedup: bool = True, force_model: Optional[str] = None) -> Optional[Dict]:
    print('\n' + '=' * 65)
    print('  SMCI — MULTICLASS SKLEARN TRAINING')
    print(f'  Dataset : {DATASET_DIR}')
    print(f'  Dedup   : {"enabled (MD5)" if dedup else "disabled"}')
    print('=' * 65)

    # 1. Load images
    print('\n[1/6] Loading image paths...')
    items = collect_images(DATASET_DIR, dedup=dedup)
    if len(items) < 30:
        print(f'\n[ERROR] Too few images ({len(items)}). Cannot train.')
        return None

    random.shuffle(items)
    print(f'\n  Total usable images: {len(items)}')

    # Class distribution
    from collections import Counter
    dist = Counter(label for _, label in items)
    for cat in CATEGORIES:
        print(f'    {cat:<25}: {dist.get(cat, 0)}')

    # 2. Feature extraction
    print('\n[2/6] Extracting image features...')
    X, y, le = extract_all_features(items)
    print(f'  Feature vector length : {X.shape[1]}')
    print(f'  Total valid samples   : {len(X)}')

    if len(X) < 20:
        print('[ERROR] Too few valid feature vectors.')
        return None

    # 3. Train / val / test split  (70 / 15 / 15)
    print('\n[3/6] Splitting dataset...')
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15 / 0.85,
        random_state=RANDOM_SEED, stratify=y_trainval,
    )
    print(f'  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

    # 4. Cross-validation model selection
    print('\n[4/6] Cross-validating candidates (5-fold, metric=macro-F1)...')
    pipelines = build_pipelines(force_model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}
    best_name  = None
    best_score = -1.0

    for name, pipe in pipelines.items():
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring='f1_macro', n_jobs=-1,
        )
        mean_f1 = scores.mean()
        cv_results[name] = {
            'mean_f1_macro': round(float(mean_f1), 4),
            'std':           round(float(scores.std()), 4),
        }
        flag = '  ← best' if mean_f1 > best_score else ''
        print(f'    {name:<25}: macro-F1 = {mean_f1:.4f} ± {scores.std():.4f}{flag}')
        if mean_f1 > best_score:
            best_score = mean_f1
            best_name  = name

    print(f'\n  Selected: {best_name}  (macro-F1 = {best_score:.4f})')

    # 5. Final training on full train set
    print('\n[5/6] Training final model...')
    best_pipe = pipelines[best_name]
    best_pipe.fit(X_train, y_train)

    # Validation
    y_val_pred = best_pipe.predict(X_val)
    val_acc    = accuracy_score(y_val, y_val_pred)
    val_f1     = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    print(f'  Val accuracy : {val_acc:.4f}')
    print(f'  Val macro-F1 : {val_f1:.4f}')

    # Test
    y_test_pred = best_pipe.predict(X_test)
    test_acc    = accuracy_score(y_test, y_test_pred)
    test_f1     = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    cm          = confusion_matrix(y_test, y_test_pred)
    report      = classification_report(
        y_test, y_test_pred,
        target_names=le.classes_,
        zero_division=0,
    )

    print('\n' + '─' * 65)
    print('  TEST RESULTS')
    print('─' * 65)
    print(f'  Accuracy        : {test_acc:.4f}')
    print(f'  Macro F1-Score  : {test_f1:.4f}')
    print(f'\n  Per-class report:\n{report}')
    print(f'  Confusion matrix (rows=actual, cols=predicted):')
    print(f'  Classes: {list(le.classes_)}')
    for i, row in enumerate(cm):
        print(f'    {le.classes_[i]:<25}: {row.tolist()}')

    # 6. Save model and metrics
    print('\n[6/6] Saving model and metrics...')

    model_data = {
        'classifier_name'  : 'multiclass_6class',
        'model_name'       : best_name,
        'pipeline'         : best_pipe,
        'label_encoder'    : le,
        'categories'       : list(le.classes_),
        'feature_dim'      : int(X.shape[1]),
        'train_size'       : len(X_train),
        'val_metrics'      : {'accuracy': round(float(val_acc), 4),
                              'macro_f1': round(float(val_f1), 4)},
        'test_metrics'     : {'accuracy': round(float(test_acc), 4),
                              'macro_f1': round(float(test_f1), 4),
                              'confusion_matrix': cm.tolist()},
        'cv_results'       : cv_results,
        'dataset_counts'   : {cat: int(dist.get(cat, 0)) for cat in CATEGORIES},
        'dedup_enabled'    : dedup,
    }

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Model saved  → {MODEL_FILE}')

    metrics_out = {
        'model'          : best_name,
        'categories'     : list(le.classes_),
        'accuracy'       : round(float(test_acc), 4),
        'macro_f1'       : round(float(test_f1), 4),
        'val_metrics'    : model_data['val_metrics'],
        'cv_results'     : cv_results,
        'confusion_matrix': cm.tolist(),
        'dataset_counts' : model_data['dataset_counts'],
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'  Metrics saved → {METRICS_FILE}')

    print('\n' + '=' * 65)
    print(f'  TRAINING COMPLETE')
    print(f'  Model   : {best_name}')
    print(f'  Accuracy: {test_acc:.4f}   Macro-F1: {test_f1:.4f}')
    print('=' * 65)

    return metrics_out


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train 6-class SMCI sklearn classifier on indian_road_issues/.'
    )
    parser.add_argument(
        '--no-dedup', dest='dedup', action='store_false',
        help='Disable MD5 exact-duplicate removal during training.',
    )
    parser.add_argument(
        '--model',
        choices=['random_forest', 'svm_rbf', 'gradient_boosting', 'logistic_regression'],
        default=None,
        help='Force a specific model (skip cross-validation selection).',
    )
    args = parser.parse_args()
    train(dedup=args.dedup, force_model=args.model)


if __name__ == '__main__':
    main()
