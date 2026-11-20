"""
Model Evaluation Script — SMCI ML Pipeline
===========================================
Evaluates saved models on the held-out test set and generates
confusion matrices, ROC curves, and a comprehensive JSON report.
Run after training is complete.

Usage:
    python ml/evaluation/evaluate_models.py --classifier all
    python ml/evaluation/evaluate_models.py --classifier pothole --plots
"""
import os
import sys
import json
import pickle
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
)

ML_DIR      = Path(__file__).resolve().parent.parent
REPO_ROOT   = ML_DIR.parent
DATASET_DIR = REPO_ROOT / 'dataset'
MODELS_DIR  = ML_DIR / 'models'
EVAL_DIR    = ML_DIR / 'evaluation'

EVAL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ML_DIR / 'training'))
from preprocessing import extract_features, get_torch_transforms, TARGET_SIZE

RANDOM_SEED = 42
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

CLASSIFIER_CONFIG = {
    'pothole': {
        'positive_dir': DATASET_DIR / 'pothole' / 'yes',
        'negative_dir': DATASET_DIR / 'pothole' / 'no',
        'class_names':  ['no_pothole', 'pothole'],
        'sklearn_file': MODELS_DIR / 'pothole_sklearn.pkl',
        'pytorch_file': MODELS_DIR / 'pothole_mobilenetv3.pt',
    },
    'garbage': {
        'positive_dir': DATASET_DIR / 'garbage' / 'yes',
        'negative_dir': DATASET_DIR / 'garbage' / 'no',
        'class_names':  ['no_garbage', 'garbage'],
        'sklearn_file': MODELS_DIR / 'garbage_sklearn.pkl',
        'pytorch_file': MODELS_DIR / 'garbage_mobilenetv3.pt',
    },
}


def load_test_items(cfg: Dict, test_fraction: float = 0.15) -> List[Tuple[Path, int]]:
    """
    Load test-set images using the same seed/split as training.
    Returns list of (path, label).
    """
    from sklearn.model_selection import train_test_split

    pos = [(p, 1) for p in sorted(cfg['positive_dir'].iterdir())
           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS] \
          if cfg['positive_dir'].exists() else []
    neg = [(p, 0) for p in sorted(cfg['negative_dir'].iterdir())
           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS] \
          if cfg['negative_dir'].exists() else []

    all_items = pos + neg
    if not all_items:
        return []

    paths  = [p for p, _ in all_items]
    labels = [l for _, l in all_items]

    # Replicate the train/test split from training
    _, p_test, _, l_test = train_test_split(
        paths, labels, test_size=test_fraction, random_state=RANDOM_SEED, stratify=labels
    )
    return list(zip(p_test, l_test))


def evaluate_sklearn(clf_name: str, test_items: List, plots: bool = False) -> Optional[Dict]:
    """Evaluate sklearn model on held-out test items."""
    cfg       = CLASSIFIER_CONFIG[clf_name]
    model_file = cfg['sklearn_file']

    if not model_file.exists():
        print(f"  [sklearn] Model not found: {model_file}")
        return None

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    pipeline   = model_data['pipeline']
    labels_map = model_data['labels']

    # Extract features
    print(f"  Extracting sklearn features for {len(test_items)} test images...")
    X, y = [], []
    for path, label in test_items:
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(label)

    if not X:
        print("  ❌ No features extracted.")
        return None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    y_pred  = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    return _compute_metrics(clf_name, 'sklearn', y, y_pred, y_proba, cfg['class_names'], plots)


def evaluate_pytorch(clf_name: str, test_items: List, plots: bool = False) -> Optional[Dict]:
    """Evaluate PyTorch model on held-out test items."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("  [pytorch] PyTorch not installed — skipping.")
        return None

    cfg        = CLASSIFIER_CONFIG[clf_name]
    model_file = cfg['pytorch_file']
    if not model_file.exists():
        print(f"  [pytorch] Model not found: {model_file}")
        return None

    from torchvision.models import mobilenet_v3_small
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    num_classes = ckpt.get('num_classes', 2)

    model = mobilenet_v3_small(weights=None)
    in_f  = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, num_classes)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    transform = get_torch_transforms('test')

    class _DS(Dataset):
        def __init__(self, items, tf):
            self.items = items
            self.tf = tf
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            path, label = self.items[idx]
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                img = Image.new('RGB', TARGET_SIZE, (128, 128, 128))
            return self.tf(img) if self.tf else img, label

    loader = DataLoader(_DS(test_items, transform), batch_size=32, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, lbls in loader:
            outputs = model(inputs)
            probs   = F.softmax(outputs, dim=1)[:, 1]
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(lbls.numpy().tolist())
            all_probs.extend(probs.numpy().tolist())

    return _compute_metrics(
        clf_name, 'pytorch_mobilenetv3',
        np.array(all_labels), np.array(all_preds), np.array(all_probs),
        cfg['class_names'], plots
    )


def _compute_metrics(
    clf_name: str, backend: str,
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
    class_names: List[str], plots: bool,
) -> Dict:
    """Compute and optionally plot full evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'classifier':  clf_name,
        'backend':     backend,
        'n_test':      int(len(y_true)),
        'accuracy':    round(float(accuracy_score(y_true, y_pred)), 4),
        'precision':   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall':      round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1':          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'roc_auc':     round(float(roc_auc_score(y_true, y_proba)), 4)
                       if len(np.unique(y_true)) > 1 else None,
        'confusion_matrix': cm.tolist(),
        'class_names':  class_names,
    }

    print(f"\n  {'─'*50}")
    print(f"  EVALUATION: {clf_name.upper()} [{backend}]  n={metrics['n_test']}")
    print(f"  {'─'*50}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1-score  : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              {class_names}")
    for row_lbl, row in zip(class_names, cm):
        print(f"    {row_lbl:12s}: {row.tolist()}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")

    if plots:
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f'{clf_name.capitalize()} — {backend}', fontsize=14)

            # Confusion matrix heatmap
            ax = axes[0]
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(class_names, rotation=20)
            ax.set_yticklabels(class_names)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')
            plt.colorbar(im, ax=ax)

            # ROC curve
            if metrics['roc_auc']:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                axes[1].plot(fpr, tpr, lw=2, label=f'AUC={metrics["roc_auc"]:.3f}')
                axes[1].plot([0,1],[0,1],'--', color='grey', alpha=0.5)
                axes[1].set_xlabel('False Positive Rate')
                axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title('ROC Curve')
                axes[1].legend()

            # Precision-Recall curve
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            axes[2].plot(rec, prec, lw=2, color='darkorange')
            axes[2].set_xlabel('Recall')
            axes[2].set_ylabel('Precision')
            axes[2].set_title('Precision-Recall Curve')

            plt.tight_layout()
            plot_path = EVAL_DIR / f'{clf_name}_{backend}_eval.png'
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"  📊 Plot saved → {plot_path}")
            metrics['plot_path'] = str(plot_path)
        except Exception as e:
            print(f"  [plot] Warning: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMCI trained models.")
    parser.add_argument('--classifier', choices=['pothole', 'garbage', 'all'], default='all')
    parser.add_argument('--plots', action='store_true', help='Generate evaluation plots')
    parser.add_argument('--backend', choices=['sklearn', 'pytorch', 'both'], default='both')
    args = parser.parse_args()

    targets = list(CLASSIFIER_CONFIG.keys()) if args.classifier == 'all' else [args.classifier]
    all_results = {}

    for clf in targets:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {clf.upper()}")
        print(f"{'='*60}")

        cfg = CLASSIFIER_CONFIG[clf]
        test_items = load_test_items(cfg)

        if not test_items:
            print(f"  ❌ No test images found — dataset not available.")
            continue

        print(f"  Test set size: {len(test_items)}")
        results = {}

        if args.backend in ('sklearn', 'both'):
            r = evaluate_sklearn(clf, test_items, plots=args.plots)
            if r:
                results['sklearn'] = r

        if args.backend in ('pytorch', 'both'):
            r = evaluate_pytorch(clf, test_items, plots=args.plots)
            if r:
                results['pytorch'] = r

        if results:
            all_results[clf] = results
            # Save per-classifier report
            report_path = EVAL_DIR / f'{clf}_evaluation_report.json'
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  ✅ Report saved → {report_path}")

    # Save combined report
    if all_results:
        combined_path = EVAL_DIR / 'combined_evaluation_report.json'
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Combined report → {combined_path}")

    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)
    if not all_results:
        print("  ❌ No models evaluated — train first with train_sklearn.py")
    else:
        for clf, bkds in all_results.items():
            for bkd, m in bkds.items():
                print(f"  {clf:10s} [{bkd:20s}]: acc={m['accuracy']} f1={m['f1']} auc={m['roc_auc']}")
    print("="*60)


if __name__ == '__main__':
    main()

