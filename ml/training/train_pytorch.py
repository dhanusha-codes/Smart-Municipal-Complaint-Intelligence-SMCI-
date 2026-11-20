"""
train_pytorch.py — SMCI ML Pipeline
=====================================
6-class MobileNetV3-Small classifier for indian_road_issues/ dataset.

Classes:
  0  broken_edges
  1  construction_zones
  2  cracks
  3  open_manholes
  4  potholes
  5  waterlogged_roads

Design:
  • Transfer learning from ImageNet pretrained MobileNetV3-Small weights
  • Phase 1 (epochs 1–finetune_epoch)  : freeze backbone, train head only
  • Phase 2 (epochs finetune_epoch+1–N): unfreeze all, fine-tune at low LR
  • Exact MD5 dedup applied before splitting (matches sklearn baseline)
  • Stratified 70 / 15 / 15 train / val / test split
  • WeightedRandomSampler + weighted CrossEntropyLoss for class imbalance
  • Best checkpoint saved on val macro-F1 improvement
  • Early stopping on val macro-F1

Usage:
  python ml/training/train_pytorch.py
  python ml/training/train_pytorch.py --epochs 30 --batch_size 16
  python ml/training/train_pytorch.py --no-pretrained
  python ml/training/train_pytorch.py --no-dedup
"""

import sys
import json
import time
import hashlib
import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

# ─── Path setup ───────────────────────────────────────────────────────────────
ML_DIR      = Path(__file__).resolve().parent.parent
REPO_ROOT   = ML_DIR.parent
DATASET_DIR = REPO_ROOT / 'dataset' / 'indian_road_issues'
MODELS_DIR  = ML_DIR / 'models'
EVAL_DIR    = ML_DIR / 'evaluation'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ML_DIR / 'training'))
from preprocessing import get_torch_transforms, TARGET_SIZE, SUPPORTED_EXTS

RANDOM_SEED = 42

CATEGORIES = [
    'broken_edges',
    'construction_zones',
    'cracks',
    'open_manholes',
    'potholes',
    'waterlogged_roads',
]
NUM_CLASSES = len(CATEGORIES)

MODEL_FILE      = MODELS_DIR / 'multiclass_mobilenetv3.pt'
BEST_CKPT_FILE  = MODELS_DIR / 'multiclass_mobilenetv3_best.pt'
METRICS_FILE    = EVAL_DIR   / 'multiclass_pytorch_metrics.json'


# ─── PyTorch requirement check ────────────────────────────────────────────────

def require_torch():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except ImportError:
        print('=' * 60)
        print('PYTORCH NOT INSTALLED')
        print('Install with:')
        print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu')
        print('=' * 60)
        sys.exit(1)


# ─── MD5 deduplication ────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_images(dedup: bool = True) -> List[Tuple[Path, int]]:
    """
    Returns list of (image_path, class_index) for all 6 categories.
    If dedup=True, exact MD5 duplicates are removed globally before returning
    (same logic as train_multiclass_sklearn.py).
    """
    items: List[Tuple[Path, int]] = []
    seen_hashes: set = set()
    skipped = 0

    for idx, cat in enumerate(CATEGORIES):
        cat_dir = DATASET_DIR / cat
        if not cat_dir.exists():
            print(f'  [WARN] Category folder missing: {cat_dir}')
            continue
        cat_count = 0
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXTS or not img_path.is_file():
                continue
            if dedup:
                h = _md5(img_path)
                if h in seen_hashes:
                    skipped += 1
                    continue
                seen_hashes.add(h)
            items.append((img_path, idx))
            cat_count += 1
        print(f'  {cat:<25}: {cat_count} images')

    if dedup and skipped:
        print(f'\n  [dedup] Skipped {skipped} exact MD5 duplicate files.')
    return items


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

def build_dataset_class(torch_mod):
    from torch.utils.data import Dataset
    from PIL import Image

    class MulticlassDataset(Dataset):
        def __init__(self, items: List[Tuple[Path, int]], transform=None):
            self.items = items
            self.transform = transform

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    return MulticlassDataset


# ─── MobileNetV3-Small model ──────────────────────────────────────────────────

def build_mobilenetv3(torchvision_mod, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    try:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        if pretrained:
            print('  Loaded pretrained ImageNet weights.')
    except Exception as e:
        print(f'  [WARN] Could not load pretrained weights: {e}')
        print('  Using randomly initialized MobileNetV3-Small.')
        model = mobilenet_v3_small(weights=None)

    # Freeze entire feature extractor (phase 1)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final linear layer for num_classes output
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def unfreeze_backbone(model):
    """Unfreeze all backbone layers for phase-2 fine-tuning."""
    for param in model.features.parameters():
        param.requires_grad = True
    return model


# ─── Training one epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, torch_mod):
    model.train()
    total_loss = running_correct = total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss     += loss.item() * inputs.size(0)
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total           += inputs.size(0)

    return total_loss / total, running_correct / total


# ─── Evaluation (val / test) ──────────────────────────────────────────────────

def evaluate(model, loader, criterion, device, torch_mod):
    """
    Returns (avg_loss, accuracy, all_preds, all_labels, all_probs).
    all_probs is a list of softmax probability vectors (one per sample).
    """
    import torch.nn.functional as F
    model.eval()

    total_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch_mod.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            probs       = F.softmax(outputs, dim=1)
            preds       = probs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += inputs.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())   # list of 6-element lists

    return total_loss / total, correct / total, all_preds, all_labels, all_probs


# ─── Compute class weights for imbalanced data ────────────────────────────────

def compute_class_weights(train_labels: List[int], num_classes: int, torch_mod, device):
    """
    Inverse-frequency weighting: w_c = total / (num_classes * count_c)
    Returns a FloatTensor of shape [num_classes] on the given device.
    """
    counts = Counter(train_labels)
    total  = len(train_labels)
    weights = [
        total / (num_classes * max(counts.get(c, 1), 1))
        for c in range(num_classes)
    ]
    return torch_mod.tensor(weights, dtype=torch_mod.float32).to(device)


# ─── Main training pipeline ───────────────────────────────────────────────────

def train(
    epochs:         int   = 25,
    batch_size:     int   = 16,
    lr_head:        float = 1e-3,
    lr_finetune:    float = 1e-4,
    finetune_epoch: int   = 10,
    patience:       int   = 7,
    pretrained:     bool  = True,
    dedup:          bool  = True,
) -> Optional[Dict]:

    torch_mod, torchvision_mod = require_torch()

    from torch.utils.data import DataLoader, WeightedRandomSampler
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report,
    )

    # Reproducibility
    torch_mod.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(RANDOM_SEED)

    device = torch_mod.device('cuda' if torch_mod.cuda.is_available() else 'cpu')

    print('\n' + '=' * 65)
    print('  SMCI — MOBILENETV3 6-CLASS TRAINING')
    print(f'  Device    : {device}')
    print(f'  Epochs    : {epochs}')
    print(f'  Batch     : {batch_size}')
    print(f'  Pretrained: {pretrained}')
    print(f'  Dedup     : {"MD5 exact" if dedup else "disabled"}')
    print('=' * 65)

    # ── 1. Load images ────────────────────────────────────────────
    print('\n[1/8] Loading image paths...')
    items = collect_images(dedup=dedup)
    if len(items) < 30:
        print(f'[ERROR] Too few images ({len(items)}). Cannot train.')
        return None

    random.shuffle(items)
    dist = Counter(label for _, label in items)
    print(f'\n  Total usable images: {len(items)}')
    for idx, cat in enumerate(CATEGORIES):
        print(f'    {cat:<25}: {dist.get(idx, 0)}')

    # ── 2. Stratified train / val / test split (70/15/15) ─────────
    print('\n[2/8] Splitting dataset (stratified 70/15/15)...')
    paths  = [str(p) for p, _ in items]
    labels = [l for _, l in items]

    p_trainval, p_test, l_trainval, l_test = train_test_split(
        paths, labels, test_size=0.15, random_state=RANDOM_SEED, stratify=labels
    )
    p_train, p_val, l_train, l_val = train_test_split(
        p_trainval, l_trainval,
        test_size=0.15 / 0.85,
        random_state=RANDOM_SEED, stratify=l_trainval
    )

    train_items = [(Path(p), l) for p, l in zip(p_train, l_train)]
    val_items   = [(Path(p), l) for p, l in zip(p_val,   l_val)]
    test_items  = [(Path(p), l) for p, l in zip(p_test,  l_test)]

    print(f'  Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}')

    # ── 3. DataLoaders ────────────────────────────────────────────
    print('\n[3/8] Building DataLoaders...')
    Dataset = build_dataset_class(torch_mod)

    train_ds = Dataset(train_items, transform=get_torch_transforms('train'))
    val_ds   = Dataset(val_items,   transform=get_torch_transforms('val'))
    test_ds  = Dataset(test_items,  transform=get_torch_transforms('val'))

    # WeightedRandomSampler — oversample minority classes
    sample_weights = [
        1.0 / max(dist.get(label, 1), 1)
        for _, label in train_items
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,    num_workers=0)

    # ── 4. Build model ────────────────────────────────────────────
    print('\n[4/8] Building MobileNetV3-Small...')
    model = build_mobilenetv3(torchvision_mod, num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)

    head_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total params     : {total_params:,}')
    print(f'  Trainable (head) : {head_params:,}  (backbone frozen)')

    # ── 5. Loss + optimizer + scheduler ──────────────────────────
    class_weights = compute_class_weights(l_train, NUM_CLASSES, torch_mod, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch_mod.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head, weight_decay=1e-4,
    )
    scheduler = torch_mod.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    # ── 6. Training loop ──────────────────────────────────────────
    print('\n[5/8] Training...\n')

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [], 'val_macro_f1': [],
    }

    best_val_f1    = -1.0
    best_state     = None
    no_improve     = 0
    phase          = 1

    for epoch in range(1, epochs + 1):

        # Switch to phase 2 (unfreeze backbone)
        if epoch == finetune_epoch + 1 and phase == 1:
            phase = 2
            print(f'\n  [Epoch {epoch}] Unfreezing backbone for fine-tuning (lr={lr_finetune}).')
            model = unfreeze_backbone(model)
            optimizer = torch_mod.optim.AdamW(
                model.parameters(), lr=lr_finetune, weight_decay=1e-4,
            )
            scheduler = torch_mod.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3,
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'  Trainable params (all): {trainable:,}\n')

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, torch_mod
        )
        val_loss, val_acc, val_preds, val_labels, _ = evaluate(
            model, val_loader, criterion, device, torch_mod
        )

        val_f1 = float(f1_score(val_labels, val_preds, average='macro', zero_division=0))
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        history['train_loss'].append(round(train_loss, 4))
        history['train_acc'].append(round(train_acc, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['val_macro_f1'].append(round(val_f1, 4))

        improved = '  ← best' if val_f1 > best_val_f1 else ''
        print(
            f'  Epoch {epoch:3d}/{epochs} | Ph{phase} | '
            f'TLoss={train_loss:.4f} TAcc={train_acc:.4f} | '
            f'VLoss={val_loss:.4f} VAcc={val_acc:.4f} VF1={val_f1:.4f} | '
            f'{elapsed:.1f}s{improved}'
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = deepcopy(model.state_dict())
            torch_mod.save(
                {'state_dict': best_state, 'epoch': epoch, 'val_macro_f1': val_f1},
                BEST_CKPT_FILE,
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'\n  Early stopping triggered at epoch {epoch}.')
            break

    if best_state is None:
        print('[ERROR] Training did not produce a valid checkpoint.')
        return None

    # ── 7. Test evaluation on held-out set ────────────────────────
    print(f'\n[6/8] Loading best checkpoint (val macro-F1={best_val_f1:.4f})...')
    model.load_state_dict(best_state)

    print('[7/8] Running test set evaluation...')
    _, _, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device, torch_mod
    )

    if not test_labels:
        print('[ERROR] Test set is empty.')
        return None

    acc     = float(accuracy_score(test_labels, test_preds))
    mac_f1  = float(f1_score(test_labels, test_preds, average='macro',    zero_division=0))
    wgt_f1  = float(f1_score(test_labels, test_preds, average='weighted', zero_division=0))
    cm      = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))
    report  = classification_report(
        test_labels, test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CATEGORIES,
        zero_division=0, digits=4,
    )

    # Per-class metrics
    prec_per = precision_score(test_labels, test_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    rec_per  = recall_score(   test_labels, test_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    f1_per   = f1_score(       test_labels, test_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    support  = [sum(1 for l in test_labels if l == c) for c in range(NUM_CLASSES)]

    # Terminal report
    SEP = '─' * 65
    print(f'\n{"=" * 65}')
    print('  TEST RESULTS — MOBILENETV3 6-CLASS')
    print(f'{"=" * 65}')
    print(f'  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)')
    print(f'  Macro F1        : {mac_f1:.4f}')
    print(f'  Weighted F1     : {wgt_f1:.4f}')
    print(f'\n  Per-class metrics:')
    print(f'  {"Class":<25}  {"Prec":>7}  {"Recall":>7}  {"F1":>7}  {"Supp":>6}')
    print(f'  {SEP}')
    for i, cat in enumerate(CATEGORIES):
        bar = '█' * int(f1_per[i] * 15)
        print(f'  {cat:<25}  {prec_per[i]:>7.4f}  {rec_per[i]:>7.4f}  {f1_per[i]:>7.4f}  {support[i]:>6}  {bar}')
    print(f'  {SEP}')
    print(f'  {"macro avg":<25}  {prec_per.mean():>7.4f}  {rec_per.mean():>7.4f}  {mac_f1:>7.4f}')

    print(f'\n  Confusion matrix (rows=actual, cols=predicted):')
    abbr = ['br_edg','constr','cracks','manhol','pothole','waterlog']
    hdr  = '  {:>14}  '.format('') + '  '.join(f'{a:>8}' for a in abbr)
    print(hdr)
    for i, row in enumerate(cm):
        print(f'  {CATEGORIES[i][:14]:>14}  ' + '  '.join(f'{v:>8}' for v in row))

    print(f'\n{report}')

    # ── 8. Save model + metrics ───────────────────────────────────
    print('[8/8] Saving model and metrics...')

    per_class_metrics = {
        CATEGORIES[i]: {
            'precision': round(float(prec_per[i]), 4),
            'recall':    round(float(rec_per[i]),  4),
            'f1':        round(float(f1_per[i]),   4),
            'support':   support[i],
        }
        for i in range(NUM_CLASSES)
    }

    test_metrics = {
        'accuracy':          round(acc,    4),
        'macro_f1':          round(mac_f1, 4),
        'weighted_f1':       round(wgt_f1, 4),
        'per_class':         per_class_metrics,
        'confusion_matrix':  cm.tolist(),
        'best_val_macro_f1': round(best_val_f1, 4),
    }

    torch_mod.save(
        {
            'state_dict':    model.state_dict(),
            'num_classes':   NUM_CLASSES,
            'categories':    CATEGORIES,
            'architecture':  'mobilenetv3_small',
            'target_size':   list(TARGET_SIZE),
            'test_metrics':  test_metrics,
            'dataset_counts': {cat: dist.get(i, 0) for i, cat in enumerate(CATEGORIES)},
            'training_params': {
                'epochs':         epochs,
                'batch_size':     batch_size,
                'lr_head':        lr_head,
                'lr_finetune':    lr_finetune,
                'finetune_epoch': finetune_epoch,
                'patience':       patience,
                'pretrained':     pretrained,
                'dedup':          dedup,
            },
            'history': history,
        },
        MODEL_FILE,
    )
    print(f'  Model saved  → {MODEL_FILE}')

    with open(METRICS_FILE, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f'  Metrics saved → {METRICS_FILE}')

    print('\n' + '=' * 65)
    print(f'  TRAINING COMPLETE')
    print(f'  Accuracy : {acc:.4f}   Macro-F1 : {mac_f1:.4f}')
    print('=' * 65)

    return test_metrics


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train SMCI 6-class MobileNetV3-Small on indian_road_issues/.'
    )
    parser.add_argument('--epochs',         type=int,   default=25)
    parser.add_argument('--batch_size',     type=int,   default=16)
    parser.add_argument('--lr_head',        type=float, default=1e-3)
    parser.add_argument('--lr_finetune',    type=float, default=1e-4)
    parser.add_argument('--finetune_epoch', type=int,   default=10)
    parser.add_argument('--patience',       type=int,   default=7)
    parser.add_argument('--no-pretrained',  dest='pretrained', action='store_false')
    parser.add_argument('--no-dedup',       dest='dedup',      action='store_false')
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        finetune_epoch=args.finetune_epoch,
        patience=args.patience,
        pretrained=args.pretrained,
        dedup=args.dedup,
    )


if __name__ == '__main__':
    main()
