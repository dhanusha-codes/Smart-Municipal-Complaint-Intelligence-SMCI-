"""
train_multiclass_pytorch.py — SMCI ML Pipeline
================================================
6-class MobileNetV3-Small image classifier.
Dataset: dataset/indian_road_issues/

Classes (index → folder name):
  0  broken_edges
  1  construction_zones
  2  cracks
  3  open_manholes
  4  potholes
  5  waterlogged_roads

Design decisions:
  • Transfer learning: ImageNet pretrained MobileNetV3-Small weights
  • Phase 1 (epochs 1 → finetune_epoch):  only classifier head is trained
  • Phase 2 (epochs finetune_epoch+1 → N): entire network fine-tuned at low LR
  • Global MD5 exact-duplicate removal applied BEFORE any split
  • Stratified 70 / 15 / 15  train / val / test split (matches sklearn baseline)
  • WeightedRandomSampler oversamples minority classes in every training batch
  • CrossEntropyLoss (no manual class weighting — sampler handles imbalance)
  • Best checkpoint selected on validation MACRO F1 (never on test metrics)
  • Early stopping on validation macro F1

Pretrained weights:
  The script expects TORCH_HOME to be set, or uses the project-local cache at
  .torch_cache/ inside the repository root.  Run the download command once
  (requires network) before training in the sandbox:

      set TORCH_HOME=<repo>\.torch_cache && python -c "
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
      "

Usage:
  python ml/training/train_multiclass_pytorch.py
  python ml/training/train_multiclass_pytorch.py --epochs 30 --batch_size 16
  python ml/training/train_multiclass_pytorch.py --no-pretrained   # random init
  python ml/training/train_multiclass_pytorch.py --no-dedup        # skip MD5 dedup
  python ml/training/train_multiclass_pytorch.py --finetune_epoch 8 --patience 5
"""

import sys
import os
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

# ─── Paths ────────────────────────────────────────────────────────────────────
ML_DIR      = Path(__file__).resolve().parent.parent
REPO_ROOT   = ML_DIR.parent
DATASET_DIR = REPO_ROOT / 'dataset' / 'indian_road_issues'
MODELS_DIR  = ML_DIR / 'models'
EVAL_DIR    = ML_DIR / 'evaluation'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Local weights cache inside the repo — avoids permission issues with ~/.cache
TORCH_CACHE = REPO_ROOT / '.torch_cache'

sys.path.insert(0, str(ML_DIR / 'training'))
from preprocessing import get_torch_transforms, TARGET_SIZE, SUPPORTED_EXTS

RANDOM_SEED = 42

CATEGORIES: List[str] = [
    'broken_edges',
    'construction_zones',
    'cracks',
    'open_manholes',
    'potholes',
    'waterlogged_roads',
]
NUM_CLASSES = len(CATEGORIES)

MODEL_FILE     = MODELS_DIR / 'multiclass_mobilenetv3.pt'
BEST_CKPT_FILE = MODELS_DIR / 'multiclass_mobilenetv3_best.pt'
METRICS_FILE   = EVAL_DIR   / 'multiclass_pytorch_metrics.json'


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


# ─── MD5 exact deduplication ──────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_images(dedup: bool = True) -> List[Tuple[Path, int]]:
    """
    Returns list of (image_path, class_index).
    If dedup=True, a global set of MD5 hashes is maintained across ALL
    categories — identical to the logic in train_multiclass_sklearn.py.
    The first occurrence (by sorted folder + sorted filename) is kept.
    No files are deleted or moved.
    """
    items: List[Tuple[Path, int]] = []
    seen_hashes: set = set()
    skipped = 0

    for idx, cat in enumerate(CATEGORIES):
        cat_dir = DATASET_DIR / cat
        if not cat_dir.exists():
            print(f'  [WARN] Missing folder: {cat_dir}')
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
        print(f'    {cat:<25}: {cat_count}')

    if dedup and skipped:
        print(f'  [dedup] {skipped} exact MD5 duplicates removed (in-memory only — no files deleted)')
    return items


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

def _make_dataset_class(torch_mod):
    from torch.utils.data import Dataset
    from PIL import Image as PILImage

    class MulticlassImageDataset(Dataset):
        """Minimal dataset: (PIL Image → transform → tensor, int label)."""

        def __init__(self, items: List[Tuple[Path, int]], transform=None):
            self.items = items
            self.transform = transform

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            path, label = self.items[idx]
            img = PILImage.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    return MulticlassImageDataset


# ─── MobileNetV3-Small model ──────────────────────────────────────────────────

def build_model(torchvision_mod, num_classes: int, pretrained: bool):
    """
    Load MobileNetV3-Small.  When pretrained=True, reads weights from
    TORCH_CACHE (the project-local cache), falling back to random init
    if the cache file is absent.
    """
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    if pretrained:
        weights_file = TORCH_CACHE / 'hub' / 'checkpoints' / 'mobilenet_v3_small-047dcff4.pth'
        if weights_file.exists():
            # Load from local file — no network needed
            import torch
            model = mobilenet_v3_small(weights=None)
            state = torch.load(str(weights_file), map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            print(f'  Pretrained weights loaded from local cache: {weights_file}')
        else:
            # Try to download (requires network — will fail in sandbox)
            try:
                os.environ.setdefault('TORCH_HOME', str(TORCH_CACHE))
                model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                print('  Pretrained weights downloaded from PyTorch hub.')
            except Exception as e:
                print(f'  [WARN] Could not load pretrained weights: {e}')
                print('  Falling back to random initialization.')
                model = mobilenet_v3_small(weights=None)
    else:
        model = mobilenet_v3_small(weights=None)
        print('  Random weight initialization (--no-pretrained).')

    # Freeze entire backbone — only the new head will be trained in phase 1
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final linear layer for 6-class output
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def unfreeze_backbone(model):
    """Enable gradients on all backbone layers for phase-2 fine-tuning."""
    for param in model.features.parameters():
        param.requires_grad = True
    return model


# ─── Training one epoch ───────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, device, torch_mod):
    model.train()
    total_loss = n_correct = n_total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        n_correct  += (outputs.argmax(dim=1) == labels).sum().item()
        n_total    += inputs.size(0)

    return total_loss / n_total, n_correct / n_total


# ─── Validation / test evaluation ────────────────────────────────────────────

def _evaluate(model, loader, criterion, device, torch_mod):
    """
    Returns (avg_loss, accuracy, all_preds, all_labels).
    Model is set to eval mode; gradients are disabled.
    """
    import torch.nn.functional as F
    model.eval()

    total_loss = n_correct = n_total = 0
    all_preds, all_labels = [], []

    with torch_mod.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds       = outputs.argmax(dim=1)
            n_correct  += (preds == labels).sum().item()
            n_total    += inputs.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / n_total, n_correct / n_total, all_preds, all_labels


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

    # ── Reproducibility ───────────────────────────────────────────
    torch_mod.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device = torch_mod.device('cuda' if torch_mod.cuda.is_available() else 'cpu')

    print('\n' + '=' * 65)
    print('  SMCI — MOBILENETV3-SMALL  6-CLASS TRAINING')
    print(f'  Device         : {device}')
    print(f'  Epochs         : {epochs}  (head-only until epoch {finetune_epoch})')
    print(f'  Batch size     : {batch_size}')
    print(f'  LR (head)      : {lr_head}')
    print(f'  LR (finetune)  : {lr_finetune}')
    print(f'  Patience       : {patience}')
    print(f'  Pretrained     : {pretrained}')
    print(f'  Dedup          : {"MD5 exact (global)" if dedup else "disabled"}')
    print('=' * 65)

    # ── 1. Load images with MD5 dedup ─────────────────────────────
    print('\n[1/8] Loading image paths...')
    items = collect_images(dedup=dedup)

    if len(items) < 30:
        print(f'[ERROR] Only {len(items)} images found. Cannot train.')
        return None

    random.shuffle(items)
    dist = Counter(label for _, label in items)
    print(f'\n  Total usable images : {len(items)}')
    for idx, cat in enumerate(CATEGORIES):
        print(f'    {cat:<25}: {dist.get(idx, 0)}')

    # ── 2. Stratified 70 / 15 / 15 split ─────────────────────────
    print('\n[2/8] Stratified train / val / test split (70/15/15)...')
    paths  = [str(p) for p, _ in items]
    labels = [l    for _, l in items]

    p_tv, p_test, l_tv, l_test = train_test_split(
        paths, labels,
        test_size=0.15, random_state=RANDOM_SEED, stratify=labels,
    )
    p_train, p_val, l_train, l_val = train_test_split(
        p_tv, l_tv,
        test_size=0.15 / 0.85, random_state=RANDOM_SEED, stratify=l_tv,
    )

    train_items = [(Path(p), l) for p, l in zip(p_train, l_train)]
    val_items   = [(Path(p), l) for p, l in zip(p_val,   l_val)]
    test_items  = [(Path(p), l) for p, l in zip(p_test,  l_test)]

    print(f'  Train : {len(train_items)}')
    print(f'  Val   : {len(val_items)}')
    print(f'  Test  : {len(test_items)}')

    # ── 3. DataLoaders ────────────────────────────────────────────
    print('\n[3/8] Building DataLoaders...')
    Dataset = _make_dataset_class(torch_mod)

    train_ds = Dataset(train_items, transform=get_torch_transforms('train'))
    val_ds   = Dataset(val_items,   transform=get_torch_transforms('val'))
    test_ds  = Dataset(test_items,  transform=get_torch_transforms('val'))

    # WeightedRandomSampler: weight per sample = 1 / class_count_in_train
    train_class_dist = Counter(l_train)
    sample_weights   = [1.0 / max(train_class_dist[l], 1) for l in l_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,    num_workers=0, pin_memory=False)

    # ── 4. Build model ────────────────────────────────────────────
    print('\n[4/8] Building MobileNetV3-Small...')
    model = build_model(torchvision_mod, num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params          : {total_params:,}')
    print(f'  Trainable (phase 1)   : {trainable_params:,}  (head only)')

    # ── 5. Loss / optimizer / scheduler ──────────────────────────
    # Standard CrossEntropyLoss — class imbalance handled by WeightedRandomSampler
    criterion = nn.CrossEntropyLoss()

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

    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0
    phase       = 1

    for epoch in range(1, epochs + 1):

        # ── Phase transition: unfreeze backbone ───────────────────
        if epoch == finetune_epoch + 1 and phase == 1:
            phase = 2
            model = unfreeze_backbone(model)
            optimizer = torch_mod.optim.AdamW(
                model.parameters(), lr=lr_finetune, weight_decay=1e-4,
            )
            scheduler = torch_mod.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3,
            )
            all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'\n  [Epoch {epoch}] Phase 2 — backbone unfrozen | '
                  f'trainable params: {all_trainable:,} | lr={lr_finetune}')

        t0 = time.time()
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device, torch_mod
        )
        val_loss, val_acc, val_preds, val_labels = _evaluate(
            model, val_loader, criterion, device, torch_mod
        )
        val_f1  = float(f1_score(val_labels, val_preds, average='macro', zero_division=0))
        elapsed = time.time() - t0

        scheduler.step(val_f1)

        history['train_loss'].append(round(train_loss, 4))
        history['train_acc'].append(round(train_acc, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['val_macro_f1'].append(round(val_f1, 4))

        flag = '  ★ best' if val_f1 > best_val_f1 else ''
        print(
            f'  Ep {epoch:3d}/{epochs} | Ph{phase} | '
            f'TLoss={train_loss:.4f} TAcc={train_acc:.4f} | '
            f'VLoss={val_loss:.4f} VAcc={val_acc:.4f} VF1={val_f1:.4f} | '
            f'{elapsed:.1f}s{flag}'
        )

        # ── Checkpoint on improvement ─────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = deepcopy(model.state_dict())
            torch_mod.save(
                {
                    'state_dict':    best_state,
                    'epoch':         epoch,
                    'val_macro_f1':  val_f1,
                    'num_classes':   NUM_CLASSES,
                    'categories':    CATEGORIES,
                },
                BEST_CKPT_FILE,
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs).')
            break

    if best_state is None:
        print('[ERROR] No checkpoint was saved during training.')
        return None

    # ── 7. Final test evaluation (best checkpoint, held-out set) ──
    print(f'\n[6/8] Loading best checkpoint (val macro-F1 = {best_val_f1:.4f})...')
    model.load_state_dict(best_state)

    print('[7/8] Evaluating on held-out test set...')
    _, _, test_preds, test_labels = _evaluate(
        model, test_loader, criterion, device, torch_mod
    )

    if not test_labels:
        print('[ERROR] Test set is empty.')
        return None

    acc    = float(accuracy_score(test_labels, test_preds))
    mac_f1 = float(f1_score(test_labels, test_preds, average='macro',    zero_division=0))
    wgt_f1 = float(f1_score(test_labels, test_preds, average='weighted', zero_division=0))
    cm     = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        test_labels, test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CATEGORIES,
        zero_division=0, digits=4,
    )

    prec_per = precision_score(test_labels, test_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    rec_per  = recall_score(   test_labels, test_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    f1_per   = f1_score(       test_labels, test_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    support  = [sum(1 for l in test_labels if l == c) for c in range(NUM_CLASSES)]

    # Terminal report
    SEP = '─' * 65
    print('\n' + '=' * 65)
    print('  TEST RESULTS — MOBILENETV3-SMALL  6-CLASS')
    print('=' * 65)
    print(f'  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)')
    print(f'  Macro F1        : {mac_f1:.4f}')
    print(f'  Weighted F1     : {wgt_f1:.4f}')
    print(f'  Best val F1     : {best_val_f1:.4f}  (used for checkpoint)')
    print(f'\n  Per-class metrics:')
    print(f'  {"Class":<25}  {"Prec":>7}  {"Recall":>7}  {"F1":>7}  {"Supp":>5}')
    print(f'  {SEP}')
    for i, cat in enumerate(CATEGORIES):
        bar = '█' * int(f1_per[i] * 15)
        print(f'  {cat:<25}  {prec_per[i]:>7.4f}  {rec_per[i]:>7.4f}  {f1_per[i]:>7.4f}  {support[i]:>5}  {bar}')
    print(f'  {SEP}')
    print(f'  {"macro avg":<25}  {prec_per.mean():>7.4f}  {rec_per.mean():>7.4f}  {mac_f1:>7.4f}')
    print(f'  {"weighted avg":<25}  {wgt_f1:>7.4f}')

    print(f'\n  Confusion matrix (rows=actual, cols=predicted):')
    abbr = ['br_edg', 'constr', 'cracks', 'manhol', 'pothole', 'wtrlog']
    print('  {:>14}  '.format('') + '  '.join(f'{a:>7}' for a in abbr))
    print('  ' + SEP)
    for i, row in enumerate(cm):
        print(f'  {CATEGORIES[i][:14]:>14}  ' + '  '.join(f'{v:>7}' for v in row))

    print(f'\n{report}')
    print('=' * 65)
    print('  TRAINING COMPLETE')
    print('=' * 65)

    # ── 8. Save final model + metrics ─────────────────────────────
    print('\n[8/8] Saving model and metrics...')

    per_class_out = {
        CATEGORIES[i]: {
            'precision': round(float(prec_per[i]), 4),
            'recall':    round(float(rec_per[i]),  4),
            'f1':        round(float(f1_per[i]),   4),
            'support':   int(support[i]),
        }
        for i in range(NUM_CLASSES)
    }

    test_metrics = {
        'accuracy':          round(acc,    4),
        'macro_f1':          round(mac_f1, 4),
        'weighted_f1':       round(wgt_f1, 4),
        'best_val_macro_f1': round(best_val_f1, 4),
        'per_class':         per_class_out,
        'confusion_matrix':  cm.tolist(),
    }

    torch_mod.save(
        {
            'state_dict':    model.state_dict(),
            'num_classes':   NUM_CLASSES,
            'categories':    CATEGORIES,
            'architecture':  'mobilenetv3_small',
            'target_size':   list(TARGET_SIZE),
            'test_metrics':  test_metrics,
            'dataset_counts': {cat: int(dist.get(i, 0)) for i, cat in enumerate(CATEGORIES)},
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
    print(f'  Final model → {MODEL_FILE}')

    with open(METRICS_FILE, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f'  Metrics     → {METRICS_FILE}')

    return test_metrics


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Train SMCI 6-class MobileNetV3-Small on indian_road_issues/.'
    )
    p.add_argument('--epochs',         type=int,   default=25,  help='Total training epochs (default 25)')
    p.add_argument('--batch_size',     type=int,   default=16,  help='Batch size (default 16)')
    p.add_argument('--lr_head',        type=float, default=1e-3, help='LR for head-only phase (default 1e-3)')
    p.add_argument('--lr_finetune',    type=float, default=1e-4, help='LR for fine-tune phase (default 1e-4)')
    p.add_argument('--finetune_epoch', type=int,   default=10,  help='Epoch at which backbone is unfrozen (default 10)')
    p.add_argument('--patience',       type=int,   default=7,   help='Early-stopping patience in epochs (default 7)')
    p.add_argument('--no-pretrained',  dest='pretrained', action='store_false',
                   help='Use random weight init instead of ImageNet pretrained')
    p.add_argument('--no-dedup',       dest='dedup',      action='store_false',
                   help='Disable MD5 exact-duplicate removal')
    args = p.parse_args()

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

