# SMCI — ML Pipeline

Machine Learning pipeline for the Smart Municipal Complaint Intelligence system.

## Directory Structure

```
ml/
├── training/
│   ├── inspect_dataset.py   # Dataset readiness check
│   ├── preprocessing.py     # Shared image preprocessing
│   ├── train_sklearn.py     # Baseline (no GPU required)
│   └── train_pytorch.py     # MobileNetV3 (requires PyTorch)
├── inference/
│   └── classifier.py        # Unified classifier interface (used by Flask)
├── evaluation/
│   └── evaluate_models.py   # Post-training evaluation + plots
└── models/                  # Saved model files (created after training)

dataset/
├── pothole/
│   ├── yes/    ← place pothole images here
│   └── no/     ← place non-pothole images here
└── garbage/
    ├── yes/    ← place garbage images here
    └── no/     ← place non-garbage images here
```

## Workflow

### Step 1 — Prepare Dataset

After the Mendeley dataset finishes downloading, extract it and place images into
the four folders above (pothole/yes, pothole/no, garbage/yes, garbage/no).

Then verify:
```
python ml/training/inspect_dataset.py
```
Expected output when ready:
- Classifier: POTHOLE ✅
- Classifier: GARBAGE ✅
- TRAINING READINESS: ✅ READY FOR TRAINING

### Step 2 — Train Baseline (Scikit-Learn, no GPU needed)

Scikit-learn is already installed. This trains RandomForest/SVM with handcrafted features:

```bash
# Train both classifiers
python ml/training/train_sklearn.py --classifier all

# Or individual
python ml/training/train_sklearn.py --classifier pothole
python ml/training/train_sklearn.py --classifier garbage
```

Outputs:
- `ml/models/pothole_sklearn.pkl`
- `ml/models/garbage_sklearn.pkl`
- `ml/evaluation/pothole_sklearn_metrics.json`
- `ml/evaluation/garbage_sklearn_metrics.json`

### Step 3 — Train Deep Learning Model (MobileNetV3, optional)

First install PyTorch (CPU-only, works on any laptop):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then train:
```bash
python ml/training/train_pytorch.py --classifier all --epochs 25
```

For fine-tuning control:
```bash
python ml/training/train_pytorch.py --classifier all --epochs 30 --finetune_epoch 12 --patience 8
```

Outputs:
- `ml/models/pothole_mobilenetv3.pt`
- `ml/models/garbage_mobilenetv3.pt`
- `ml/evaluation/pothole_pytorch_metrics.json`
- `ml/evaluation/garbage_pytorch_metrics.json`

### Step 4 — Evaluate Models

After training, evaluate on the held-out test set:
```bash
python ml/evaluation/evaluate_models.py --classifier all --plots
```
Generates confusion matrices, ROC curves, and PR curves as PNG files.

### Step 5 — Restart Flask

Flask auto-loads available models on startup:
```bash
cd backend
python run.py
```
Check `/api/ml/status` to confirm model is loaded.

## Inference Priority

The Flask classifier uses this fallback chain:
1. **PyTorch MobileNetV3** (highest accuracy, requires PyTorch + trained .pt file)
2. **Scikit-learn RandomForest** (no GPU, requires trained .pkl file)  
3. **Heuristic** (always available, keyword + colour analysis, capped at 65% confidence)

## Environment

| Package     | Version | Required for         |
|-------------|---------|----------------------|
| Pillow      | 12.3.0  | ✅ installed         |
| NumPy       | 2.4.2   | ✅ installed         |
| scikit-learn| 1.8.0   | ✅ installed         |
| scipy       | 1.17.1  | ✅ installed         |
| matplotlib  | 3.11.1  | ✅ installed (plots) |
| torch       | —       | ❌ not installed     |
| torchvision | —       | ❌ not installed     |

## Model Architecture (MobileNetV3-Small)

- Input: 224×224 RGB, ImageNet-normalized
- Backbone: MobileNetV3-Small (pretrained on ImageNet)
- Phase 1 (epochs 1–10): Backbone frozen, only classifier head trained (lr=1e-3)
- Phase 2 (epoch 11+): Full fine-tuning (lr=1e-4)
- Output: 2-class softmax (binary: yes/no)
- Augmentations: RandomCrop, HorizontalFlip, ColorJitter, Rotation

## Notes

- All results come from actual model inference — no hardcoded predictions.
- Heuristic fallback is clearly labelled `heuristic_fallback` in API responses.
- Near-duplicate removal (perceptual hash, Hamming ≤5) prevents data leakage.
- Stratified 70/15/15 split with the same random seed for reproducibility.

