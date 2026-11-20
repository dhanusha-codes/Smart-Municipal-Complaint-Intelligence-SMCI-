"""
Dataset Inspection — dataset/indian_road_issues/
Checks: image counts, formats, corruption, exact duplicates.
READ-ONLY: does not modify, move or delete any files.
"""
import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image, UnidentifiedImageError
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[WARN] Pillow not found — corruption check skipped.")

DATASET_ROOT = Path(r"c:\Users\dhanu\OneDrive\Scans\Desktop\Smart-Municipal-Complaint-Intelligence\dataset\indian_road_issues")
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def check_image(path: Path):
    if not PIL_OK:
        return True, None, None
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            return True, None, img.size
    except UnidentifiedImageError:
        return False, "Unidentified/unknown format", None
    except Exception as e:
        return False, str(e)[:80], None

# ── Gather all files ──────────────────────────────────────────────────────────
categories = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])

SEP  = "─" * 64
SEP2 = "═" * 64

print()
print(SEP2)
print("  SMCI DATASET INSPECTION  —  indian_road_issues/")
print(SEP2)
print(f"  Dataset path : {DATASET_ROOT}")
print(f"  Pillow       : {'available' if PIL_OK else 'NOT available (install Pillow)'}")
print()

global_valid   = 0
global_corrupt = 0
global_unsup   = 0
all_hashes     = defaultdict(list)   # md5 → [paths]
category_rows  = []

for cat_dir in categories:
    cat_name = cat_dir.name
    all_files   = list(cat_dir.rglob("*"))
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
    unsup_files = [f for f in all_files if f.is_file() and f.suffix.lower() not in SUPPORTED_EXTS and not f.name.startswith('.')]

    valid_count   = 0
    corrupt_files = []
    ext_counts    = defaultdict(int)

    for img_path in image_files:
        ext_counts[img_path.suffix.lower()] += 1
        is_ok, err_msg, size = check_image(img_path)
        if is_ok:
            valid_count += 1
            # Hash for duplicate detection
            h = file_md5(img_path)
            if h:
                all_hashes[h].append(str(img_path))
        else:
            corrupt_files.append((img_path.name, err_msg))

    global_valid   += valid_count
    global_corrupt += len(corrupt_files)
    global_unsup   += len(unsup_files)

    ext_str = "  ".join(f"{ext}:{cnt}" for ext, cnt in sorted(ext_counts.items()))
    category_rows.append({
        'name':       cat_name,
        'total':      len(image_files),
        'valid':      valid_count,
        'corrupt':    corrupt_files,
        'unsup':      unsup_files,
        'ext_str':    ext_str,
    })

# ── Per-category table ────────────────────────────────────────────────────────
print(f"  {'Category':<22} {'Images':>7}  {'Valid':>6}  {'Corrupt':>7}  Formats")
print(f"  {'-'*22} {'-'*7}  {'-'*6}  {'-'*7}  {'-'*20}")
for row in category_rows:
    status = "✅" if row['corrupt'] == [] else "⚠️ "
    print(f"  {status} {row['name']:<20} {row['total']:>7}  {row['valid']:>6}  {len(row['corrupt']):>7}  {row['ext_str']}")

# ── Totals ────────────────────────────────────────────────────────────────────
grand_total = global_valid + global_corrupt + global_unsup
print()
print(SEP)
print(f"  TOTALS")
print(SEP)
print(f"  Total files found      : {grand_total}")
print(f"  Supported image files  : {global_valid + global_corrupt}")
print(f"  Valid (readable)       : {global_valid}")
print(f"  Corrupted / unreadable : {global_corrupt}")
print(f"  Unsupported formats    : {global_unsup}")

# ── Corruption details ────────────────────────────────────────────────────────
all_corrupt = [(row['name'], f, e) for row in category_rows for f, e in row['corrupt']]
if all_corrupt:
    print()
    print(SEP)
    print(f"  CORRUPTED FILES ({len(all_corrupt)} total)")
    print(SEP)
    for cat, fname, err in all_corrupt:
        print(f"  [{cat}] {fname}  →  {err}")
else:
    print()
    print(f"  ✅ No corrupted or unreadable images found.")

# ── Exact duplicate detection ────────────────────────────────────────────────
dup_groups = {h: paths for h, paths in all_hashes.items() if len(paths) > 1}
n_dup_files = sum(len(v) - 1 for v in dup_groups.values())

print()
print(SEP)
print(f"  EXACT DUPLICATE CHECK  (MD5 hash)")
print(SEP)
if dup_groups:
    print(f"  ⚠️  Found {len(dup_groups)} groups of exact duplicates ({n_dup_files} redundant files):")
    for i, (h, paths) in enumerate(list(dup_groups.items())[:10], 1):
        print(f"  Group {i} (hash {h[:12]}…):")
        for p in paths:
            folder = Path(p).parent.name
            fname  = Path(p).name
            print(f"    [{folder}] {fname}")
    if len(dup_groups) > 10:
        print(f"  … and {len(dup_groups)-10} more groups.")
else:
    print(f"  ✅ No exact duplicate images detected.")

# ── Class imbalance ───────────────────────────────────────────────────────────
counts = {row['name']: row['valid'] for row in category_rows}
min_c  = min(counts.values()) if counts else 0
max_c  = max(counts.values()) if counts else 0
ratio  = round(max_c / max(min_c, 1), 2)

print()
print(SEP)
print(f"  CLASS BALANCE")
print(SEP)
for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    bar_len = int((cnt / max(max_c, 1)) * 30)
    bar = "█" * bar_len
    print(f"  {cat:<22}  {cnt:>4}  {bar}")
print()
print(f"  Min class : {min_c}  |  Max class : {max_c}  |  Imbalance ratio : {ratio}")
if ratio > 2.0:
    print(f"  ⚠️  Imbalance ratio > 2.0 — consider class weighting or augmentation during training.")
elif ratio > 1.5:
    print(f"  ⚠️  Mild imbalance (ratio {ratio}) — monitor per-class metrics during training.")
else:
    print(f"  ✅ Dataset is reasonably balanced.")

# ── Training readiness ───────────────────────────────────────────────────────
print()
print(SEP2)
print(f"  TRAINING READINESS SUMMARY")
print(SEP2)
ready = global_corrupt == 0 and min_c >= 50 and not dup_groups
print(f"  Total valid images     : {global_valid}")
print(f"  Categories             : {len(category_rows)}")
print(f"  Min images per class   : {min_c}  {'✅' if min_c >= 50 else '❌ (<50)'}")
print(f"  Corrupted images       : {global_corrupt}  {'✅' if global_corrupt == 0 else '❌ fix before training'}")
print(f"  Exact duplicates       : {n_dup_files}  {'✅' if n_dup_files == 0 else '⚠️  review before training'}")
print(f"  Dataset ready          : {'✅ YES — ready for training' if ready else '⚠️  Review warnings above'}")
print(SEP2)

