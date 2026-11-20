"""
Dataset Inspection Script — SMCI ML Pipeline
=============================================
Inspects the Mendeley Municipal Complaint Dataset structure before training.
Expected layout:
    dataset/
        pothole/
            yes/   <- pothole images
            no/    <- non-pothole images
        garbage/
            yes/   <- garbage images
            no/    <- non-garbage images

Usage:
    python ml/training/inspect_dataset.py
    python ml/training/inspect_dataset.py --dataset /custom/path
"""
import os
import sys
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

# Try to import PIL for corruption detection
try:
    from PIL import Image, UnidentifiedImageError
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed — corruption detection disabled.")

# Supported image extensions
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

EXPECTED_STRUCTURE = {
    'pothole': ['yes', 'no'],
    'garbage': ['yes', 'no'],
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """MD5 hash of file for exact duplicate detection."""
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def check_image(path: Path):
    """
    Attempts to open and verify an image.
    Returns (is_valid: bool, error_msg: str | None, resolution: tuple | None)
    """
    if not PIL_AVAILABLE:
        return True, None, None
    try:
        with Image.open(path) as img:
            img.verify()          # catches truncated / corrupt headers
        # Re-open (verify closes the file)
        with Image.open(path) as img:
            img.load()            # forces full decode
            return True, None, img.size
    except UnidentifiedImageError:
        return False, "Unidentified image format", None
    except Exception as e:
        return False, str(e), None


def collect_images(folder: Path):
    """Recursively collect image paths from folder, categorising by extension."""
    supported = []
    unsupported = []
    if not folder.exists():
        return supported, unsupported
    for f in folder.rglob('*'):
        if f.is_file():
            if f.suffix.lower() in SUPPORTED_EXTS:
                supported.append(f)
            else:
                # skip hidden / system files silently
                if not f.name.startswith('.'):
                    unsupported.append(f)
    return supported, unsupported


def imbalance_ratio(counts: dict) -> float:
    """Max / Min ratio — 1.0 is perfect balance."""
    vals = [v for v in counts.values() if v > 0]
    if len(vals) < 2:
        return float('inf')
    return max(vals) / min(vals)


def imbalance_label(ratio: float) -> str:
    if ratio == float('inf'):
        return "SEVERE (one class missing)"
    if ratio < 1.5:
        return "Balanced ✅"
    if ratio < 3.0:
        return "Mild imbalance ⚠️"
    return "Severe imbalance ❌"


# ─── Main inspection ──────────────────────────────────────────────────────────

def inspect_dataset(dataset_root: str) -> dict:
    root = Path(dataset_root)
    report = {
        'dataset_root': str(root.resolve()),
        'dataset_exists': root.exists(),
        'classifiers': {},
        'summary': {},
        'ready_for_training': False,
        'warnings': [],
        'errors': [],
    }

    if not root.exists():
        report['errors'].append(f"Dataset root not found: {root.resolve()}")
        return report

    total_images = 0
    total_corrupted = 0
    total_unsupported = 0
    all_hashes = defaultdict(list)   # hash -> list of paths (for duplicate detection)

    for classifier, classes in EXPECTED_STRUCTURE.items():
        clf_report = {
            'classifier': classifier,
            'classes': {},
            'total_images': 0,
            'corrupted': [],
            'unsupported': [],
            'duplicate_hashes': [],
            'ready': False,
        }

        for cls in classes:
            folder = root / classifier / cls
            images, unsup = collect_images(folder)

            class_info = {
                'folder': str(folder),
                'exists': folder.exists(),
                'image_count': 0,
                'corrupted': [],
                'unsupported_files': [str(u) for u in unsup],
                'sample_resolutions': [],
            }

            if not folder.exists():
                report['warnings'].append(
                    f"Missing folder: {folder} — create it and populate with images."
                )
            else:
                valid_count = 0
                resolutions = []
                for img_path in images:
                    is_valid, err_msg, res = check_image(img_path)
                    if is_valid:
                        valid_count += 1
                        if res:
                            resolutions.append(res)
                        # Hash for duplicate detection
                        h = file_hash(img_path)
                        if h:
                            all_hashes[h].append(str(img_path))
                    else:
                        class_info['corrupted'].append({
                            'file': str(img_path),
                            'error': err_msg,
                        })
                        clf_report['corrupted'].append(str(img_path))

                class_info['image_count'] = valid_count
                class_info['unsupported_files'] = [str(u) for u in unsup]
                # Representative resolutions (up to 5)
                class_info['sample_resolutions'] = list(set(resolutions))[:5]

                total_unsupported += len(unsup)
                total_corrupted += len(class_info['corrupted'])

            clf_report['classes'][cls] = class_info
            clf_report['total_images'] += class_info['image_count']

        # Class balance for this classifier
        counts = {cls: clf_report['classes'][cls]['image_count'] for cls in classes}
        ratio = imbalance_ratio(counts)
        clf_report['class_counts'] = counts
        clf_report['imbalance_ratio'] = round(ratio, 2) if ratio != float('inf') else 'inf'
        clf_report['imbalance_label'] = imbalance_label(ratio)
        clf_report['ready'] = all(counts[cls] >= 50 for cls in classes)

        if not clf_report['ready']:
            missing = [cls for cls in classes if counts[cls] < 50]
            report['warnings'].append(
                f"{classifier}: classes {missing} have fewer than 50 images — "
                "need at least 50 per class for meaningful training."
            )

        total_images += clf_report['total_images']
        report['classifiers'][classifier] = clf_report

    # Exact duplicate detection across all images
    duplicates = {h: paths for h, paths in all_hashes.items() if len(paths) > 1}
    report['exact_duplicates'] = {
        'count': sum(len(v) - 1 for v in duplicates.values()),
        'groups': [
            {'hash': h, 'files': paths} for h, paths in list(duplicates.items())[:10]
        ],
    }
    if duplicates:
        report['warnings'].append(
            f"Found {len(duplicates)} groups of exact duplicate images "
            f"({sum(len(v)-1 for v in duplicates.values())} redundant files). "
            "Remove duplicates before training to prevent data leakage."
        )

    report['summary'] = {
        'total_valid_images': total_images,
        'total_corrupted': total_corrupted,
        'total_unsupported_files': total_unsupported,
        'classifiers_ready': sum(1 for c in report['classifiers'].values() if c['ready']),
        'classifiers_total': len(EXPECTED_STRUCTURE),
    }

    # Overall training readiness
    all_ready = all(c['ready'] for c in report['classifiers'].values())
    has_duplicates = report['exact_duplicates']['count'] > 0
    has_corruption = total_corrupted > 0
    report['ready_for_training'] = all_ready and not has_corruption
    if has_duplicates:
        report['ready_for_training'] = False
        report['errors'].append("Remove duplicate images before training.")
    if has_corruption:
        report['errors'].append("Fix or remove corrupted images before training.")

    return report


# ─── Pretty printer ───────────────────────────────────────────────────────────

def print_report(report: dict):
    SEP = "─" * 60

    print("\n" + "═" * 60)
    print("  SMCI DATASET INSPECTION REPORT")
    print("═" * 60)
    print(f"  Dataset root : {report['dataset_root']}")
    print(f"  Dataset exists: {'✅ YES' if report['dataset_exists'] else '❌ NO'}")

    if not report['dataset_exists']:
        print("\n  ❌ Dataset root not found. Download the Mendeley dataset and")
        print("  place it at the path above, then re-run this script.")
        return

    print()
    for clf_name, clf in report['classifiers'].items():
        print(SEP)
        icon = "✅" if clf['ready'] else "❌"
        print(f"  {icon} Classifier: {clf_name.upper()}")
        for cls, info in clf['classes'].items():
            exists = "✅" if info['exists'] else "❌ MISSING"
            count  = info['image_count']
            corr   = len(info['corrupted'])
            unsup  = len(info['unsupported_files'])
            print(f"      {cls:4s}: {count:5d} valid images | {corr} corrupted | "
                  f"{unsup} unsupported | folder {exists}")
            if info['sample_resolutions']:
                res_str = ', '.join(f"{w}×{h}" for w, h in info['sample_resolutions'])
                print(f"             sample resolutions: {res_str}")
        print(f"      Total valid   : {clf['total_images']}")
        print(f"      Class counts  : {clf['class_counts']}")
        print(f"      Imbalance     : {clf['imbalance_label']} (ratio {clf['imbalance_ratio']})")
        if clf['corrupted']:
            print(f"      Corrupted files ({len(clf['corrupted'])}):")
            for c in clf['corrupted'][:5]:
                print(f"        - {c}")

    print()
    print(SEP)
    print("  SUMMARY")
    s = report['summary']
    print(f"  Total valid images       : {s['total_valid_images']}")
    print(f"  Total corrupted          : {s['total_corrupted']}")
    print(f"  Total unsupported files  : {s['total_unsupported_files']}")
    print(f"  Classifiers ready        : {s['classifiers_ready']} / {s['classifiers_total']}")

    # Duplicates
    dup = report['exact_duplicates']
    if dup['count'] > 0:
        print(f"\n  ⚠️  Exact duplicates found: {dup['count']} redundant files")
        for grp in dup['groups'][:3]:
            print(f"     hash {grp['hash'][:8]}…: {grp['files']}")
    else:
        print("  ✅  No exact duplicate images detected.")

    print()
    if report['warnings']:
        print("  WARNINGS:")
        for w in report['warnings']:
            print(f"    ⚠️  {w}")
    if report['errors']:
        print("  ERRORS:")
        for e in report['errors']:
            print(f"    ❌  {e}")

    print()
    status = "✅ READY FOR TRAINING" if report['ready_for_training'] else "❌ NOT READY"
    print(f"  TRAINING READINESS: {status}")
    print("═" * 60 + "\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect SMCI dataset structure.")
    parser.add_argument(
        '--dataset',
        default=os.path.join(
            os.path.dirname(__file__), '..', '..', 'dataset'
        ),
        help='Path to dataset root directory (default: ../../dataset)',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output raw JSON report instead of formatted text',
    )
    args = parser.parse_args()

    report = inspect_dataset(args.dataset)

    if args.json:
        import json
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    # Return exit code based on readiness
    sys.exit(0 if report['ready_for_training'] else 1)


if __name__ == '__main__':
    main()

