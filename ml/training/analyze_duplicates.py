"""
analyze_duplicates.py
Full duplicate group analysis for dataset/indian_road_issues/
READ-ONLY: does not modify, delete, move or rename any files.
"""

import os
import hashlib
from collections import defaultdict

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset", "indian_road_issues"
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images(root: str):
    """Returns list of (category, filename, full_path)."""
    items = []
    if not os.path.isdir(root):
        print(f"[ERROR] Dataset root not found: {root}")
        return items
    for cat in sorted(os.listdir(root)):
        cat_path = os.path.join(root, cat)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                items.append((cat, fname, os.path.join(cat_path, fname)))
    return items


def main():
    root = os.path.abspath(DATASET_ROOT)
    print("=" * 70)
    print("DUPLICATE ANALYSIS REPORT")
    print(f"Dataset: {root}")
    print("=" * 70)

    items = collect_images(root)
    if not items:
        print("[ERROR] No images found.")
        return

    total = len(items)
    print(f"\nTotal images scanned: {total}")

    # --- Compute hashes ---
    print("\nComputing MD5 hashes...")
    hash_map = defaultdict(list)  # hash -> [(cat, fname, path)]
    for i, (cat, fname, path) in enumerate(items):
        h = md5(path)
        hash_map[h].append((cat, fname, path))
        if (i + 1) % 100 == 0:
            print(f"  Hashed {i + 1}/{total}...")

    # --- Identify duplicate groups ---
    dup_groups = {h: files for h, files in hash_map.items() if len(files) > 1}

    total_groups = len(dup_groups)
    total_redundant = sum(len(v) - 1 for v in dup_groups.values())

    within_category_groups = []
    cross_category_groups = []

    for h, files in sorted(dup_groups.items(), key=lambda x: x[0]):
        cats = set(f[0] for f in files)
        if len(cats) == 1:
            within_category_groups.append((h, files))
        else:
            cross_category_groups.append((h, files))

    print(f"\nDuplicate groups found     : {total_groups}")
    print(f"  Within-category groups   : {len(within_category_groups)}")
    print(f"  Cross-category groups    : {len(cross_category_groups)}")
    print(f"Redundant files (extras)   : {total_redundant}")

    # ------------------------------------------------------------------ #
    # SECTION 1: ALL DUPLICATE GROUPS
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SECTION 1 — ALL DUPLICATE GROUPS")
    print("=" * 70)

    group_num = 0
    for h, files in sorted(dup_groups.items(), key=lambda x: x[0]):
        cats = set(f[0] for f in files)
        group_num += 1
        kind = "CROSS-CATEGORY" if len(cats) > 1 else "within-category"
        print(f"\n[Group {group_num:03d}] {kind}  |  hash={h[:12]}...  |  {len(files)} copies")
        for cat, fname, _ in sorted(files, key=lambda x: (x[0], x[1])):
            print(f"    {cat}/{fname}")

    # ------------------------------------------------------------------ #
    # SECTION 2: CROSS-CATEGORY DUPLICATES — DETAILED
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SECTION 2 — CROSS-CATEGORY DUPLICATES (DETAILED)")
    print("=" * 70)

    if not cross_category_groups:
        print("  None found.")
    else:
        for idx, (h, files) in enumerate(cross_category_groups, 1):
            cats = sorted(set(f[0] for f in files))
            print(f"\n[Cross-Category Group {idx}]  hash={h[:16]}...")
            print(f"  Categories involved: {', '.join(cats)}")
            for cat, fname, _ in sorted(files, key=lambda x: (x[0], x[1])):
                print(f"    {cat}/{fname}")

    # ------------------------------------------------------------------ #
    # SECTION 3: UNIQUE IMAGE COUNT PER CATEGORY AFTER DEDUP
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SECTION 3 — UNIQUE IMAGE COUNT PER CATEGORY (AFTER DEDUP)")
    print("=" * 70)
    print("Strategy: for each hash group, keep exactly ONE copy per category.\n")

    # For each hash group, count how many UNIQUE categories it appears in.
    # Unique images per category = images that have a unique hash in that cat
    #                              + one representative per hash that appears in that cat.

    # Build: cat -> set of hashes present in that category
    cat_hashes = defaultdict(set)
    cat_total = defaultdict(int)
    for cat, fname, path in items:
        h_val = md5(path)  # already computed but simplest to redo for clarity
        cat_hashes[cat].add(h_val)
        cat_total[cat] += 1

    # Re-use already computed hash_map
    # unique per category = number of distinct hashes that have at least one file in that cat
    # (already captured in cat_hashes)

    TRAIN_THRESHOLD = 50
    any_below = False

    for cat in sorted(cat_hashes.keys()):
        unique = len(cat_hashes[cat])
        original = cat_total[cat]
        removed = original - unique
        flag = ""
        if unique < TRAIN_THRESHOLD:
            flag = "  ⚠ BELOW THRESHOLD"
            any_below = True
        print(f"  {cat:<25}  original={original:>4}  unique={unique:>4}  removed={removed:>4}{flag}")

    total_unique = sum(len(v) for v in cat_hashes.values())
    print(f"\n  Total unique images across all categories: {total_unique}")
    print(f"  (Shared cross-category images counted once per category)")

    if any_below:
        print(f"\n  ⚠ WARNING: One or more categories fall below the {TRAIN_THRESHOLD}-image training threshold.")
    else:
        print(f"\n  ✓ All categories remain above the {TRAIN_THRESHOLD}-image training threshold after dedup.")

    # ------------------------------------------------------------------ #
    # SECTION 4: SUMMARY & RECOMMENDATION
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SECTION 4 — SUMMARY & RECOMMENDATION")
    print("=" * 70)
    print(f"""
  Total images in dataset      : {total}
  Total duplicate groups       : {total_groups}
    Within-category groups     : {len(within_category_groups)}
    Cross-category groups      : {len(cross_category_groups)}
  Total redundant files        : {total_redundant}
  Unique images (global)       : {total - total_redundant}

  Cross-category duplicates are the most harmful for ML training
  because the same image appears under two different labels.

  Recommendation:
  - For cross-category groups: remove all copies except the one in
    the most representative category (requires user decision).
  - For within-category groups: remove extra copies to save disk
    space and avoid overrepresentation during training.
  - Use --no-dedup flag in training scripts to skip duplicates at
    runtime without deleting files.

  NO FILES HAVE BEEN MODIFIED BY THIS SCRIPT.
""")
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()

