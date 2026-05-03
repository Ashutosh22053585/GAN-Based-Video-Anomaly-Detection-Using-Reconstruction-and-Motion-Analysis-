"""
download_dataset.py
===================
Downloads and organises the UCSD Pedestrian Dataset (Ped2).
If auto-download fails, prints manual Kaggle instructions.

Run:
    python download_dataset.py
"""

import os
import sys
import zipfile
import tarfile
import urllib.request
import shutil
import config


DEST_ZIP    = os.path.join(config.DATA_RAW_DIR, "ucsd_ped2.zip")
EXTRACT_DIR = config.DATA_RAW_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  PROGRESS REPORTER
# ─────────────────────────────────────────────────────────────────────────────

class _Progress:
    def __init__(self):
        self._last = -1

    def __call__(self, blocks, block_size, total):
        if total <= 0:
            return
        pct = min(100, int(blocks * block_size * 100 / total))
        if pct != self._last and pct % 10 == 0:
            print(f"  {pct}% ...", flush=True)
            self._last = pct


def _try_download(url, dest):
    try:
        print(f"  Trying: {url}")
        urllib.request.urlretrieve(url, dest, _Progress())
        ok = os.path.exists(dest) and os.path.getsize(dest) > 50_000
        if ok:
            print(f"  Downloaded OK ({os.path.getsize(dest) // 1024} KB)")
        return ok
    except Exception as e:
        print(f"  Failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract(archive_path, dest_dir):
    print(f"\n[Setup] Extracting {os.path.basename(archive_path)} ...")
    # Auto-detect format: try tar.gz first (UCSD server sends tar.gz regardless of URL name)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")
    print("[Setup] Extraction complete.")


def _normalise_layout():
    """Rename extracted folders to the expected Train/ and Test/ names."""
    raw = config.DATA_RAW_DIR

    # Common folder names that appear after extraction
    train_candidates = [
        os.path.join(raw, "UCSD_ped2",          "train"),
        os.path.join(raw, "ped2",                "train"),
        os.path.join(raw, "UCSDped2",            "Train"),
        os.path.join(raw, "UCSD_Anomaly_Dataset", "UCSDped2", "Train"),
    ]
    test_candidates = [
        os.path.join(raw, "UCSD_ped2",          "test"),
        os.path.join(raw, "ped2",                "test"),
        os.path.join(raw, "UCSDped2",            "Test"),
        os.path.join(raw, "UCSD_Anomaly_Dataset", "UCSDped2", "Test"),
    ]

    train_dst = os.path.join(raw, "Train")
    test_dst  = os.path.join(raw, "Test")

    for src in train_candidates:
        if os.path.isdir(src) and src != train_dst:
            shutil.move(src, train_dst)
            print(f"[Setup] Moved {src} -> {train_dst}")
            break

    for src in test_candidates:
        if os.path.isdir(src) and src != test_dst:
            shutil.move(src, test_dst)
            print(f"[Setup] Moved {src} -> {test_dst}")
            break


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary():
    VIDEO_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    print("\n" + "=" * 50)
    print("  Dataset Summary")
    print("=" * 50)
    for split in ("Train", "Test"):
        split_dir = os.path.join(config.DATA_RAW_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"  {split}/  -> NOT FOUND")
            continue
        clips = [d for d in os.listdir(split_dir)
                 if os.path.isdir(os.path.join(split_dir, d))]
        total_frames = sum(
            len([f for f in os.listdir(os.path.join(split_dir, c))
                 if os.path.splitext(f)[1].lower() in VIDEO_EXTS])
            for c in clips
        )
        print(f"  {split}/  -> {len(clips)} clips, {total_frames} frames")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _print_manual_instructions():
    print("\n" + "=" * 60)
    print("  AUTO-DOWNLOAD FAILED -- Get the dataset manually:")
    print("=" * 60)
    print()
    print("  [OPTION 1] Kaggle (recommended, free account needed):")
    print("  https://www.kaggle.com/datasets/dineshpiyasamara/ucsd-anomaly-dataset")
    print("  -> Download and extract the zip.")
    print()
    print("  [OPTION 2] Original UCSD server:")
    print("  http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm")
    print("  -> Download UCSD_Anomaly_Dataset.v1p2.tar.gz")
    print()
    print("  After downloading, arrange files like this:")
    print("    data/raw/Train/")
    print("      Train001/  (contains 000.tif, 001.tif, ...)")
    print("      Train002/")
    print("      ...")
    print("    data/raw/Test/")
    print("      Test001/")
    print("      ...")
    print()
    print("  Then re-run:  python download_dataset.py")
    print("=" * 60)


def main():
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)

    train_dir = os.path.join(config.DATA_RAW_DIR, "Train")

    # Skip if already downloaded
    if os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0:
        print("[Setup] Dataset already present -- skipping download.")
        _print_summary()
        return

    print("[Setup] Attempting to download UCSD Ped2 dataset ...\n")

    # Mirror URLs to try (ordered by reliability)
    urls = [
        "https://github.com/StevenLiuWen/ano_pred_cvpr2018/releases/download/v1.0/UCSD_ped2.zip",
        "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
    ]
    # Try local archive first if it exists
    if os.path.exists(DEST_ZIP) and os.path.getsize(DEST_ZIP) > 1000000:
        print(f"[Setup] Found local archive: {DEST_ZIP}")
        success = True
    else:
        # Try mirrors in order
        success = False
        for url in urls:
            if _try_download(url, DEST_ZIP):
                success = True
                break

    if not success:
        _print_manual_instructions()
        sys.exit(1)

    _extract(DEST_ZIP, EXTRACT_DIR)
    _normalise_layout()

    if os.path.exists(DEST_ZIP):
        os.remove(DEST_ZIP)

    _print_summary()
    print("\n[Setup] Dataset ready!")


if __name__ == "__main__":
    main()
