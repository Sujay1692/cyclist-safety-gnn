"""
Package the photographs as GitHub release assets.

Run from the project root (the folder that holds image_data/ and dataset_split.csv):

    pip install pandas pillow piexif
    python tools/make_release_assets.py

What it does:
  1. Checks dataset_split.csv against the paper's counts (2970 images; 1518 safe, 1452 unsafe).
  2. Copies the 2970 photographs into release_assets/staging/image_data/{safe,unsafe}/,
     removing EXIF metadata (GPS, device, timestamps). JPEG pixel data are not re-encoded.
  3. Writes zip files of at most ~1.9 GB each (GitHub's per-file limit for release assets
     is 2 GiB):  cyclist_images_safe_part1.zip, cyclist_images_unsafe_part1.zip, ...
     Each zip unpacks to image_data/<class>/..., so all of them can be unzipped into the
     repository root.
  4. Writes release_assets/SHA256SUMS.txt for the zips and PROBLEMS.txt if anything is off.

Your original image_data/ folder is never modified.
"""
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "release_assets"
STAGE = OUT / "staging"
MAX_PART_BYTES = int(1.9e9)
EXPECTED = {"total": 2970, "safe": 1518, "unsafe": 1452}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")
problems = []


def say(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image(cls, stem):
    for ext in IMG_EXT + tuple(e.upper() for e in IMG_EXT):
        p = ROOT / "image_data" / cls / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def strip_metadata(src, dst):
    """Copy a photo without EXIF/GPS/device metadata."""
    ext = src.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        import piexif
        shutil.copy2(src, dst)
        piexif.remove(str(dst))  # removes the EXIF segment only; image data untouched
        return True
    if ext == ".png":
        from PIL import Image
        with Image.open(src) as im:
            im.save(dst, format="PNG")  # lossless; text/EXIF chunks are not carried over
        return True
    shutil.copy2(src, dst)
    problems.append(f"Metadata NOT removed (unsupported format {ext}): {src.name}")
    return False


def write_parts(cls, files):
    """Zip the files of one class into parts no larger than MAX_PART_BYTES."""
    parts, current, size = [], [], 0
    for f in files:
        s = f.stat().st_size
        if current and size + s > MAX_PART_BYTES:
            parts.append(current)
            current, size = [], 0
        current.append(f)
        size += s
    if current:
        parts.append(current)

    written = []
    for i, group in enumerate(parts, 1):
        target = OUT / f"cyclist_images_{cls}_part{i}.zip"
        # photos are already compressed, so store them without recompression
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for f in group:
                z.write(f, f"image_data/{cls}/{f.name}")
        written.append(target)
        say(f"  {target.name}: {len(group)} images, {target.stat().st_size / 1e9:.2f} GB")
    return written


def main():
    split_csv = ROOT / "dataset_split.csv"
    if not split_csv.exists() or not (ROOT / "image_data").exists():
        sys.exit(f"Run this from the project root: dataset_split.csv and image_data/ not found in {ROOT}")

    df = pd.read_csv(split_csv)
    if "is_flipped" in df.columns:
        df = df[~df["is_flipped"].astype(str).str.lower().eq("true")]
    df["stem"] = df["filename"].map(lambda f: Path(str(f).replace("\\", "/")).stem)

    counts = {"total": len(df), **df["class"].value_counts().to_dict()}
    say("Checking counts against the paper:")
    for k, v in EXPECTED.items():
        got = counts.get(k, 0)
        say(f"  {k:<7} expected {v:>5}  found {got:>5}  {'ok' if got == v else 'MISMATCH'}")
        if got != v:
            problems.append(f"Count mismatch for {k}: expected {v}, found {got}")

    if OUT.exists():
        shutil.rmtree(OUT)

    staged = {"safe": [], "unsafe": []}
    cleaned = 0
    for _, r in df.sort_values(["class", "stem"]).iterrows():
        cls, stem = r["class"], r["stem"]
        src = find_image(cls, stem)
        if src is None:
            problems.append(f"Photo missing: image_data/{cls}/{stem}.*")
            continue
        dst = STAGE / "image_data" / cls / f"{stem}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cleaned += strip_metadata(src, dst)
        staged.setdefault(cls, []).append(dst)
    total = sum(len(v) for v in staged.values())
    say(f"\nStaged {total} photos; metadata removed from {cleaned}.")

    say("\nWriting zip files:")
    zips = []
    for cls in sorted(staged):
        zips += write_parts(cls, staged[cls])

    lines = [f"{sha256(z)}  {z.name}" for z in zips]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(STAGE)

    if problems:
        (OUT / "PROBLEMS.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        say(f"\n{len(problems)} problem(s) found - see release_assets/PROBLEMS.txt before uploading.")
    else:
        say("\nAll checks passed. Upload the zip files and SHA256SUMS.txt from release_assets/ to the v1.0 release.")


if __name__ == "__main__":
    main()
