#!/usr/bin/env python3
"""Burn a copyright line into the bottom of gallery photos.

Processes every image under assets/img/gallery/ (recursively) and stamps
"(c) <year> <author>" into the bottom-left corner of the pixels, so the notice
survives downloading or reuse. It also writes the copyright into the file's
metadata (EXIF/PNG text).

It is idempotent: a JSON manifest (scripts/.watermark-manifest.json) records the
hash of each already-stamped file, and stamped files carry a metadata marker, so
re-running only touches new or replaced photos.

Usage:
    python3 scripts/watermark_images.py            # stamp new photos
    python3 scripts/watermark_images.py --force    # re-stamp everything
    python3 scripts/watermark_images.py --author "Jane Doe" --year 2025

The git pre-commit hook (scripts/hooks/pre-commit) runs this automatically.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join("assets", "img", "gallery")
MANIFEST = os.path.join(ROOT, "scripts", ".watermark-manifest.json")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
COPYRIGHT_TAG = 0x8298  # EXIF "Copyright"
FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def author_from_config():
    cfg = os.path.join(ROOT, "_config.yml")
    try:
        with open(cfg, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*author:\s*(.+?)\s*$", line)
                if m:
                    return m.group(1).strip().strip("\"'")
    except OSError:
        pass
    return "Alpha Diallo"


def load_font(size):
    for path in FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def has_marker(path):
    """True if the file already carries our copyright metadata marker."""
    try:
        img = Image.open(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            val = img.getexif().get(COPYRIGHT_TAG)
        elif ext == ".png":
            val = (getattr(img, "text", {}) or {}).get("Copyright") or img.info.get("Copyright")
        else:
            return False
        return bool(val) and str(val).startswith("(c)")
    except Exception:
        return False


def stamp(path, drawn_text, meta_text):
    ext = os.path.splitext(path)[1].lower()
    img = ImageOps.exif_transpose(Image.open(path))
    keep_alpha = ext == ".png" and img.mode in ("RGBA", "LA", "P")
    img = img.convert("RGBA" if keep_alpha else "RGB")

    width, height = img.size
    font_size = max(15, int(width * 0.026))
    font = load_font(font_size)
    margin = max(10, int(width * 0.022))
    stroke = max(1, font_size // 14)

    draw = ImageDraw.Draw(img, "RGBA")
    bbox = draw.textbbox((0, 0), drawn_text, font=font, stroke_width=stroke)
    x = margin - bbox[0]
    y = height - margin - bbox[3]
    draw.text((x, y), drawn_text, font=font, fill=(255, 255, 255, 235),
              stroke_width=stroke, stroke_fill=(0, 0, 0, 150))

    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        if img.mode != "RGB":
            img = img.convert("RGB")
        try:
            exif = img.getexif()
            exif[COPYRIGHT_TAG] = meta_text  # ASCII-safe marker in metadata
            save_kwargs["exif"] = exif
        except Exception:
            pass
        save_kwargs.update(quality=90, optimize=True)
    elif ext == ".png":
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Copyright", meta_text)
        save_kwargs["pnginfo"] = meta
    img.save(path, **save_kwargs)


def main():
    ap = argparse.ArgumentParser(description="Stamp a copyright into gallery photos.")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="directory to scan (default: %(default)s)")
    ap.add_argument("--author", default=None, help="copyright holder (default: author from _config.yml)")
    ap.add_argument("--year", default=str(datetime.date.today().year), help="copyright year")
    ap.add_argument("--force", action="store_true", help="re-stamp images even if already done")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    author = args.author or author_from_config()
    # Use the literal © glyph in the drawn/metadata text; "(c)" is only the marker prefix.
    text = "© {} {}".format(args.year, author)
    marker_text = "(c) {} {}".format(args.year, author)

    base = os.path.join(ROOT, args.dir)
    if not os.path.isdir(base):
        if not args.quiet:
            print("watermark: nothing to do (no %s)" % args.dir)
        return 0

    try:
        with open(MANIFEST) as f:
            manifest = json.load(f)
    except Exception:
        manifest = {}

    stamped = 0
    for dirpath, _dirs, files in os.walk(base):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in EXTS:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, ROOT)
            current = file_hash(full)
            if not args.force and (manifest.get(rel) == current or has_marker(full)):
                manifest[rel] = current
                continue
            stamp(full, text, marker_text)
            manifest[rel] = file_hash(full)
            stamped += 1
            if not args.quiet:
                print("  stamped %s" % rel)

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if not args.quiet:
        print("watermark: %d image(s) stamped  (text: '%s')" % (stamped, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
