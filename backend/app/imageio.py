"""Decoding uploaded images safely and consistently."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

try:  # iPhone photos. Optional: the app works without it.
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_OK = True
except Exception:  # pragma: no cover
    HEIF_OK = False

Image.MAX_IMAGE_PIXELS = 120_000_000  # allow big DSLR shots, still stop decompression bombs

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


class BadImageError(ValueError):
    pass


def decode(data: bytes, max_side: int) -> Image.Image:
    """Bytes -> upright RGB PIL image, longest side capped at max_side."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # phone photos are often stored sideways
        img = img.convert("RGB")
    except Exception as exc:
        raise BadImageError("This file is not an image we can read.") from exc
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def to_bgr(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def face_thumb(img: Image.Image, bbox, size: int = 160) -> bytes:
    """Square crop around a face with some margin, for the UI."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 0.8
    box = (int(max(0, cx - half)), int(max(0, cy - half)),
           int(min(img.width, cx + half)), int(min(img.height, cy + half)))
    crop = img.crop(box)
    crop.thumbnail((size, size), Image.LANCZOS)
    return jpeg(crop, 85)
