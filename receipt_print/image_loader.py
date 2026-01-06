import io
from pathlib import Path
from typing import Optional

from PIL import Image

HEIF_EXTS = {".heic", ".heif"}
_HEIF_REGISTERED = False
_HEIF_AVAILABLE = False


def _ensure_heif_opener() -> bool:
    global _HEIF_REGISTERED, _HEIF_AVAILABLE
    if _HEIF_REGISTERED:
        return _HEIF_AVAILABLE
    _HEIF_REGISTERED = True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_AVAILABLE = True
    except Exception:
        _HEIF_AVAILABLE = False
    return _HEIF_AVAILABLE


def _maybe_register_heif(source: Optional[str]) -> None:
    if not source:
        return
    if source.lower().endswith(tuple(HEIF_EXTS)):
        _ensure_heif_opener()


def _convert_heif_to_png(image: Image.Image) -> Image.Image:
    fmt = (image.format or "").lower()
    if fmt not in {"heic", "heif"}:
        return image
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    converted = Image.open(buf)
    converted.load()
    return converted


def load_image_from_bytes(data: bytes, source: Optional[str] = None) -> Image.Image:
    _ensure_heif_opener()
    _maybe_register_heif(source)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return _convert_heif_to_png(img)
    except Exception as exc:
        if source and source.lower().endswith(tuple(HEIF_EXTS)) and not _HEIF_AVAILABLE:
            raise RuntimeError(
                "HEIC support unavailable; install pillow-heif to process HEIC images."
            ) from exc
        raise


def load_image_from_path(path: Path | str) -> Image.Image:
    path_obj = Path(path)
    _ensure_heif_opener()
    _maybe_register_heif(str(path_obj))
    try:
        with Image.open(path_obj) as img:
            img.load()
            loaded = img.copy()
        return _convert_heif_to_png(loaded)
    except Exception as exc:
        if path_obj.suffix.lower() in HEIF_EXTS and not _HEIF_AVAILABLE:
            raise RuntimeError(
                f"HEIC support unavailable; install pillow-heif to open {path_obj}."
            ) from exc
        raise
