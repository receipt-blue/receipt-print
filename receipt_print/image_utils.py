import csv
import io
import math
import sys
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
from escpos.exceptions import ImageWidthError
from PIL import Image, ImageEnhance, ImageOps

from .printer import DOTS_PER_LINE, MAX_LINES


def parse_caption_csv(captions_str: Optional[str]) -> List[str]:
    """Parse a caption CSV string into a list of strings."""
    if not captions_str:
        return []
    try:
        reader = csv.reader(io.StringIO(captions_str))
        captions = next(reader, [])
        return [c.strip() for c in captions]
    except Exception as e:
        sys.stderr.write(
            f"Warning: Could not parse --caption string '{captions_str}': {e}\n"
        )
        return []


# error-diffusion kernels (weights sum to 1)
ATKINSON_KERNEL = [
    (+1, 0, 1 / 8),
    (+2, 0, 1 / 8),
    (-1, 1, 1 / 8),
    (0, 1, 1 / 8),
    (+1, 1, 1 / 8),
    (0, 2, 1 / 8),
]
FS_KERNEL = [
    (+1, 0, 7 / 16),
    (-1, 1, 3 / 16),
    (0, 1, 5 / 16),
    (+1, 1, 1 / 16),
]


def desired_orientation(al: str) -> str:
    if al in {"l-top", "l-bottom", "l-center"}:
        return "landscape"
    return "portrait"


def apply_alignment(al: str, orient: str) -> tuple[str, bool]:
    al = al.lower()
    if "center" in al:
        return ("left", True)
    if orient == "landscape":
        if al == "l-top":
            return ("right", False)
        if al == "l-bottom":
            return ("left", False)
    if al == "right":
        return ("right", False)
    return ("left", False)


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    if math.isclose(gamma, 1.0, rel_tol=1e-3):
        return image
    gamma = max(gamma, 1e-3)
    inv = 1.0 / gamma
    lut = [min(255, max(0, int((i / 255.0) ** inv * 255 + 0.5))) for i in range(256)]
    bands = len(image.getbands())
    return image.point(lut * bands)


def preprocess_image(
    image: Image.Image,
    brightness: float,
    contrast: float,
    gamma: float,
    autocontrast: bool,
) -> Image.Image:
    work = image.convert("RGB")
    if autocontrast:
        work = ImageOps.autocontrast(work, cutoff=2)
    if not math.isclose(brightness, 1.0, rel_tol=1e-3):
        work = ImageEnhance.Brightness(work).enhance(brightness)
    if not math.isclose(contrast, 1.0, rel_tol=1e-3):
        work = ImageEnhance.Contrast(work).enhance(contrast)
    if not math.isclose(gamma, 1.0, rel_tol=1e-3):
        work = _apply_gamma(work, gamma)
    return work


def apply_dither(
    img: Image.Image, mode: Optional[str], thresh: float, diff: float
) -> Image.Image:
    if not mode or mode == "none":
        return img.convert("1")
    if mode == "thresh":
        g = img.convert("L")
        cut = int(thresh * 255)

        def threshold_fn(x: int) -> int:
            return 255 if x > cut else 0

        return g.convert("L").point(threshold_fn, mode="1")
    g = img.convert("L") if img.mode != "L" else img.copy()
    px = np.asarray(g, dtype=np.float32) / 255.0
    h, w = px.shape
    kernel = FS_KERNEL if mode == "floyd" else ATKINSON_KERNEL
    for y in range(h):
        serp = y % 2 == 1 and mode == "floyd"
        xs = range(w - 1, -1, -1) if serp else range(w)
        for x in xs:
            old = px[y, x]
            new = 1.0 if old >= thresh else 0.0
            err = (old - new) * diff
            px[y, x] = new
            for dx, dy, wgt in kernel:
                nx = x - dx if serp else x + dx
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    px[ny, nx] += err * wgt
    arr = (px > 0.5).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L").convert("1")


def print_images_from_pil(
    printer,
    images: Iterable[Image.Image],
    scale_list: List[float],
    align_list: List[str],
    method_list: List[str],
    ts_format: Optional[str],
    dither_list: List[Optional[str]],
    threshold_list: List[float],
    diffusion_list: List[float],
    captions_str: Optional[str] = None,
    captions_list: Optional[List[str]] = None,
    caption_start: int = 0,
    footer_text: Optional[str] = None,
    debug: bool = False,
    spacing: int = 1,
    names: Optional[Iterable[str]] = None,
    brightness_list: Optional[List[float]] = None,
    contrast_list: Optional[List[float]] = None,
    gamma_list: Optional[List[float]] = None,
    autocontrast: bool = False,
    auto_orient: bool = False,
    cut_between: bool = False,
    no_cut: bool = False,
) -> int:
    """print pre-loaded PIL images and return next caption index"""

    if captions_list is not None:
        parsed_captions_list: List[str] = captions_list[:]
    else:
        parsed_captions_list = parse_caption_csv(captions_str)

    if cut_between:
        parsed_captions_list = []

    img_list = list(images)
    name_list = (
        list(names) if names is not None else [f"{i}" for i in range(len(img_list))]
    )

    if not brightness_list:
        brightness_list = [1.0]
    if not contrast_list:
        contrast_list = [1.0]
    if not gamma_list:
        gamma_list = [1.0]

    max_width = 576
    try:
        w = printer.profile.profile_data["media"]["width"]["pixels"]
        if w != "Unknown":
            max_width = int(w)
    except Exception:
        pass

    impl_map = {
        "raster": "bitImageRaster",
        "column": "bitImageColumn",
        "graphics": "graphics",
    }

    def get(lst, i):
        return lst[i] if i < len(lst) else lst[-1]

    processed = []
    total_lines = 0
    for idx, img in enumerate(img_list):
        scale = float(get(scale_list, idx))
        al = get(align_list, idx).lower()
        orient = (
            "landscape"
            if auto_orient and img.width > img.height
            else desired_orientation(al)
        )

        im = img.copy()
        if orient == "landscape":
            im = im.rotate(270, expand=True)

        if im.width > max_width:
            ratio = max_width / im.width
            im = im.resize(
                (int(im.width * ratio), int(im.height * ratio)),
                Image.Resampling.LANCZOS,
            )

        if not math.isclose(scale, 1.0, rel_tol=1e-5):
            im = im.resize(
                (int(im.width * scale), int(im.height * scale)),
                Image.Resampling.LANCZOS,
            )

        brightness_val = float(get(brightness_list, idx))
        contrast_val = float(get(contrast_list, idx))
        gamma_val = float(get(gamma_list, idx))

        if (
            autocontrast
            or not math.isclose(brightness_val, 1.0, rel_tol=1e-3)
            or not math.isclose(contrast_val, 1.0, rel_tol=1e-3)
            or not math.isclose(gamma_val, 1.0, rel_tol=1e-3)
        ):
            im = preprocess_image(
                im, brightness_val, contrast_val, gamma_val, autocontrast
            )

        total_lines += math.ceil(im.height / DOTS_PER_LINE)
        processed.append(
            (
                im,
                al,
                idx,
                name_list[idx],
                scale,
                orient,
                brightness_val,
                contrast_val,
                gamma_val,
            )
        )

    if total_lines > MAX_LINES:
        try:
            with open("/dev/tty") as tty:
                sys.stdout.write(
                    f"Warning: {total_lines} image-lines > limit {MAX_LINES}. Continue? [y/N] "
                )
                sys.stdout.flush()
                if tty.readline().strip().lower() not in ("y", "yes"):
                    sys.exit(0)
        except Exception:
            sys.stderr.write("No TTY for confirm. Aborting.\n")
            sys.exit(1)

    for (
        im,
        al,
        idx,
        name,
        scale,
        orient,
        brightness_val,
        contrast_val,
        gamma_val,
    ) in processed:
        method = get(method_list, idx)
        impl = impl_map[method]
        dith = get(dither_list, idx)
        thresh = float(get(threshold_list, idx))
        diff = float(get(diffusion_list, idx))
        esc_align, center = apply_alignment(al, orient)

        if debug:
            printer.set(align="left")
            printer.textln(f"[DEBUG] file={name}")
            printer.textln(
                f"[DEBUG] align={al}, scale={scale:.2f}, "
                f"method={method}, dither={dith}, "
                f"threshold={thresh:.2f}, diffusion={diff:.2f}, "
                f"brightness={brightness_val:.2f}, contrast={contrast_val:.2f}, "
                f"gamma={gamma_val:.2f}, autocontrast={autocontrast}"
            )
            printer.set(align="left")

        if ts_format is not None:
            try:
                exif = im._getexif() or {}
                raw = exif.get(36867) or exif.get(306)
                if raw:
                    dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                    printer.set(align="left", font="b", bold=True)
                    printer.text(dt.strftime(ts_format) + "\n")
            except Exception:
                pass

        img2 = apply_dither(im, dith, thresh, diff)
        printer.set(align=esc_align)
        try:
            printer.image(img_source=img2, center=center, impl=impl)

            caption_idx = caption_start + idx
            caption_text = None
            if parsed_captions_list:
                if caption_idx < len(parsed_captions_list):
                    caption_text = parsed_captions_list[caption_idx]
                else:
                    caption_text = parsed_captions_list[-1]

            if caption_text:
                printer.text("\n")
                printer.set(align="center", font="b", bold=True)
                printer.text(caption_text + "\n")
                printer.set(align="left")
                if idx < len(processed) - 1:
                    printer.text("\n")

            if spacing and not cut_between:
                printer.text("\n" * spacing)

        except ImageWidthError as e:
            sys.stderr.write(f"Error printing {name}: too wide – {e}\n")
        except Exception as e:
            sys.stderr.write(f"Error printing {name}: {e}\n")

        if cut_between:
            if not no_cut:
                printer.cut()
            printer.set(align="left")

    if footer_text and not cut_between:
        printer.text("\n")
        printer.set(align="center", font="b", bold=True)
        printer.text(footer_text + "\n")
        printer.set(align="left")

    return caption_start + len(processed)


def print_images(
    printer,
    files: Iterable[str],
    scale_list: List[float],
    align_list: List[str],
    method_list: List[str],
    ts_format: Optional[str],
    dither_list: List[Optional[str]],
    threshold_list: List[float],
    diffusion_list: List[float],
    captions_str: Optional[str] = None,
    footer_text: Optional[str] = None,
    debug: bool = False,
    spacing: int = 1,
    brightness_list: Optional[List[float]] = None,
    contrast_list: Optional[List[float]] = None,
    gamma_list: Optional[List[float]] = None,
    autocontrast: bool = False,
    no_cut: bool = False,
):
    images: List[Image.Image] = []
    names: List[str] = []
    for path in files:
        try:
            img = Image.open(path)
            img.load()
            images.append(img.copy())
            names.append(path)
            img.close()
        except Exception as e:
            sys.stderr.write(f"Warning: could not open {path}: {e}\n")

    if not images:
        return

    return print_images_from_pil(
        printer,
        images,
        scale_list,
        align_list,
        method_list,
        ts_format,
        dither_list,
        threshold_list,
        diffusion_list,
        captions_str=captions_str,
        footer_text=footer_text,
        debug=debug,
        spacing=spacing,
        names=names,
        brightness_list=brightness_list,
        contrast_list=contrast_list,
        gamma_list=gamma_list,
        autocontrast=autocontrast,
        no_cut=no_cut,
    )
