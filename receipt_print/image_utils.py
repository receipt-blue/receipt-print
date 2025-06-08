import csv
import io
import math
import sys
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
from escpos.exceptions import ImageWidthError
from PIL import Image

from .printer import DOTS_PER_LINE, MAX_LINES

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
):
    # parsed captions (per-image)
    parsed_captions_list: List[str] = []
    if captions_str:
        try:
            reader = csv.reader(io.StringIO(captions_str))
            parsed_captions_list = next(reader, [])
            parsed_captions_list = [c.strip() for c in parsed_captions_list]
        except Exception as e:
            sys.stderr.write(
                f"Warning: Could not parse --caption string '{captions_str}': {e}\n"
            )

    # determine max width
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
    for idx, path in enumerate(files):
        scale = float(get(scale_list, idx))
        al = get(align_list, idx).lower()
        orient = desired_orientation(al)

        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            sys.stderr.write(f"Warning: could not open {path}: {e}\n")
            continue

        if orient == "landscape":
            img = img.rotate(270, expand=True)

        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.Resampling.LANCZOS,
            )

        if not math.isclose(scale, 1.0, rel_tol=1e-5):
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.Resampling.LANCZOS,
            )

        total_lines += math.ceil(img.height / DOTS_PER_LINE)
        processed.append((img, al, idx, path, scale))

    # line-limit warning
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

    for img, al, idx, path, scale in processed:
        orient = desired_orientation(al)
        method = get(method_list, idx)
        impl = impl_map[method]
        dith = get(dither_list, idx)
        thresh = float(get(threshold_list, idx))
        diff = float(get(diffusion_list, idx))
        esc_align, center = apply_alignment(al, orient)

        if debug:
            printer.set(align="left")
            printer.textln(f"[DEBUG] file={path}")
            printer.textln(
                f"[DEBUG] align={al}, scale={scale:.2f}, "
                f"method={method}, dither={dith}, "
                f"threshold={thresh:.2f}, diffusion={diff:.2f}"
            )
            printer.set(align="left")

        if ts_format is not None:
            try:
                exif = img._getexif() or {}
                raw = exif.get(36867) or exif.get(306)
                if raw:
                    dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                    printer.set(align="left", font="b", bold=True)
                    printer.text(dt.strftime(ts_format) + "\n")
            except Exception:
                pass

        img2 = apply_dither(img, dith, thresh, diff)
        printer.set(align=esc_align)
        try:
            printer.image(img_source=img2, center=center, impl=impl)

            if idx < len(parsed_captions_list) and parsed_captions_list[idx]:
                printer.text("\n")
                printer.set(align="center", font="b", bold=True)
                printer.text(parsed_captions_list[idx] + "\n")
                printer.set(align="left")
                if idx < len(processed) - 1:
                    printer.text("\n")

            if spacing:
                printer.text("\n" * spacing)

        except ImageWidthError as e:
            sys.stderr.write(f"Error printing {path}: too wide – {e}\n")
        except Exception as e:
            sys.stderr.write(f"Error printing {path}: {e}\n")

    if footer_text:
        printer.text("\n")
        printer.set(align="center", font="b", bold=True)
        printer.text(footer_text + "\n")
        printer.set(align="left")
