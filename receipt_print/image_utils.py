import csv
import io
import math
import sys
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
from escpos.exceptions import ImageWidthError
from PIL import Image, ImageEnhance, ImageOps

from .multitone import print_multitone_image
from .printer import CHAR_WIDTH, DOTS_PER_LINE, MAX_LINES, wrap_text


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
EDGE_PAD_NEAR_FULL_THRESHOLD = 0.90


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


def _apply_edge_pad_compensation(
    image: Image.Image,
    max_width: int,
    left_pad: float,
    right_pad: float,
    align: str,
) -> tuple[Image.Image, bool]:
    if left_pad <= 0.0 and right_pad <= 0.0:
        return image, False
    # Preserve explicit edge alignment intent for one-sided compensation.
    if align == "left" and left_pad > 0.0 and right_pad <= 0.0:
        return image, False
    if align == "right" and right_pad > 0.0 and left_pad <= 0.0:
        return image, False
    if image.width < max_width * EDGE_PAD_NEAR_FULL_THRESHOLD:
        return image, False

    total_pad = left_pad + right_pad
    target_content_width = max(1, int(math.floor(max_width * (1.0 - total_pad))))
    if image.width > target_content_width:
        ratio = target_content_width / image.width
        image = image.resize(
            (target_content_width, int(image.height * ratio)),
            Image.Resampling.LANCZOS,
        )

    canvas = Image.new("RGB", (max_width, image.height), "white")
    left_offset = int(math.floor(max_width * left_pad))
    max_offset = max(0, max_width - image.width)
    canvas.paste(image.convert("RGB"), (min(left_offset, max_offset), 0))
    return canvas, True


def _tile_image_for_receipts(
    image: Image.Image,
    max_width: int,
    tile_count: int,
    left_pad: float,
    right_pad: float,
) -> List[Image.Image]:
    if tile_count <= 1:
        return [image]

    total_pad = left_pad + right_pad
    content_width = max(1, int(math.floor(max_width * (1.0 - total_pad))))
    target_width = max(1, content_width * tile_count)

    if image.width != target_width:
        ratio = target_width / max(1, image.width)
        target_height = max(1, int(round(image.height * ratio)))
        working = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        working = image.copy()

    use_canvas = left_pad > 0.0 or right_pad > 0.0
    left_offset = int(math.floor(max_width * left_pad))
    max_offset = max(0, max_width - content_width)
    paste_x = min(left_offset, max_offset)

    strips: List[Image.Image] = []
    for part_idx in range(tile_count):
        x0 = part_idx * content_width
        x1 = x0 + content_width
        strip = working.crop((x0, 0, x1, working.height))
        if use_canvas:
            canvas = Image.new("RGB", (max_width, strip.height), "white")
            canvas.paste(strip.convert("RGB"), (paste_x, 0))
            strips.append(canvas)
        else:
            strips.append(strip)
    return strips


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    if math.isclose(gamma, 1.0, rel_tol=1e-3):
        return image
    gamma = max(gamma, 1e-3)
    # gamma < 1 brightens, gamma > 1 darkens.
    lut = [min(255, max(0, int((i / 255.0) ** gamma * 255 + 0.5))) for i in range(256)]
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
    tile: int = 1,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
    names: Optional[Iterable[str]] = None,
    brightness_list: Optional[List[float]] = None,
    contrast_list: Optional[List[float]] = None,
    gamma_list: Optional[List[float]] = None,
    autocontrast: bool = False,
    multitone: bool = False,
    multitone_white_clip: int = 248,
    multitone_diffusion: float = 1.0,
    auto_orient: bool = False,
    cut_between: bool = False,
    no_cut: bool = False,
    wrap_mode: str = "hyphen",
) -> int:
    """print pre-loaded PIL images and return next caption index"""

    if captions_list is not None:
        parsed_captions_list: List[str] = captions_list[:]
    else:
        parsed_captions_list = parse_caption_csv(captions_str)

    tile_mode = tile > 1
    if cut_between or tile_mode:
        parsed_captions_list = []
    if tile_mode:
        footer_text = None
        spacing = 0
        cut_between = True

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

    def print_wrapped(text: str) -> None:
        wrapped = wrap_text(text, CHAR_WIDTH, wrap_mode)
        if not wrapped.endswith("\n"):
            wrapped += "\n"
        printer.text(wrapped)

    def get(lst, i):
        return lst[i] if i < len(lst) else lst[-1]

    processed = []
    total_lines = 0
    for idx, img in enumerate(img_list):
        scale = float(get(scale_list, idx))
        al = get(align_list, idx).lower()
        if tile_mode:
            scale = 1.0
            al = "left"
        orient = (
            "landscape"
            if auto_orient and img.width > img.height
            else desired_orientation(al)
        )

        im = img.copy()
        if orient == "landscape":
            im = im.rotate(270, expand=True)

        if not tile_mode and im.width > max_width:
            ratio = max_width / im.width
            im = im.resize(
                (int(im.width * ratio), int(im.height * ratio)),
                Image.Resampling.LANCZOS,
            )

        if not tile_mode and not math.isclose(scale, 1.0, rel_tol=1e-5):
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
        if tile_mode:
            strips = _tile_image_for_receipts(
                im,
                max_width=max_width,
                tile_count=tile,
                left_pad=left_pad,
                right_pad=right_pad,
            )
            edge_pad_applied = left_pad > 0.0 or right_pad > 0.0
            for part_idx, strip in enumerate(strips, start=1):
                total_lines += math.ceil(strip.height / DOTS_PER_LINE)
                processed.append(
                    (
                        strip,
                        al,
                        idx,
                        f"{name_list[idx]}#part{part_idx}",
                        scale,
                        orient,
                        brightness_val,
                        contrast_val,
                        gamma_val,
                        edge_pad_applied,
                    )
                )
        else:
            im, edge_pad_applied = _apply_edge_pad_compensation(
                im,
                max_width=max_width,
                left_pad=left_pad,
                right_pad=right_pad,
                align=al,
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
                    edge_pad_applied,
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
        edge_pad_applied,
    ) in processed:
        method = get(method_list, idx)
        impl = impl_map[method]
        dith = get(dither_list, idx)
        thresh = float(get(threshold_list, idx))
        diff = float(get(diffusion_list, idx))
        if tile_mode:
            esc_align, center = ("left", False)
        else:
            esc_align, center = apply_alignment(al, orient)

        if debug:
            printer.set(align="left")
            print_wrapped(f"[DEBUG] file={name}")
            path_label = "multitone-gs8l" if multitone else method
            print_wrapped(
                f"[DEBUG] align={al}, scale={scale:.2f}, "
                f"method={path_label}, dither={dith}, "
                f"threshold={thresh:.2f}, diffusion={diff:.2f}, "
                f"brightness={brightness_val:.2f}, contrast={contrast_val:.2f}, "
                f"gamma={gamma_val:.2f}, autocontrast={autocontrast}, "
                f"left_pad={left_pad:.3f}, right_pad={right_pad:.3f}, "
                f"edge_pad_applied={edge_pad_applied}, "
                f"final_width={im.width}, "
                f"multitone={multitone}, "
                f"multitone_white_clip={multitone_white_clip}, "
                f"multitone_diffusion={multitone_diffusion:.2f}"
            )
            printer.set(align="left")

        if ts_format is not None:
            try:
                exif = im._getexif() or {}
                raw = exif.get(36867) or exif.get(306)
                if raw:
                    dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                    printer.set(align="left", font="b", bold=True)
                    print_wrapped(dt.strftime(ts_format))
            except Exception:
                pass

        target_align = "center" if center else esc_align
        printer.set(align=target_align)
        try:
            if multitone:
                print_multitone_image(
                    printer,
                    im,
                    white_clip=int(multitone_white_clip),
                    diffusion=float(multitone_diffusion),
                )
            else:
                img2 = apply_dither(im, dith, thresh, diff)
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
                print_wrapped(caption_text)
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
        print_wrapped(footer_text)
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
    tile: int = 1,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
    brightness_list: Optional[List[float]] = None,
    contrast_list: Optional[List[float]] = None,
    gamma_list: Optional[List[float]] = None,
    autocontrast: bool = False,
    multitone: bool = False,
    multitone_white_clip: int = 248,
    multitone_diffusion: float = 1.0,
    no_cut: bool = False,
    wrap_mode: str = "hyphen",
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
        tile=tile,
        left_pad=left_pad,
        right_pad=right_pad,
        names=names,
        brightness_list=brightness_list,
        contrast_list=contrast_list,
        gamma_list=gamma_list,
        autocontrast=autocontrast,
        multitone=multitone,
        multitone_white_clip=multitone_white_clip,
        multitone_diffusion=multitone_diffusion,
        wrap_mode=wrap_mode,
        no_cut=no_cut,
    )
