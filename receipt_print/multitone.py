from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance

# Tuned error diffusion kernel from calibration work on Epson multi-tone output.
DEFAULT_DIFFUSION_KERNEL: tuple[tuple[int, int, float], ...] = (
    (1, 0, 0.5423),
    (2, 0, 0.0533),
    (-2, 1, 0.0246),
    (-1, 1, 0.2191),
    (0, 1, 0.4715),
    (1, 1, -0.0023),
    (2, 1, -0.1241),
    (-2, 2, -0.0065),
    (-1, 2, -0.0692),
    (0, 2, 0.0168),
    (1, 2, -0.0952),
    (2, 2, -0.0304),
)

# Map calibrated grayscale palette values to Epson 4-bit multi-tone levels.
DEFAULT_LEVEL_LUT: dict[int, int] = {
    0: 15,
    9: 13,
    45: 12,
    54: 11,
    98: 10,
    107: 9,
    157: 8,
    210: 7,
    242: 6,
    251: 5,
    255: 0,
}

DEFAULT_PALETTE = np.array(sorted(DEFAULT_LEVEL_LUT.keys()), dtype=np.uint8)
DEFAULT_LEVELS = np.array(
    [DEFAULT_LEVEL_LUT[int(v)] for v in DEFAULT_PALETTE], dtype=np.uint8
)


def _dither_to_levels(
    gray: Image.Image,
    palette: np.ndarray,
    levels: np.ndarray,
    kernel: Sequence[tuple[int, int, float]],
    serpentine: bool = True,
) -> np.ndarray:
    source = np.asarray(gray.convert("L"), dtype=np.uint8)
    work = source.astype(np.float32, copy=True)
    height, width = source.shape
    last_palette_idx = len(palette) - 1
    palette_i16 = palette.astype(np.int16)
    out_idx = np.zeros((height, width), dtype=np.uint8)

    direction = 1
    for y in range(height):
        xs: Iterable[int]
        if direction > 0:
            xs = range(width)
        else:
            xs = range(width - 1, -1, -1)

        for x in xs:
            original = int(source[y, x])
            if original == 0:
                idx = 0
                out_idx[y, x] = idx
                work[y, x] = float(palette[idx])
                continue
            if original == 255:
                idx = last_palette_idx
                out_idx[y, x] = idx
                work[y, x] = float(palette[idx])
                continue

            old = float(work[y, x])
            idx = int(np.argmin(np.abs(palette_i16 - int(round(old)))))
            new = float(palette[idx])
            out_idx[y, x] = idx
            work[y, x] = new
            err = old - new

            for dx, dy, weight in kernel:
                nx = x + dx if direction > 0 else x - dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    work[ny, nx] = float(
                        min(255.0, max(0.0, work[ny, nx] + (err * float(weight))))
                    )

        if serpentine:
            direction *= -1

    return levels[out_idx]


def _planar_bitplane(
    codes: np.ndarray, width: int, width_nbytes: int, mask: int
) -> bytearray:
    flat = codes.ravel()
    bitplane = bytearray(width_nbytes * codes.shape[0])
    for index, pixel in enumerate(flat):
        if int(pixel) & mask:
            row = index // width
            col = index % width
            bitplane[row * width_nbytes + (col // 8)] |= 1 << (7 - (col % 8))
    return bitplane


def _gs_8_l_fn112_packet(
    bitplane: bytearray, width: int, height: int, color_code: int
) -> bytes:
    payload_len = 10 + len(bitplane)
    return bytes(
        [
            0x1D,
            0x38,
            0x4C,
            (payload_len >> 0) & 0xFF,
            (payload_len >> 8) & 0xFF,
            (payload_len >> 16) & 0xFF,
            (payload_len >> 24) & 0xFF,
            0x30,
            0x70,
            52,
            1,
            1,
            color_code,
            (width >> 0) & 0xFF,
            (width >> 8) & 0xFF,
            (height >> 0) & 0xFF,
            (height >> 8) & 0xFF,
        ]
    ) + bytes(bitplane)


def print_multitone_image(
    printer,
    image: Image.Image,
    *,
    num_lines: int = 100,
    sharpness: float = 1.0,
    contrast: float = 1.0,
    white_clip: int = 248,
    speed: int = 1,
    heads_energizing: int = 1,
) -> None:
    raw = getattr(printer, "_raw", None)
    if not callable(raw):
        raise RuntimeError("printer connector does not expose raw ESC/POS writes")

    gray = image.convert("L")
    if sharpness and sharpness > 0 and abs(float(sharpness) - 1.0) > 1e-3:
        gray = ImageEnhance.Sharpness(gray).enhance(sharpness)
    if contrast and contrast > 0 and abs(float(contrast) - 1.0) > 1e-3:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)

    clip_val = max(0, min(255, int(white_clip)))
    if clip_val < 255:
        # Preserve paper white by forcing near-highlight pixels to no-ink.
        arr = np.asarray(gray, dtype=np.uint8).copy()
        arr[arr >= clip_val] = 255
        gray = Image.fromarray(arr, mode="L")

    codes = _dither_to_levels(
        gray=gray,
        palette=DEFAULT_PALETTE,
        levels=DEFAULT_LEVELS,
        kernel=DEFAULT_DIFFUSION_KERNEL,
        serpentine=True,
    )

    height, width = codes.shape
    width_nbytes = (width + 7) // 8
    slice_lines = max(1, int(num_lines))

    # Epson print density controls used by multi-tone calibration scripts.
    raw(bytes([0x1D, 0x28, 0x4B, 0x02, 0x00, 0x61, int(heads_energizing) & 0xFF]))
    raw(bytes([0x1D, 0x28, 0x4B, 0x02, 0x00, 0x32, int(speed) & 0xFF]))

    for y_start in range(0, height, slice_lines):
        y_end = min(y_start + slice_lines, height)
        slice_codes = codes[y_start:y_end, :]
        slice_height = slice_codes.shape[0]

        for plane_idx, color_code in enumerate((49, 50, 51, 52)):
            mask = 0b1000 >> plane_idx
            bitplane = _planar_bitplane(slice_codes, width, width_nbytes, mask)
            raw(_gs_8_l_fn112_packet(bitplane, width, slice_height, color_code))

        # GS ( L <Function 2>: print buffered graphics data.
        raw(bytes([0x1D, 0x28, 0x4C, 0x02, 0x00, 0x30, 2]))
