import os
import platform
import re
import sys
from typing import List, Optional, Tuple

from escpos.exceptions import DeviceNotFoundError, USBNotFoundError
from escpos.printer import Network, Usb

# configuration
NETWORK_HOST = os.getenv("RP_HOST")
VENDOR_HEX = os.getenv("RP_VENDOR", "04b8")
PRODUCT_HEX = os.getenv("RP_PRODUCT", "0e2a")
PRINTER_PROFILE = os.getenv("RP_PROFILE", "TM-T20II")
CHAR_WIDTH = int(os.getenv("RP_CHAR_WIDTH", "42"))
CHARCODE = os.getenv("RP_CHARCODE", "CP437")
MAX_LINES = int(os.getenv("RP_MAX_LINES", "40"))
DOTS_PER_LINE = 24  # ~1 text line per 24 px
WRAP_MODES = {"hyphen", "word", "none"}
WRAP_TOKEN_RE = re.compile(r"\S+|\s+")


def remove_ansi(text: str) -> str:
    """remove common ANSI escapes"""
    ansi_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_re.sub("", text)


def sanitize_output(text: str) -> str:
    """strip ansi escapes and simplify punctuation"""
    text = remove_ansi(text)
    punctuation_map = str.maketrans(
        {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
        }
    )
    return text.translate(punctuation_map)


def count_lines(text: str, width: int) -> int:
    """count how many printed lines of width fit the text"""
    total = 0
    for ln in text.splitlines():
        if not ln:
            total += 1
        else:
            total += (len(ln) + width - 1) // width
    return total


def normalize_text_size(
    text_width: Optional[int], text_height: Optional[int]
) -> Optional[Tuple[int, int]]:
    if text_width is None and text_height is None:
        return None
    width = text_width if text_width is not None else 1
    height = text_height if text_height is not None else 1
    if not (1 <= width <= 8 and 1 <= height <= 8):
        raise ValueError("Text size multipliers must be between 1 and 8.")
    return width, height


def scaled_char_width(width: int, multiplier: int) -> int:
    return max(1, width // multiplier)


def _split_long_word(word: str, width: int, hyphenate: bool) -> List[str]:
    if width <= 0:
        return [word]
    if not hyphenate or width == 1:
        return [word[i : i + width] for i in range(0, len(word), width)]
    chunks = []
    while len(word) > width:
        chunks.append(word[: width - 1] + "-")
        word = word[width - 1 :]
    if word:
        chunks.append(word)
    return chunks


def wrap_text(text: str, width: int, mode: Optional[str]) -> str:
    mode = (mode or "hyphen").lower()
    if mode not in WRAP_MODES or mode == "none" or width <= 0:
        return text
    hyphenate = mode == "hyphen" and width > 1
    lines = text.split("\n")
    wrapped_lines: List[str] = []
    for line in lines:
        if line == "":
            wrapped_lines.append("")
            continue
        tokens = WRAP_TOKEN_RE.findall(line)
        current = ""
        for token in tokens:
            if token.isspace():
                if not current:
                    if len(token) <= width:
                        current = token
                    else:
                        for idx in range(0, len(token), width):
                            chunk = token[idx : idx + width]
                            if idx + width >= len(token):
                                current = chunk
                            else:
                                wrapped_lines.append(chunk)
                    continue
                if len(current) + len(token) <= width:
                    current += token
                else:
                    wrapped_lines.append(current.rstrip())
                    current = ""
                continue

            if len(current) + len(token) <= width:
                current += token
                continue

            if not current:
                if len(token) <= width:
                    current = token
                else:
                    chunks = _split_long_word(token, width, hyphenate)
                    wrapped_lines.extend(chunks[:-1])
                    current = chunks[-1] if chunks else ""
                continue

            if not hyphenate:
                wrapped_lines.append(current.rstrip())
                current = ""
                if len(token) <= width:
                    current = token
                else:
                    chunks = _split_long_word(token, width, hyphenate)
                    wrapped_lines.extend(chunks[:-1])
                    current = chunks[-1] if chunks else ""
                continue

            remaining = width - len(current)
            if remaining > 1:
                current += token[: remaining - 1] + "-"
                wrapped_lines.append(current)
                token = token[remaining - 1 :]
            else:
                wrapped_lines.append(current.rstrip())
            current = ""
            if token:
                chunks = _split_long_word(token, width, hyphenate)
                wrapped_lines.extend(chunks[:-1])
                current = chunks[-1] if chunks else ""
        if current:
            wrapped_lines.append(current.rstrip())
    return "\n".join(wrapped_lines)


def enforce_line_limit(n: int) -> None:
    """Ensure the line count does not exceed the maximum without confirmation."""
    if n <= MAX_LINES:
        return
    try:
        with open("/dev/tty") as tty:
            sys.stdout.write(
                f"Warning: {n} lines > limit {MAX_LINES}. Continue? [y/N] "
            )
            sys.stdout.flush()
            if tty.readline().strip().lower() not in ("y", "yes"):
                sys.exit(1)
    except Exception:
        sys.stderr.write("No TTY available; aborting.\n")
        sys.exit(1)


def env_no_cut() -> bool:
    return os.getenv("RP_NO_CUT", "0") == "1"


def should_cut(no_cut: bool = False) -> bool:
    if env_no_cut():
        return False
    return not no_cut


def maybe_cut(printer, no_cut: bool = False) -> None:
    if should_cut(no_cut):
        printer.cut()


def connect_printer():
    """connect to the ESC/POS printer"""
    skip_usb = os.getenv("RP_NO_USB", "0") == "1"
    if not skip_usb:
        try:
            if platform.system() == "Linux":
                p = Usb(
                    idVendor=int(VENDOR_HEX, 16),
                    idProduct=int(PRODUCT_HEX, 16),
                    profile=PRINTER_PROFILE,
                )
            else:
                import usb.backend.libusb1, usb.core  # noqa

                backend = usb.backend.libusb1.get_backend(
                    find_library=lambda _: "/opt/homebrew/lib/libusb-1.0.dylib"
                )
                p = Usb(profile=PRINTER_PROFILE, backend=backend)
            p.open()
            p.charcode(CHARCODE)
            return p
        except (USBNotFoundError, DeviceNotFoundError, Exception):
            sys.stderr.write("USB printer not found; falling back to network.\n")

    if not NETWORK_HOST:
        sys.stderr.write("No USB and RP_HOST unset.\n")
        sys.exit(1)

    p = Network(host=NETWORK_HOST, profile=PRINTER_PROFILE)
    p.charcode(CHARCODE)
    return p


def print_text(
    text: str,
    no_cut: bool = False,
    *,
    text_width: Optional[int] = None,
    text_height: Optional[int] = None,
    wrap_mode: str = "hyphen",
):
    """print arbitrary text with line-limit warnings"""
    text = sanitize_output(text)
    size = normalize_text_size(text_width, text_height)
    if size:
        width_mult, height_mult = size
        line_width = scaled_char_width(CHAR_WIDTH, width_mult)
        wrapped_text = wrap_text(text, line_width, wrap_mode)
        n = count_lines(wrapped_text, line_width) * height_mult
    else:
        wrapped_text = wrap_text(text, CHAR_WIDTH, wrap_mode)
        n = count_lines(wrapped_text, CHAR_WIDTH)
    enforce_line_limit(n)

    printer = connect_printer()
    if size:
        width_mult, height_mult = size
        printer.set(
            align="left",
            font="a",
            custom_size=True,
            width=width_mult,
            height=height_mult,
        )
    else:
        printer.set(align="left", font="a")
    printer.text(wrapped_text)
    if size:
        printer.set(normal_textsize=True)
    maybe_cut(printer, no_cut=no_cut)
    printer.close()


def cat_files(files: List[str], no_cut: bool = False, *, wrap_mode: str = "hyphen"):
    buf = []
    for f in files:
        try:
            with open(f) as fh:
                buf.append(fh.read())
        except Exception as e:
            sys.stderr.write(f"Error reading {f}: {e}\n")
            sys.exit(1)
    print_text("".join(buf), no_cut=no_cut, wrap_mode=wrap_mode)
