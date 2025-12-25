import os
import platform
import re
import sys
from typing import List

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


def print_text(text: str, no_cut: bool = False):
    """print arbitrary text with line-limit warnings"""
    text = sanitize_output(text)
    n = count_lines(text, CHAR_WIDTH)
    enforce_line_limit(n)

    printer = connect_printer()
    printer.set(align="left", font="a")
    printer.text(text)
    maybe_cut(printer, no_cut=no_cut)
    printer.close()


def cat_files(files: List[str], no_cut: bool = False):
    buf = []
    for f in files:
        try:
            with open(f) as fh:
                buf.append(fh.read())
        except Exception as e:
            sys.stderr.write(f"Error reading {f}: {e}\n")
            sys.exit(1)
    print_text("".join(buf), no_cut=no_cut)
