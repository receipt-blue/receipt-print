import glob
import os
import platform
import re
import sys
import time
from typing import List, Optional, Tuple

from escpos.exceptions import DeviceNotFoundError, USBNotFoundError
from escpos.printer import File, Network, Usb

# configuration
NETWORK_HOST = os.getenv("RP_HOST")
DEVICE_PATH = os.getenv("RP_DEVICE")
VENDOR_HEX = os.getenv("RP_VENDOR", "04b8")
PRODUCT_HEX = os.getenv("RP_PRODUCT")
PRINTER_PROFILE = os.getenv("RP_PROFILE", "TM-T20II")
CHAR_WIDTH = int(os.getenv("RP_CHAR_WIDTH", "42"))
CHARCODE = os.getenv("RP_CHARCODE", "CP437")
MAX_LINES = int(os.getenv("RP_MAX_LINES", "40"))
DOTS_PER_LINE = 24  # ~1 text line per 24 px
WRAP_MODES = {"hyphen", "word", "none"}
WRAP_TOKEN_RE = re.compile(r"\S+|\s+")
SPEED_OVERRIDE_ENV = "RP_SPEED_OVERRIDE"
RAW_CHUNK_BYTES = 16 * 1024
DIRECT_CONNECT_ATTEMPTS = 3
DIRECT_CONNECT_RETRY_SECONDS = 0.25
_line_limit_confirmed = False


class PrinterUnavailableError(RuntimeError):
    pass


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


def enforce_line_limit(n: int, unit: str = "lines") -> None:
    """Ensure the line count does not exceed the maximum without confirmation."""
    global _line_limit_confirmed
    if n <= MAX_LINES or _line_limit_confirmed:
        return
    try:
        with open("/dev/tty") as tty:
            sys.stdout.write(
                f"Warning: {n} {unit} > limit {MAX_LINES}. "
                "Continue this print job without further size warnings? [y/N] "
            )
            sys.stdout.flush()
            if tty.readline().strip().lower() not in ("y", "yes"):
                sys.exit(1)
            _line_limit_confirmed = True
    except Exception:
        sys.stderr.write("No TTY available; aborting.\n")
        sys.exit(1)


def reset_line_limit_confirmation() -> None:
    global _line_limit_confirmed
    _line_limit_confirmed = False


def env_no_cut() -> bool:
    return os.getenv("RP_NO_CUT", "0") == "1"


def should_cut(no_cut: bool = False) -> bool:
    if env_no_cut():
        return False
    return not no_cut


def cut_paper(printer, partial_cut: bool = False) -> None:
    if partial_cut:
        printer.cut(mode="PART")
    else:
        printer.cut()


def maybe_cut(
    printer, no_cut: bool = False, partial_cut: bool = False
) -> None:
    if should_cut(no_cut):
        cut_paper(printer, partial_cut=partial_cut)


def _resolve_speed() -> Optional[int]:
    value = os.getenv(SPEED_OVERRIDE_ENV)
    if value in (None, ""):
        value = os.getenv("RP_SPEED")
    if value in (None, ""):
        return None
    try:
        speed = int(value)
    except ValueError:
        sys.stderr.write(f"Invalid speed value '{value}' for {SPEED_OVERRIDE_ENV}/RP_SPEED.\n")
        return None
    if not 0 <= speed <= 255:
        sys.stderr.write("Speed must be between 0 and 255.\n")
        return None
    return speed


def _apply_speed(printer, speed: Optional[int]) -> None:
    if speed is None:
        return
    try:
        raw = getattr(printer, "_raw", None)
        if callable(raw):
            raw(bytes([0x1D, 0x28, 0x4B, 0x02, 0x00, 0x32, speed & 0xFF]))
    except Exception as exc:
        sys.stderr.write(f"Warning: could not apply printer speed {speed}: {exc}\n")


def _permission_denied(exc: Exception) -> bool:
    message = str(exc)
    return "Access denied" in message or "Errno 13" in message or "Permission denied" in message


def _configured_device_path() -> Optional[str]:
    if DEVICE_PATH in (None, "", "auto", "*"):
        return None
    return DEVICE_PATH


def _device_auto_discover_enabled() -> bool:
    return os.getenv("RP_DEVICE_AUTO_DISCOVER", "1") != "0"


def _device_candidates() -> List[str]:
    configured = os.getenv("RP_DEVICE_CANDIDATES", "")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    paths = glob.glob("/dev/receipt-printer*") + glob.glob("/dev/usb/lp*")
    return sorted(dict.fromkeys(paths))


def _open_device(path: str, speed: Optional[int]):
    p = File(path, profile=PRINTER_PROFILE)
    p.charcode(CHARCODE)
    _apply_speed(p, speed)
    return p


def _connect_auto_device(speed: Optional[int]):
    if not _device_auto_discover_enabled():
        return None
    last_error: Exception | None = None
    for path in _device_candidates():
        try:
            printer = _open_device(path, speed)
            sys.stderr.write(f"Using kernel USB printer device {path}.\n")
            return printer
        except Exception as exc:
            last_error = exc
    if last_error is not None and _permission_denied(last_error):
        sys.stderr.write(f"USB printer device permission denied: {last_error}\n")
    return None


def _connect_explicit_device(path: str, speed: Optional[int]):
    try:
        return _open_device(path, speed)
    except Exception as exc:
        if _permission_denied(exc):
            sys.stderr.write(f"USB printer device permission denied: {exc}\n")
            sys.stderr.write("On Linux, load usblp and make sure your user can write the printer device.\n")
        else:
            sys.stderr.write(f"USB printer device failed: {exc}\n")
        sys.exit(1)


def _hex_id(value: str, name: str) -> int:
    try:
        return int(value, 16)
    except ValueError:
        sys.stderr.write(f"Invalid {name} value '{value}'; expected a hexadecimal USB ID.\n")
        sys.exit(1)


def _configured_product_id() -> Optional[int]:
    if PRODUCT_HEX in (None, "", "auto", "*"):
        return None
    return _hex_id(PRODUCT_HEX, "RP_PRODUCT")


def _open_usb(
    vendor_id: int,
    product_id: Optional[int],
    speed: Optional[int],
    *,
    backend=None,
):
    kwargs = {
        "idVendor": vendor_id,
        "profile": PRINTER_PROFILE,
        "timeout": 30000,
    }
    if product_id is not None:
        kwargs["idProduct"] = product_id
    if backend is not None:
        kwargs["usb_args"] = {"backend": backend}
    p = Usb(**kwargs)
    p.open()
    p.charcode(CHARCODE)
    _apply_speed(p, speed)
    return p


def _discovered_product_ids(vendor_id: int) -> List[int]:
    try:
        import usb.core
    except Exception:
        return []

    product_ids: List[int] = []
    try:
        devices = usb.core.find(find_all=True, idVendor=vendor_id)
        for device in devices:
            product_id = int(device.idProduct)
            if product_id not in product_ids:
                product_ids.append(product_id)
    except Exception:
        return []
    return product_ids


def printer_device_available() -> bool:
    device_path = _configured_device_path()
    if device_path:
        return os.path.exists(device_path)

    if platform.system() == "Linux" and any(
        os.path.exists(path) for path in _device_candidates()
    ):
        return True

    if os.getenv("RP_NO_USB", "0") != "1":
        try:
            import usb.backend.libusb1
            import usb.core

            backend = usb.backend.libusb1.get_backend()
            if backend is not None:
                vendor_id = int(VENDOR_HEX, 16)
                product_id = _configured_product_id()
                usb_args = {"idVendor": vendor_id, "backend": backend}
                if product_id is not None:
                    usb_args["idProduct"] = product_id
                if usb.core.find(**usb_args) is not None:
                    return True
        except (SystemExit, Exception):
            pass

    return bool(NETWORK_HOST)


def _connect_linux_usb(speed: Optional[int]):
    vendor_id = _hex_id(VENDOR_HEX, "RP_VENDOR")
    product_id = _configured_product_id()
    auto_discover = os.getenv("RP_USB_AUTO_DISCOVER", "1") != "0"
    attempts: List[Optional[int]] = []
    if product_id is not None:
        attempts.append(product_id)
    if auto_discover:
        for discovered_product_id in _discovered_product_ids(vendor_id):
            if discovered_product_id not in attempts:
                attempts.append(discovered_product_id)
    if product_id is None and None not in attempts:
        attempts.append(None)

    last_error: Exception | None = None
    for candidate in attempts:
        try:
            printer = _open_usb(vendor_id, candidate, speed)
            if product_id is not None and candidate != product_id:
                sys.stderr.write(
                    f"Using auto-discovered USB printer {vendor_id:04x}:{candidate:04x}.\n"
                )
            return printer
        except (USBNotFoundError, DeviceNotFoundError, Exception) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise USBNotFoundError("No USB product candidates configured")


def connect_direct_printer():
    """Connect directly to the ESC/POS printer."""
    speed = _resolve_speed()
    device_path = _configured_device_path()
    if device_path:
        return _connect_explicit_device(device_path, speed)

    if platform.system() == "Linux":
        printer = _connect_auto_device(speed)
        if printer is not None:
            return printer

    skip_usb = os.getenv("RP_NO_USB", "0") == "1"
    usb_permission_denied = False
    if not skip_usb:
        try:
            if platform.system() == "Linux":
                return _connect_linux_usb(speed)
            else:
                import usb.backend.libusb1, usb.core  # noqa

                backend = usb.backend.libusb1.get_backend()
                if backend is None:
                    raise RuntimeError("libusb backend is unavailable")
                p = _open_usb(
                    _hex_id(VENDOR_HEX, "RP_VENDOR"),
                    _configured_product_id(),
                    speed,
                    backend=backend,
                )
            return p
        except (SystemExit, Exception) as exc:
            message = str(exc)
            if _permission_denied(exc):
                usb_permission_denied = True
                sys.stderr.write(f"USB printer permission denied: {message}\n")
                sys.stderr.write("On Linux, install a udev rule or add your user to the printer group.\n")
            else:
                sys.stderr.write("USB printer not found; falling back to network.\n")

    if not NETWORK_HOST:
        if usb_permission_denied:
            sys.stderr.write("USB printer matched but could not be opened, and RP_HOST is unset.\n")
        else:
            sys.stderr.write("No USB and RP_HOST unset.\n")
        raise PrinterUnavailableError("USB receipt printer is unavailable and RP_HOST is unset")

    p = Network(host=NETWORK_HOST, profile=PRINTER_PROFILE)
    p.charcode(CHARCODE)
    _apply_speed(p, speed)
    return p


def _connect_locked_direct_printer(lock=None):
    from receipt_print.routing import DeviceLock, LockedPrinter

    device_lock = lock or DeviceLock()
    if lock is None:
        device_lock.acquire()
    try:
        return LockedPrinter(connect_direct_printer(), device_lock)
    except BaseException:
        device_lock.release()
        raise


def connect_printer():
    """Use the local print service when ready, otherwise take a direct lease."""
    from receipt_print.routing import (
        DeviceLock,
        ServicePrinter,
        print_mode,
        service_ready,
        service_url,
    )

    mode = print_mode()
    url = service_url()
    if mode != "direct" and service_ready(url):
        return ServicePrinter(
            profile=PRINTER_PROFILE,
            charcode=CHARCODE,
            speed=_resolve_speed(),
            apply_speed=_apply_speed,
            url=url,
        )
    if mode == "service":
        raise RuntimeError(f"receipt-print service is not ready at {url}")

    lock = DeviceLock()
    lock.acquire()
    if mode == "auto" and service_ready(url):
        lock.release()
        return ServicePrinter(
            profile=PRINTER_PROFILE,
            charcode=CHARCODE,
            speed=_resolve_speed(),
            apply_speed=_apply_speed,
            url=url,
        )
    return _connect_locked_direct_printer(lock)


def print_raw_bytes_direct(data: bytes, cut: bool = False) -> None:
    printer = None
    for attempt in range(DIRECT_CONNECT_ATTEMPTS):
        try:
            printer = _connect_locked_direct_printer()
            break
        except PrinterUnavailableError:
            if attempt + 1 == DIRECT_CONNECT_ATTEMPTS:
                raise
            time.sleep(DIRECT_CONNECT_RETRY_SECONDS)
    if printer is None:
        raise PrinterUnavailableError("USB receipt printer is unavailable")
    try:
        raw = getattr(printer, "_raw", None)
        if not callable(raw):
            raise RuntimeError("printer backend does not expose raw byte output")
        for offset in range(0, len(data), RAW_CHUNK_BYTES):
            raw(data[offset : offset + RAW_CHUNK_BYTES])
        if cut:
            maybe_cut(printer)
    finally:
        printer.close()


def print_raw_bytes(data: bytes, cut: bool = False) -> None:
    printer = connect_printer()
    try:
        raw = getattr(printer, "_raw", None)
        if not callable(raw):
            raise RuntimeError("printer backend does not expose raw byte output")
        raw(data)
        if cut:
            maybe_cut(printer)
    finally:
        close = getattr(printer, "close", None)
        if callable(close):
            close()


def print_text(
    text: str,
    no_cut: bool = False,
    *,
    partial_cut: bool = False,
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
    if should_cut(no_cut):
        maybe_cut(printer, partial_cut=partial_cut)
    elif not wrapped_text.endswith("\n"):
        printer.text("\n")
    printer.close()


def cat_files(
    files: List[str],
    no_cut: bool = False,
    *,
    partial_cut: bool = False,
    wrap_mode: str = "hyphen",
):
    buf = []
    for f in files:
        try:
            with open(f) as fh:
                buf.append(fh.read())
        except Exception as e:
            sys.stderr.write(f"Error reading {f}: {e}\n")
            sys.exit(1)
    print_text(
        "".join(buf), no_cut=no_cut, partial_cut=partial_cut, wrap_mode=wrap_mode
    )
