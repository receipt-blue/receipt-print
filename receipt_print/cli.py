#!/usr/bin/env python3
import argparse
import fcntl
import math
import os
import platform
import pty
import re
import select
import struct
import subprocess
import sys
import termios
from datetime import datetime

from escpos.exceptions import DeviceNotFoundError, ImageWidthError, USBNotFoundError
from escpos.printer import Network, Usb

# configuration
NETWORK_HOST = os.getenv("RP_HOST")  # e.g. "192.168.1.100", optional
VENDOR_HEX = os.getenv("RP_VENDOR", "04b8")
PRODUCT_HEX = os.getenv("RP_PRODUCT", "0e2a")
PRINTER_PROFILE = os.getenv("RP_PROFILE", "TM-T20II")
CHAR_WIDTH = int(os.getenv("RP_CHAR_WIDTH", "42"))
CHARCODE = os.getenv("RP_CHARCODE", "CP437")
MAX_LINES = int(os.getenv("RP_MAX_LINES", "40"))
DOTS_PER_LINE = 24  # 24-dot head → ~1 text line per 24 px

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


def remove_ansi(text: str) -> str:
    """
    Remove common ANSI escape sequences so they don't garble receipt output.
    """
    ansi_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_re.sub("", text)


def sanitize_output(text: str) -> str:
    """
    Strip ANSI escapes and normalize fancy punctuation to simple ASCII.
    """
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
    """
    Count how many printed lines (with wrapping) a given text will produce.
    """
    total = 0
    for ln in text.splitlines():
        if not ln:
            total += 1
        else:
            total += (len(ln) + width - 1) // width
    return total


def connect_printer():
    """
    Connect to the ESC/POS printer over USB or fallback to network.
    """
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


def print_text(text: str):
    """
    Print arbitrary text, warning if it exceeds MAX_LINES.
    """
    text = sanitize_output(text)
    n = count_lines(text, CHAR_WIDTH)
    if n > MAX_LINES:
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

    printer = connect_printer()
    printer.set(align="left", font="a")
    printer.text(text)
    printer.cut()
    printer.close()


def cat_files(files):
    buf = []
    for f in files:
        try:
            with open(f) as fh:
                buf.append(fh.read())
        except Exception as e:
            sys.stderr.write(f"Error reading {f}: {e}\n")
            sys.exit(1)
    print_text("".join(buf))


def run_command_standard(cmd: str) -> str:
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = res.stdout + (("\n" + res.stderr) if res.stderr else "")
        return sanitize_output(out).rstrip()
    except Exception as e:
        return f"Error running '{cmd}': {e}"


def run_command_in_wrapped_tty(cmd: str, columns: int) -> str:
    """
    Run `cmd` in a PTY sized to `columns`, capturing its output.
    """
    master_fd, slave_fd = pty.openpty()
    rows = 999
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        text=True,
        close_fds=True,
    )
    chunks = []
    while proc.poll() is None:
        rlist, _, _ = select.select([master_fd], [], [], 0.1)
        if master_fd in rlist:
            data = os.read(master_fd, 1024)
            if not data:
                break
            chunks.append(data.decode("utf-8", "replace"))
    while True:
        rlist, _, _ = select.select([master_fd], [], [], 0)
        if not rlist:
            break
        data = os.read(master_fd, 1024)
        if not data:
            break
        chunks.append(data.decode("utf-8", "replace"))
    os.close(master_fd)
    try:
        os.close(slave_fd)
    except OSError:
        pass
    return sanitize_output("".join(chunks)).rstrip()


def run_shell_commands(commands, wrap_tty=True, columns=80):
    pairs = []
    for cmd in commands:
        if wrap_tty:
            out = run_command_in_wrapped_tty(cmd, columns)
        else:
            out = run_command_standard(cmd)
        pairs.append(f"$ {cmd}\n{out}")
    return "\n\n\n".join(pairs)


def print_images(
    printer,
    files,
    scale_list,
    align_list,
    method_list,
    ts_format,
    dither_list,
    threshold_list,
    diffusion_list,
    debug=False,
    spacing=1,
):
    import numpy as np
    from PIL import Image

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

    def desired_orientation(al):
        if al in {"l-top", "l-bottom", "l-center"}:
            return "landscape"
        return "portrait"

    def apply_alignment(al, orient):
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
        img: Image.Image, mode: str, thresh: float, diff: float
    ) -> Image.Image:
        if not mode or mode == "none":
            return img.convert("1")
        if mode == "thresh":
            g = img.convert("L")
            cut = int(thresh * 255)
            def threshold_fn(x): return 255 if x > cut else 0
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

    # load & preprocess
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

    # print each
    for img, al, idx, path, scale in processed:
        orient = desired_orientation(al)
        method = get(method_list, idx)
        impl = impl_map[method]
        dith = get(dither_list, idx)
        thresh = float(get(threshold_list, idx))
        diff = float(get(diffusion_list, idx))
        esc_align, center = apply_alignment(al, orient)

        # debug
        if debug:
            printer.set(align="left")
            printer.textln(f"[DEBUG] file={path}")
            printer.textln(
                f"[DEBUG] align={al}, scale={scale:.2f}, "
                f"method={method}, dither={dith}, "
                f"threshold={thresh:.2f}, diffusion={diff:.2f}"
            )
            printer.set(align="left")

        # timestamp
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

        # dither & print
        img2 = apply_dither(img, dith, thresh, diff)
        printer.set(align=esc_align)
        try:
            printer.image(img_source=img2, center=center, impl=impl)
            if spacing:
                printer.text("\n" * spacing)
        except ImageWidthError as e:
            sys.stderr.write(f"Error printing {path}: too wide – {e}\n")
        except Exception as e:
            sys.stderr.write(f"Error printing {path}: {e}\n")


def create_parser():
    p = argparse.ArgumentParser(
        description="Print text or images to a receipt printer."
    )
    subs = p.add_subparsers(dest="command", help="Subcommands")

    # echo
    echo = subs.add_parser("echo", help="Print literal text.")
    echo.add_argument("text", nargs="+", help="Text to print.")
    echo.add_argument(
        "-l",
        "--lines",
        action="store_true",
        help="Join args with newlines instead of spaces.",
    )

    # cat
    cat = subs.add_parser("cat", help="Print files' contents.")
    cat.add_argument("files", nargs="+", help="Files to print.")

    # count
    cnt = subs.add_parser("count", help="Count printed lines.")
    cnt.add_argument("files", nargs="*", help="Files to count; omit for stdin.")

    # shell
    sh = subs.add_parser("shell", help="Run shell commands and print output.")
    sh.add_argument("commands", nargs="+", help="Commands to run.")
    sh.add_argument(
        "--no-wrap", action="store_true", help="Standard capture instead of PTY wrap."
    )

    # image
    img = subs.add_parser("image", help="Print one or more images.")
    img.add_argument("files", nargs="+", help="Image files or directories.")
    img.add_argument(
        "--scale", default="1.0", help="Comma-separated floats for per-image scale."
    )
    img.add_argument(
        "--align",
        default="center",
        help="Comma-separated: left,right,center,p-center,l-top,l-bottom,l-center.",
    )
    img.add_argument(
        "--method", default="raster", help="Comma-separated: raster,column,graphics."
    )
    img.add_argument(
        "--timestamp",
        default="none",
        help="strftime string or 'none' to skip timestamp.",
    )
    img.add_argument("--dither", help="Comma-separated: none,thresh,floyd,atkinson.")
    img.add_argument(
        "--threshold",
        default="0.5",
        help="Comma-separated cutoff 0–1 for thresh/diffusion.",
    )
    img.add_argument(
        "--diffusion",
        default="1.0",
        help="Comma-separated diffusion strength (0=no spread,1=classic).",
    )
    img.add_argument("--heading", help="Optional heading before images.")
    img.add_argument("--caption", help="Optional caption after images.")
    img.add_argument(
        "--spacing", type=int, default=1, help="Blank lines between each image."
    )
    img.add_argument("--debug", action="store_true", help="Emit per-image debug info.")

    # action
    action = subs.add_parser("action", help="Perform printer actions.")
    act_subs = action.add_subparsers(dest="action_command", help="Action")
    act_subs.add_parser("cut", help="Cut the paper.")

    return p


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        # print stdin if piped, else show help
        if not sys.stdin.isatty():
            print_text(sys.stdin.read())
        else:
            parser.print_help()
            sys.exit(1)
        return

    if args.command == "echo":
        txt = "\n".join(args.text) if args.lines else " ".join(args.text)
        print_text(txt)
        return

    if args.command == "cat":
        cat_files(args.files)
        return

    if args.command == "count":
        if args.files:
            buf = ""
            for f in args.files:
                try:
                    buf += open(f).read()
                except Exception as e:
                    sys.stderr.write(f"Error reading {f}: {e}\n")
                    sys.exit(1)
        else:
            if not sys.stdin.isatty():
                buf = sys.stdin.read()
            else:
                sys.stderr.write("No input provided for counting.\n")
                sys.exit(1)
        print(count_lines(buf, CHAR_WIDTH))
        return

    if args.command == "shell":
        out = run_shell_commands(
            args.commands, wrap_tty=not args.no_wrap, columns=CHAR_WIDTH
        )
        print_text(out)
        return

    if args.command == "action":
        if args.action_command == "cut":
            p = connect_printer()
            p.cut()
            p.close()
        else:
            sys.stderr.write("Invalid action command.\n")
            sys.exit(1)
        return

    # ── image sub-command ───────────────────────────────────────────────
    if args.command == "image":
        # parse lists
        scales = [float(x) for x in args.scale.split(",")]
        aligns = [a.strip().lower() for a in args.align.split(",")]
        methods = [m.strip().lower() for m in args.method.split(",")]
        dithers = (
            [d.strip().lower() for d in args.dither.split(",")]
            if args.dither
            else [None]
        )
        thresholds = [float(x) for x in args.threshold.split(",")]
        diffusions = [float(x) for x in args.diffusion.split(",")]

        # validations
        if any(m not in {"raster", "column", "graphics"} for m in methods):
            sys.stderr.write("--method must be raster|column|graphics\n")
            sys.exit(1)
        if any(d not in {None, "none", "thresh", "floyd", "atkinson"} for d in dithers):
            sys.stderr.write("--dither invalid\n")
            sys.exit(1)
        if any(not (0.0 <= t <= 1.0) for t in thresholds):
            sys.stderr.write("--threshold values must be 0–1\n")
            sys.exit(1)
        if any(d < 0 for d in diffusions):
            sys.stderr.write("--diffusion must be >= 0\n")
            sys.exit(1)

        ts_fmt = None if args.timestamp.lower() == "none" else args.timestamp

        # collect files
        def collect(paths):
            exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
            out = []
            for p in paths:
                if os.path.isdir(p):
                    for e in sorted(os.listdir(p)):
                        if e.lower().endswith(exts):
                            out.append(os.path.join(p, e))
                else:
                    out.append(p)
            return out

        img_files = collect(args.files)
        if not img_files:
            sys.stderr.write("No usable images found.\n")
            sys.exit(1)

        printer = connect_printer()

        if args.heading:
            printer.set(align="center", double_height=True, double_width=True)
            printer.text(args.heading + "\n")
            printer.set(align="left")

        print_images(
            printer,
            img_files,
            scales,
            aligns,
            methods,
            ts_fmt,
            dithers,
            thresholds,
            diffusions,
            debug=args.debug,
            spacing=args.spacing,
        )

        if args.caption:
            printer.text("\n")
            printer.set(align="center", font="b", bold=True)
            printer.text(args.caption + "\n")
            printer.set(align="left")

        printer.cut()
        printer.close()
        return
