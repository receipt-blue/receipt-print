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


def count_lines(text, width):
    """
    Count the number of printed lines by wrapping each line at the given width.
    Empty lines count as one printed line.
    """
    lines = text.splitlines()
    total = 0
    for line in lines:
        if not line:
            total += 1
        else:
            total += (len(line) + width - 1) // width
    return total


def connect_printer():
    """
    Connect to the printer, prioritizing USB unless RP_NO_USB=1,
    otherwise fall back to network if NETWORK_HOST is set.
    """
    skip_usb = os.environ.get("RP_NO_USB", "0") == "1"

    if not skip_usb:
        try:
            if platform.system() == "Linux":
                vendor_id = int(VENDOR_HEX, 16)
                product_id = int(PRODUCT_HEX, 16)
                printer = Usb(
                    idVendor=vendor_id, idProduct=product_id, profile=PRINTER_PROFILE
                )
            else:
                import usb.backend.libusb1  # noqa: F401
                import usb.core  # noqa: F401

                backend = usb.backend.libusb1.get_backend(
                    find_library=lambda x: "/opt/homebrew/lib/libusb-1.0.dylib"
                )
                printer = Usb(profile=PRINTER_PROFILE, backend=backend)

            printer.open()
            printer.charcode(CHARCODE)
            return printer

        except (USBNotFoundError, DeviceNotFoundError, Exception):
            sys.stderr.write(
                "USB printer not found. Falling back to network printer.\n"
            )

    if not NETWORK_HOST:
        sys.stderr.write(
            "Error: No usable USB printer and NETWORK_HOST is not defined.\n"
        )
        sys.exit(1)

    printer = Network(host=NETWORK_HOST, profile=PRINTER_PROFILE)
    printer.charcode(CHARCODE)
    return printer


def print_text(text):
    """
    Print arbitrary text to the receipt, enforcing MAX_LINES limit.
    """
    text = sanitize_output(text)
    line_count = count_lines(text, CHAR_WIDTH)
    if line_count > MAX_LINES:
        prompt = (
            f"Warning: The text will resolve to {line_count} printed lines, "
            f"which exceeds the limit of {MAX_LINES}.\n"
            "Do you want to continue? [y/N] "
        )
        try:
            with open("/dev/tty", "r") as tty:
                sys.stdout.write(prompt)
                sys.stdout.flush()
                confirm = tty.readline().strip()
        except Exception:
            sys.stderr.write("No interactive input available. Aborting.\n")
            sys.exit(1)

        if confirm.lower() not in ("y", "yes"):
            sys.exit(0)

    printer = connect_printer()
    printer.set(
        align="left",
        font="a",
        bold=False,
        double_height=False,
        double_width=False,
        invert=False,
    )
    printer.text(text)
    printer.cut()
    printer.close()


def cat_files(files):
    """
    Reads and concatenates the contents of provided files, then prints them.
    """
    combined_text = ""
    for fname in files:
        try:
            with open(fname, "r") as f:
                combined_text += f.read()
        except Exception as e:
            sys.stderr.write(f"Error reading {fname}: {e}\n")
            sys.exit(1)
    print_text(combined_text)


def run_command_standard(cmd: str) -> str:
    """
    Run the command via subprocess.run and capture stdout+stderr.
    """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        combined = result.stdout
        if result.stderr:
            combined += "\n" + result.stderr
        return sanitize_output(combined).rstrip("\n")
    except Exception as e:
        return f"Error running '{cmd}': {e}"


def run_command_in_wrapped_tty(cmd: str, columns: int) -> str:
    """
    Run the command in a PTY sized to `columns`, capturing its output.
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
    bleed_list,
    debug=False,
    spacing=1,
):
    """
    Load and print one or more images with optional:
      • print method (raster/column/graphics)
      • timestamp injection (from EXIF)
      • client-side dithering with threshold/bleed controls
    """
    from datetime import datetime

    import numpy as np
    from PIL import ExifTags, Image

    # determine maximum printable width from printer profile
    max_width = 576
    try:
        pconf = printer.profile.profile_data
        w = pconf["media"]["width"]["pixels"]
        if w != "Unknown":
            max_width = int(w)
    except Exception:
        pass

    # mapping print methods to escpos impl names
    impl_map = {
        "raster": "bitImageRaster",
        "column": "bitImageColumn",
        "graphics": "graphics",
    }

    def get_val(lst, i):
        return lst[i] if i < len(lst) else lst[-1]

    def desired_orientation(al):
        if al in {"l-top", "l-bottom", "l-center"}:
            return "landscape"
        return "portrait"

    processed = []
    total_lines = 0

    for i, path in enumerate(files):
        scale = float(get_val(scale_list, i))
        al = get_val(align_list, i).lower()
        orient = desired_orientation(al)
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            sys.stderr.write(f"Warning: could not open {path}: {e}\n")
            processed.append((None, al, scale, path))
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
        processed.append((img, al, scale, path))

    if total_lines > MAX_LINES:
        prompt = (
            f"Warning: Image(s) will print approximately {total_lines} lines, "
            f"which exceeds the limit of {MAX_LINES}.\n"
            "Do you want to continue? [y/N] "
        )
        try:
            with open("/dev/tty", "r") as tty:
                sys.stdout.write(prompt)
                sys.stdout.flush()
                ans = tty.readline().strip()
        except Exception:
            sys.stderr.write("No interactive input available. Aborting.\n")
            sys.exit(1)

        if ans.lower() not in ("y", "yes"):
            sys.exit(0)

    printer.set(
        font="a", bold=False, double_height=False, double_width=False, invert=False
    )

    def apply_alignment(alignment: str, orient: str):
        alignment = alignment.lower()
        if "center" in alignment:
            return "left", True
        if orient == "landscape":
            if alignment == "l-top":
                return "right", False
            if alignment == "l-bottom":
                return "left", False
        if alignment == "right":
            return "right", False
        return "left", False

    def apply_dither(
        img: Image.Image, mode: str, thresh: float, bleed: float
    ) -> Image.Image:
        """
        Return a 1-bit Pillow image using the requested algorithm.
        """
        if not mode or mode == "none":
            return img.convert("1", dither=Image.NONE)
        if mode == "thresh":
            g = img.convert("L")
            t = int(thresh * 255)
            return g.point(lambda p: 255 if p >= t else 0, mode="1")
        if img.mode != "L":
            g = img.convert("L")
        else:
            g = img.copy()
        px = np.asarray(g, dtype=np.float32) / 255.0
        h, w = px.shape
        kernel = FS_KERNEL if mode == "floyd" else ATKINSON_KERNEL
        for y in range(h):
            serp = (y % 2 == 1) and (mode == "floyd")
            rng = range(w - 1, -1, -1) if serp else range(w)
            for x in rng:
                old = px[y, x]
                new = 1.0 if old >= thresh else 0.0
                err = (old - new) * bleed
                px[y, x] = new
                for dx, dy, wgt in kernel:
                    nx = x - dx if serp else x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        px[ny, nx] += err * wgt
        out = (px > 0.5).astype(np.uint8) * 255
        return Image.fromarray(out, mode="L").convert("1")

    for idx, (img_obj, al, scale, path) in enumerate(processed):
        if img_obj is None:
            continue

        if debug:
            printer.set(align="left", font="a", bold=False)
            printer.textln(f"[DEBUG] file={path}")
            printer.textln(
                f"[DEBUG] align={al}, "
                f"scale={scale:.2f}, "
                f"method={get_val(method_list, idx)}, "
                f"dither={get_val(dither_list, idx)}, "
                f"thresh={get_val(threshold_list, idx)}, "
                f"bleed={get_val(bleed_list, idx)}"
            )
            # After debug lines, reset to normal text style:
            printer.set(align="left", font="a", bold=False)

        orient = desired_orientation(al)
        # timestamp
        if ts_format:
            try:
                exif = img_obj._getexif() or {}
                ts_raw = exif.get(36867) or exif.get(306)
                if ts_raw:
                    dt = datetime.strptime(ts_raw, "%Y:%m:%d %H:%M:%S")
                    line = dt.strftime(ts_format)
                    printer.set(align="left", font="b", bold=True)
                    printer.text(line + "\n")
            except Exception:
                pass

        method = get_val(method_list, idx)
        impl = impl_map[method]
        dither_mode = get_val(dither_list, idx)
        thresh_v = float(get_val(threshold_list, idx))
        bleed_v = float(get_val(bleed_list, idx))

        esc_align, center_flag = apply_alignment(al, orient)
        printer.set(
            align=esc_align,
            font="a",
            bold=False,
            double_height=False,
            double_width=False,
            invert=False,
        )

        img_to_print = apply_dither(img_obj, dither_mode, thresh_v, bleed_v)

        try:
            printer.image(
                img_source=img_to_print,
                center=center_flag,
                impl=impl,
            )
            if spacing > 0:
                printer.text("\n" * spacing)
        except ImageWidthError as e:
            sys.stderr.write(f"Error printing {path}: too wide – {e}\n")
        except Exception as e:
            sys.stderr.write(f"Error printing {path}: {e}\n")


def create_parser():
    """
    Build the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Print text or images to a receipt printer."
    )
    subs = parser.add_subparsers(dest="command", help="Subcommands")

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
    cnt = subs.add_parser("count", help="Count printed lines for files or piped stdin.")
    cnt.add_argument("files", nargs="*", help="Files to count; omit to read stdin.")

    # shell
    sh = subs.add_parser("shell", help="Run shell commands and print output.")
    sh.add_argument("commands", nargs="+", help="Commands to run.")
    sh.add_argument(
        "--no-wrap",
        action="store_true",
        help="Standard subprocess capture instead of PTY wrap.",
    )

    # image
    img = subs.add_parser("image", help="Print one or more images.")
    img.add_argument(
        "files", nargs="+", help="Image files or directories containing images."
    )
    img.add_argument(
        "--scale",
        default="1.0",
        help="Comma-separated floats for per-image scale factors.",
    )
    img.add_argument(
        "--align",
        default="center",
        help="Comma-separated alignments: left, right, center, p-center, l-top, l-bottom, l-center.",
    )
    img.add_argument(
        "--method",
        default="raster",
        help="Comma-separated print methods per image: raster, column, graphics.",
    )
    img.add_argument(
        "--timestamp-format",
        default="Image: %Y-%m-%d %H:%M %Z",
        help="strftime format for image timestamp injection before each image.",
    )
    img.add_argument(
        "--dither",
        help="Comma-separated dither modes per image: none, floyd, atkinson, thresh.",
    )
    img.add_argument(
        "--threshold",
        default="0.5",
        help="Comma-separated thresholds (0–1) for `thresh` mode (default 0.5).",
    )
    img.add_argument(
        "--bleed",
        default="1.0",
        help="Comma-separated bleed factors (0–1) for error-diffusion modes (default 1).",
    )
    img.add_argument(
        "--heading",
        help="Optional heading printed before images.",
    )
    img.add_argument("--caption", help="Optional caption printed after images.")
    img.add_argument(
        "--spacing",
        type=int,
        default=1,
        help="Number of blank lines to insert between each image (default 1).",
    )
    img.add_argument(
        "--debug", action="store_true", help="Print debug info above each image."
    )

    # action
    action = subs.add_parser("action", help="Perform printer actions.")
    action_subs = action.add_subparsers(
        dest="action_command", help="Action subcommands"
    )
    action_subs.add_parser("cut", help="Cut the paper.")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        if not sys.stdin.isatty():
            print_text(sys.stdin.read())
        else:
            parser.print_help()
            sys.exit(1)

    elif args.command == "echo":
        txt = "\n".join(args.text) if args.lines else " ".join(args.text)
        print_text(txt)

    elif args.command == "cat":
        cat_files(args.files)

    elif args.command == "count":
        if args.files:
            txt = ""
            for f in args.files:
                try:
                    with open(f, "r") as fh:
                        txt += fh.read()
                except Exception as e:
                    sys.stderr.write(f"Error reading {f}: {e}\n")
                    sys.exit(1)
        else:
            if not sys.stdin.isatty():
                txt = sys.stdin.read()
            else:
                sys.stderr.write(
                    "No input provided for counting. Use files or pipe text.\n"
                )
                sys.exit(1)
        print(count_lines(txt, CHAR_WIDTH))

    elif args.command == "shell":
        wrap = not args.no_wrap
        out = run_shell_commands(args.commands, wrap, CHAR_WIDTH)
        print_text(out)

    elif args.command == "image":
        # parse scales
        parts = [p.strip() for p in args.scale.split(",")]
        try:
            scales = [float(x) for x in parts]
        except ValueError as e:
            sys.stderr.write(f"Invalid scale factor: {e}\n")
            sys.exit(1)

        # parse aligns
        aligns = [a.strip().lower() for a in args.align.split(",")]
        valid_align = {
            "left",
            "right",
            "center",
            "p-center",
            "l-top",
            "l-bottom",
            "l-center",
        }
        for a in aligns:
            if a not in valid_align:
                sys.stderr.write(
                    f"Invalid alignment '{a}'. Valid: {sorted(valid_align)}\n"
                )
                sys.exit(1)

        # parse methods
        methods = [m.strip().lower() for m in args.method.split(",")]
        valid_methods = {"raster", "column", "graphics"}
        for m in methods:
            if m not in valid_methods:
                sys.stderr.write(
                    f"Invalid method '{m}'. Valid: {sorted(valid_methods)}\n"
                )
                sys.exit(1)

        # parse dithers
        if args.dither:
            dithers = [d.strip().lower() for d in args.dither.split(",")]
            valid_dithers = {"none", "floyd", "atkinson", "thresh"}
            for d in dithers:
                if d not in valid_dithers:
                    sys.stderr.write(
                        f"Invalid dither '{d}'. Valid: {sorted(valid_dithers)}\n"
                    )
                    sys.exit(1)
        else:
            dithers = [None]

        # thresholds & bleeds
        try:
            thresholds = [float(x) for x in args.threshold.split(",")]
            bleeds = [float(x) for x in args.bleed.split(",")]
        except ValueError:
            sys.stderr.write("--threshold and --bleed must be floats between 0 and 1\n")
            sys.exit(1)
        if any(not (0.0 <= t <= 1.0) for t in thresholds):
            sys.stderr.write("--threshold values must be 0–1\n")
            sys.exit(1)
        if any(not (0.0 <= b <= 1.0) for b in bleeds):
            sys.stderr.write("--bleed values must be 0–1\n")
            sys.exit(1)

        # collect files (expand dirs)
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

        image_files = collect(args.files)
        if not image_files:
            sys.stderr.write("No usable images found.\n")
            sys.exit(1)

        # open printer once
        printer = connect_printer()

        # heading
        if args.heading:
            printer.set(
                align="center",
                font="a",
                bold=False,
                double_height=True,
                double_width=True,
            )
            printer.text(args.heading + "\n")
            printer.set(
                align="left",
                font="a",
                bold=False,
                double_height=False,
                double_width=False,
            )

        # print images
        print_images(
            printer,
            image_files,
            scales,
            aligns,
            methods,
            args.timestamp_format,
            dithers,
            thresholds,
            bleeds,
            debug=args.debug,
            spacing=args.spacing,
        )

        # caption
        if args.caption:
            printer.text("\n")
            printer.set(align="center", font="b", bold=True)
            printer.text(args.caption + "\n")
            printer.set(align="left", font="a", bold=False)

        printer.cut()
        printer.close()

    elif args.command == "action":
        if args.action_command == "cut":
            printer = connect_printer()
            printer.cut()
            printer.close()
        else:
            sys.stderr.write("Invalid action command.\n")
            sys.exit(1)

    else:
        sys.stderr.write("Invalid command.\n")
        sys.exit(1)
