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

from escpos.exceptions import (
    DeviceNotFoundError,
    ImageWidthError,
    USBNotFoundError,
)
from escpos.printer import Network, Usb

# configuration
NETWORK_HOST = os.getenv("RP_HOST")  # e.g. "192.168.1.100", optional
VENDOR_HEX = os.getenv("RP_VENDOR", "04b8")
PRODUCT_HEX = os.getenv("RP_PRODUCT", "0e2a")
PRINTER_PROFILE = os.getenv("RP_PROFILE", "TM-T20II")
CHAR_WIDTH = int(os.getenv("RP_CHAR_WIDTH", "42"))
CHARCODE = os.getenv("RP_CHARCODE", "CP437")
MAX_LINES = int(os.getenv("RP_MAX_LINES", "40"))
DOTS_PER_LINE = 24  # for estimating how many lines of text a printed image occupies


def remove_ansi(text: str) -> str:
    """
    Remove common ANSI escape sequences so they don't garble receipt output.
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def sanitize_output(text: str) -> str:
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
    Connect to the printer, prioritizing USB unless the environment variable
    RP_NO_USB is set to 1, in which case skip USB entirely
    and connect via the network if NETWORK_HOST is defined.
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
    Connects to the printer, applies a basic text style, prints the text,
    cuts the receipt, and then closes the connection.
    If the text will print more than MAX_LINES, ask the user to confirm.
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
    Reads and concatenates the contents of provided files, then sends them to the printer.
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
    Run the command via subprocess.run with shell=True and capture both stdout & stderr.
    Does not force text wrapping, so the output is whatever the local environment
    or command chooses to do.
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
    Run the command inside a pseudo-terminal that has `columns` set,
    so programs that check TTY size will wrap output at `columns`.
    Return the entire captured (stdout+stderr) text as a string.
    """
    master_fd, slave_fd = pty.openpty()

    # set the PTY window size: (rows, columns, xpix, ypix)
    rows = 999  # arbitrary large number of rows
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

    output_chunks = []
    while True:
        # check if process has ended
        if proc.poll() is not None:
            # drain any remaining output
            while True:
                rlist, _, _ = select.select([master_fd], [], [], 0)
                if not rlist:
                    break
                try:
                    data = os.read(master_fd, 1024)
                    if not data:
                        break
                    output_chunks.append(data.decode("utf-8", "replace"))
                except OSError:
                    break
            break

        # read available output
        rlist, _, _ = select.select([master_fd], [], [], 0.1)
        if master_fd in rlist:
            try:
                data = os.read(master_fd, 1024)
                if not data:
                    break
                output_chunks.append(data.decode("utf-8", "replace"))
            except OSError:
                break

    # clean up
    os.close(master_fd)
    try:
        os.close(slave_fd)
    except OSError:
        pass

    final_output = "".join(output_chunks)
    return sanitize_output(final_output).rstrip("\n")


def run_shell_commands(commands, wrap_tty=True, columns=80):
    """
    Runs each command and captures its output in a format like:
        $ command
        [command output]
        $ command2
        [command2 output]
    When wrap_tty is True (the default), each command is run in a PTY with `columns` set.
    Otherwise, a normal subprocess capture is used.
    """
    pairs = []
    for cmd in commands:
        if wrap_tty:
            cmd_output = run_command_in_wrapped_tty(cmd, columns)
        else:
            cmd_output = run_command_standard(cmd)
        pairs.append(f"$ {cmd}\n{cmd_output}")
    return "\n\n\n".join(pairs)


def print_images(printer, files, scale_list, align_list, debug=False):
    """
    Load and print one or more images with optional scaling and alignment/orientation.
    """
    from PIL import Image

    resample = Image.Resampling.LANCZOS

    # determine maximum printable width from printer profile
    max_width = 576
    try:
        pconf = printer.profile.profile_data
        w = pconf["media"]["width"]["pixels"]
        if w != "Unknown":
            max_width = int(w)
    except Exception:
        pass

    # "extend" either scale or alignment lists if they have fewer entries than files
    def get_val(lst, i):
        if i < len(lst):
            return lst[i]
        return lst[-1]

    expanded_aligns = [get_val(align_list, i).lower() for i in range(len(files))]
    expanded_scales = [float(get_val(scale_list, i)) for i in range(len(files))]

    def desired_orientation(align_val):
        if align_val in {"l-top", "l-bottom", "l-center"}:
            return "landscape"
        return "portrait"

    force_unified = len(set(expanded_aligns)) == 1
    batch_orientation = (
        desired_orientation(expanded_aligns[0]) if force_unified else None
    )

    total_estimated_lines = 0
    processed_images = []  # will store (image_obj, alignment_used, scale_used, file_path)

    for i, file_path in enumerate(files):
        user_scale = expanded_scales[i]
        user_align = expanded_aligns[i]
        if batch_orientation is not None:
            user_align = expanded_aligns[0]
        orientation = (
            batch_orientation
            if batch_orientation is not None
            else desired_orientation(user_align)
        )

        try:
            img = Image.open(file_path)
            img.load()
        except Exception as e:
            sys.stderr.write(f"Warning: could not open {file_path}: {e}\n")
            processed_images.append((None, user_align, user_scale, file_path))
            continue

        # rotate if landscape
        if orientation == "landscape":
            img = img.rotate(270, expand=True)

        # force-fit image to printer's max width if necessary
        if img.width > max_width:
            force_ratio = max_width / img.width
            new_w = int(img.width * force_ratio)
            new_h = int(img.height * force_ratio)
            img = img.resize((new_w, new_h), resample)

        # apply user scale relative to forced size
        if not math.isclose(user_scale, 1.0, rel_tol=1e-5):
            scaled_w = int(img.width * user_scale)
            scaled_h = int(img.height * user_scale)
            img = img.resize((scaled_w, scaled_h), resample)

        lines_estimate = math.ceil(img.height / DOTS_PER_LINE)
        total_estimated_lines += lines_estimate

        processed_images.append((img, user_align, user_scale, file_path))

    if total_estimated_lines > MAX_LINES:
        prompt = (
            f"Warning: Image(s) will print approximately {total_estimated_lines} lines, "
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

    printer.set(
        font="a", bold=False, double_height=False, double_width=False, invert=False
    )

    def apply_alignment(alignment: str, orient: str):
        """
        Return (escpos_align, center_flag) based on alignment and orientation.
        """
        alignment = alignment.lower()
        if "center" in alignment:
            return ("left", True)
        if orient == "landscape":
            if alignment == "l-top":
                return ("right", False)
            if alignment == "l-bottom":
                return ("left", False)
        if alignment == "right":
            return ("right", False)
        return ("left", False)

    for img_obj, align_val, scale_val, path in processed_images:
        if img_obj is None:
            continue

        this_orient = (
            batch_orientation
            if batch_orientation is not None
            else desired_orientation(align_val)
        )

        if debug:
            printer.set(align="left")
            debug_text = f"[DEBUG] file={path}\n[DEBUG] align={align_val}, scale={scale_val:.2f}\n"
            printer.textln(debug_text)

        escpos_align, pixel_center = apply_alignment(align_val, this_orient)
        printer.set(align=escpos_align)

        try:
            printer.image(img_source=img_obj, center=pixel_center)
        except ImageWidthError as e:
            sys.stderr.write(f"Error printing {path}: image is too wide - {e}\n")
        except Exception as e:
            sys.stderr.write(f"Error printing {path}: {e}\n")


def create_parser():
    """
    Create the argument parser with subcommands for text, file, count, shell, and image printing.
    """
    parser = argparse.ArgumentParser(
        description="Print text to a receipt printer. "
        "If no subcommand is provided, piped input is read and printed directly."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # echo
    echo_parser = subparsers.add_parser(
        "echo", help="Print direct text passed as arguments."
    )
    echo_parser.add_argument("text", nargs="+", help="Text to print.")
    echo_parser.add_argument(
        "-l",
        "--lines",
        action="store_true",
        help="Join the input arguments with newlines instead of spaces.",
    )

    # cat
    cat_parser = subparsers.add_parser("cat", help="Print the contents of file(s).")
    cat_parser.add_argument("files", nargs="+", help="File(s) to print")

    # count
    count_parser = subparsers.add_parser(
        "count",
        help="Calculate the number of lines that would be printed if running 'cat' on a given input.",
    )
    count_parser.add_argument(
        "files",
        nargs="*",
        help="File(s) to count lines from. If omitted, piped input is used.",
    )

    # shell
    shell_parser = subparsers.add_parser(
        "shell", help="Run shell commands and print their output."
    )
    shell_parser.add_argument(
        "commands",
        nargs="+",
        help="One or more commands to run, e.g. 'ls -l' 'uname -a'",
    )
    shell_parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="Disable PTY wrapping; run commands using standard subprocess capture.",
    )

    # image
    image_parser = subparsers.add_parser("image", help="Print one or more images.")
    image_parser.add_argument("files", nargs="+", help="Image file(s) to print.")
    image_parser.add_argument(
        "--scale",
        default="1.0",
        help="One or more comma-separated scale factors (floats). Default 1.0.",
    )
    image_parser.add_argument(
        "--align",
        default="center",
        help="One or more comma-separated alignments. Valid: left, right, center, p-center, l-top, l-bottom, l-center. Default 'center' (where orientation is determined automatically).",
    )
    image_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info (including alignment, scale) above each image.",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        # read from stdin if no subcommand is specified
        if not sys.stdin.isatty():
            piped_text = sys.stdin.read()
            print_text(piped_text)
        else:
            parser.print_help()
            sys.exit(1)

    elif args.command == "echo":
        text = "\n".join(args.text) if args.lines else " ".join(args.text)
        print_text(text)

    elif args.command == "cat":
        cat_files(args.files)

    elif args.command == "count":
        # if files are provided, read them; otherwise, read from stdin
        if args.files:
            combined_text = ""
            for fname in args.files:
                try:
                    with open(fname, "r") as f:
                        combined_text += f.read()
                except Exception as e:
                    sys.stderr.write(f"Error reading {fname}: {e}\n")
                    sys.exit(1)
        else:
            if not sys.stdin.isatty():
                combined_text = sys.stdin.read()
            else:
                sys.stderr.write(
                    "No input provided for counting. Use files or pipe text.\n"
                )
                sys.exit(1)
        print(count_lines(combined_text, CHAR_WIDTH))

    elif args.command == "shell":
        wrap_tty = not args.no_wrap
        output_text = run_shell_commands(args.commands, wrap_tty, CHAR_WIDTH)
        print_text(output_text)

    elif args.command == "image":
        scale_parts = [part.strip() for part in args.scale.split(",")]
        try:
            scale_list = [float(x) for x in scale_parts]
        except ValueError as e:
            sys.stderr.write(f"Invalid scale factor: {e}\n")
            sys.exit(1)

        # parse alignments
        align_parts = [a.strip().lower() for a in args.align.split(",")]
        valid_aligns = {
            "left",
            "right",
            "center",
            "p-center",
            "l-top",
            "l-bottom",
            "l-center",
        }
        for a in align_parts:
            if a not in valid_aligns:
                sys.stderr.write(
                    f"Invalid alignment '{a}'. Valid: {sorted(valid_aligns)}\n"
                )
                sys.exit(1)

        printer = connect_printer()
        print_images(printer, args.files, scale_list, align_parts, debug=args.debug)
        printer.cut()
        printer.close()

    else:
        sys.stderr.write("Invalid command.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
