#!/usr/bin/env python3
import argparse
import os
import platform
import sys

from escpos.exceptions import DeviceNotFoundError, USBNotFoundError
from escpos.printer import Network, Usb

# configuration
VENDOR_HEX = os.getenv("RP_VENDOR", "04b8")
PRODUCT_HEX = os.getenv("RP_PRODUCT", "0e2a")
PRINTER_PROFILE = os.getenv("RP_PROFILE", "TM-T20II")
PRINTER_HOST = os.getenv("RP_HOST")  # e.g. "192.168.1.100"
CHAR_WIDTH = int(os.getenv("RP_CHAR_WIDTH", "72"))
MAX_LINES = int(os.getenv("RP_MAX_LINES", "40"))


def count_lines(text, width):
    """
    Count the number of printed lines by wrapping each line at the given width.
    Empty lines count as one printed line.
    """
    lines = text.splitlines()
    total = 0
    for line in lines:
        if len(line) == 0:
            total += 1
        else:
            total += (len(line) + width - 1) // width
    return total


def connect_printer():
    """
    Connect to the printer via USB. If that fails and PRINTER_HOST is set,
    fall back to a network connection.
    """
    try:
        if platform.system() == "Linux":
            vendor_id = int(VENDOR_HEX, 16)
            product_id = int(PRODUCT_HEX, 16)
            printer = Usb(
                idVendor=vendor_id, idProduct=product_id, profile=PRINTER_PROFILE
            )
        else:
            from escpos import usb

            backend = usb.backend.libusb1.get_backend(
                find_library=lambda x: "/opt/homebrew/lib/libusb-1.0.dylib"
            )
            printer = Usb(profile=PRINTER_PROFILE, backend=backend)
        printer.open()
        return printer
    except (USBNotFoundError, DeviceNotFoundError) as e:
        if PRINTER_HOST:
            sys.stderr.write(
                f"USB printer not found: {e}. Falling back to network printer at {PRINTER_HOST}.\n"
            )
            printer = Network(host=PRINTER_HOST, profile=PRINTER_PROFILE)
            return printer
        else:
            sys.stderr.write(f"Error: USB printer not found: {e}\n")
            sys.exit(1)


def print_text(text):
    """
    Connects to the printer, applies a basic text style, prints the text,
    cuts the receipt, and then closes the connection.

    If the text will print more than MAX_LINES, ask the user to confirm.
    """
    line_count = count_lines(text, CHAR_WIDTH)

    if line_count > MAX_LINES:
        prompt = (
            f"Warning: The text will resolve to {line_count} printed lines, which exceeds the limit of {MAX_LINES}.\n"
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


def create_parser():
    parser = argparse.ArgumentParser(
        description="Print text to a receipt printer."
        " If no subcommand is provided, piped input will be printed directly."
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

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # if no subcommand is provided, check for piped input
    if args.command is None:
        if not sys.stdin.isatty():
            piped_text = sys.stdin.read()
            print_text(piped_text)
        else:
            parser.print_help()
            sys.exit(1)
    elif args.command == "echo":
        if args.lines:
            text = "\n".join(args.text)
        else:
            text = " ".join(args.text)
        print_text(text)
    elif args.command == "cat":
        cat_files(args.files)
    elif args.command == "count":
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
    else:
        sys.stderr.write("Invalid command.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
