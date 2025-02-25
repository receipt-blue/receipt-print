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
PRINTER_HOST = os.getenv(
    "RP_HOST"
)  # if USB connection fails, fall back to a network printer, e.g. "192.168.1.100"


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
    """
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
    parser = argparse.ArgumentParser(description="Print text to a receipt printer.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    # 'cat' subcommand for printing file contents
    cat_parser = subparsers.add_parser("cat", help="Print the contents of file(s)")
    cat_parser.add_argument("files", nargs="+", help="File(s) to print")
    return parser


def process_args(args):
    if args.command == "cat":
        cat_files(args.files)
    else:
        # should not get here normally
        sys.stderr.write("Invalid command.\n")
        sys.exit(1)


def main():
    # if there's piped input *and* no extra args, use piped input
    if not sys.stdin.isatty() and len(sys.argv) == 1:
        piped_text = sys.stdin.read()
        print_text(piped_text)
        return

    # if help is requested, let argparse show it
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        parser = create_parser()
        parser.parse_args()  # this will print help and exit
        return

    # if the first argument is "cat", then we invoke the subcommand parser
    if sys.argv[1] == "cat":
        parser = create_parser()
        args = parser.parse_args()
        process_args(args)
    else:
        # assume all arguments are text to print
        text = "\n".join(sys.argv[1:])
        print_text(text)


if __name__ == "__main__":
    main()
