#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from .printer import (
    CHAR_WIDTH,
    cat_files,
    connect_printer,
    count_lines,
    print_text,
)
from .shell import run_shell_commands


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing operations"""

    scales: List[float]
    aligns: List[str]
    methods: List[str]
    ts_fmt: Optional[str]
    dithers: List[Optional[str]]
    thresholds: List[float]
    diffusions: List[float]
    captions_str: Optional[str] = None
    footer_text: Optional[str] = None
    debug: bool = False
    spacing: int = 1


def add_image_processing_args(parser):
    """Add common image processing arguments to a parser"""
    parser.add_argument(
        "--scale", default="1.0", help="Comma-separated floats for per-image scale."
    )
    parser.add_argument(
        "--align",
        default="center",
        help="Comma-separated: left,right,center,p-center,l-top,l-bottom,l-center.",
    )
    parser.add_argument(
        "--method", default="raster", help="Comma-separated: raster,column,graphics."
    )
    parser.add_argument(
        "--timestamp",
        default="none",
        help="strftime string or 'none' to skip timestamp.",
    )
    parser.add_argument("--dither", help="Comma-separated: none,thresh,floyd,atkinson.")
    parser.add_argument(
        "--threshold",
        default="0.5",
        help="Comma-separated cutoff 0–1 for thresh/diffusion.",
    )
    parser.add_argument(
        "--diffusion",
        default="1.0",
        help="Comma-separated diffusion strength (0=no spread,1=classic).",
    )
    parser.add_argument("--heading", help="Optional heading before images.")
    parser.add_argument(
        "--caption",
        help="Comma-separated list of per-image captions.",
    )
    parser.add_argument(
        "--footer",
        help="Global footer text to print after all images.",
    )
    parser.add_argument(
        "--spacing", type=int, default=1, help="Blank lines between each image."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Emit per-image debug info."
    )


def parse_and_validate_image_args(args) -> ImageProcessingConfig:
    """Parse and validate common image processing arguments, return ImageProcessingConfig"""
    scales = [float(x) for x in args.scale.split(",")]
    aligns = [a.strip().lower() for a in args.align.split(",")]
    methods = [m.strip().lower() for m in args.method.split(",")]
    dithers = (
        [d.strip().lower() for d in args.dither.split(",")] if args.dither else [None]
    )
    thresholds = [float(x) for x in args.threshold.split(",")]
    diffusions = [float(x) for x in args.diffusion.split(",")]

    # Validation
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
    captions_str = args.caption if hasattr(args, "caption") else None
    footer_text = args.footer if hasattr(args, "footer") else None

    return ImageProcessingConfig(
        scales=scales,
        aligns=aligns,
        methods=methods,
        dithers=dithers,
        thresholds=thresholds,
        diffusions=diffusions,
        ts_fmt=ts_fmt,
        captions_str=captions_str,
        footer_text=footer_text,
        debug=args.debug,
        spacing=args.spacing,
    )


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
    img.add_argument(
        "files", nargs="*", help="Image files or directories; omit for stdin."
    )
    add_image_processing_args(img)

    # pdf
    pdf = subs.add_parser("pdf", help="Print PDF files.")
    pdf.add_argument("files", nargs="*", help="PDF files; omit for stdin")
    pdf.add_argument(
        "--format",
        choices=["image", "text"],
        default="image",
        help="Render as images or markdown text",
    )

    # Page selection - mutually exclusive
    page_group = pdf.add_mutually_exclusive_group()
    page_group.add_argument(
        "--range", help="Page range: '1,5' (pages 1-5), '5,-1' (page 5 to end)"
    )
    page_group.add_argument(
        "--pages", help="Specific pages: '1,2,5' (pages 1, 2, and 5)"
    )

    add_image_processing_args(pdf)

    # action
    action = subs.add_parser("action", help="Perform printer actions.")
    act_subs = action.add_subparsers(dest="action_command", help="Action")
    act_subs.add_parser("cut", help="Cut the paper.")

    return p


def _print_with_images(config: ImageProcessingConfig, images, names=None, heading=None):
    """Shared helper to print images with common workflow"""
    printer = connect_printer()

    if heading:
        printer.set(align="center", double_height=True, double_width=True)
        printer.text(heading + "\n")
        printer.set(align="left")

    from .image_utils import print_images_from_pil

    print_images_from_pil(
        printer,
        images,
        config.scales,
        config.aligns,
        config.methods,
        config.ts_fmt,
        config.dithers,
        config.thresholds,
        config.diffusions,
        captions_str=config.captions_str,
        footer_text=config.footer_text,
        debug=config.debug,
        spacing=config.spacing,
        names=names,
    )

    printer.cut()
    printer.close()


def _parse_page_filter(args):
    """Parse page selection arguments into filter tuple"""
    page_filter = None
    if args.range:
        try:
            parts = args.range.split(",")
            if len(parts) != 2:
                raise ValueError("Range must be two comma-separated values")
            start, end = parts
            start = int(start)
            end = int(end) if end != "-1" else -1
            if start < 1:
                raise ValueError("Start page must be >= 1")
            page_filter = ("range", start, end)
        except ValueError as e:
            sys.stderr.write(f"Invalid --range: {e}\n")
            sys.exit(1)
    elif args.pages:
        try:
            pages = [int(p) for p in args.pages.split(",")]
            if any(p < 1 for p in pages):
                raise ValueError("Page numbers must be >= 1")
            page_filter = ("pages", pages)
        except ValueError as e:
            sys.stderr.write(f"Invalid --pages: {e}\n")
            sys.exit(1)
    return page_filter


def _handle_image_based_command(args):
    """Unified handler for image and PDF commands"""
    if not args.files:
        if not sys.stdin.isatty():
            args.files = [p for p in sys.stdin.read().split() if p]
        else:
            cmd_type = "image" if args.command == "image" else "PDF"
            sys.stderr.write(f"No {cmd_type} paths provided.\n")
            sys.exit(1)

    if args.command == "image":

        def collect(paths):
            exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
            out = []
            for p in paths:
                if os.path.isdir(p):
                    for e in sorted(os.listdir(p)):
                        if e.lower().endswith(exts):
                            out.append(os.path.join(p, e))
                else:
                    if not os.path.isfile(p):
                        sys.stderr.write(f"missing file: {p}\n")
                        sys.exit(1)
                    if not p.lower().endswith(exts):
                        sys.stderr.write(f"unsupported file type: {p}\n")
                        sys.exit(1)
                    out.append(p)
            return out

        img_files = collect(args.files)
        if not img_files:
            sys.stderr.write("No usable images found.\n")
            sys.exit(1)

        from PIL import Image

        images = [Image.open(f) for f in img_files]
        names = img_files

    elif args.command == "pdf":
        page_filter = _parse_page_filter(args)

        if args.format == "text":
            from .pdf_utils import pdf_to_text

            txt = pdf_to_text(args.files, page_filter)
            print_text(txt)
            return

        from .pdf_utils import pdf_to_images

        images, names = pdf_to_images(args.files, page_filter)

        if not images:
            sys.stderr.write("No pages to print.\n")
            return

    config = parse_and_validate_image_args(args)
    _print_with_images(config, images, names=names, heading=args.heading)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
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

    if args.command in ("image", "pdf"):
        _handle_image_based_command(args)
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


if __name__ == "__main__":
    main()
