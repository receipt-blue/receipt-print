#!/usr/bin/env python3
import os
import sys
import io
import shlex
from dataclasses import dataclass
from typing import List, Optional
from tempfile import NamedTemporaryFile

import click
from PIL import Image

from .printer import CHAR_WIDTH, cat_files, connect_printer, count_lines, print_text
from .shell import run_shell_commands
from .wiki import WikipediaError, load_article, print_articles


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


def parse_comma_separated(value, converter, validator=None):
    """Helper to parse comma-separated values with optional validation"""
    if not value:
        return []
    items = [converter(x.strip()) for x in value.split(",")]
    if validator:
        for item in items:
            if not validator(item):
                raise click.BadParameter(f"Invalid value: {item}")
    return items


def validate_method(value):
    return value.lower() in {"raster", "column", "graphics"}


def validate_dither(value):
    return value is None or value.lower() in {"none", "thresh", "floyd", "atkinson"}


def validate_threshold(value):
    return 0.0 <= value <= 1.0


def validate_diffusion(value):
    return value >= 0.0


# Common image processing options
image_options = [
    click.option(
        "--scale", default="1.0", help="Comma-separated floats for per-image scale."
    ),
    click.option(
        "--align",
        default="center",
        help="Comma-separated: left,right,center,p-center,l-top,l-bottom,l-center.",
    ),
    click.option(
        "--method", default="raster", help="Comma-separated: raster,column,graphics."
    ),
    click.option(
        "--timestamp",
        default="none",
        help="strftime string or 'none' to skip timestamp.",
    ),
    click.option("--dither", help="Comma-separated: none,thresh,floyd,atkinson."),
    click.option(
        "--threshold",
        default="0.5",
        help="Comma-separated cutoff 0–1 for thresh/diffusion.",
    ),
    click.option(
        "--diffusion",
        default="1.0",
        help="Comma-separated diffusion strength (0=no spread,1=classic).",
    ),
    click.option("--heading", help="Optional heading before images."),
    click.option("--caption", help="Comma-separated list of per-image captions."),
    click.option("--footer", help="Global footer text to print after all images."),
    click.option(
        "--spacing", type=int, default=1, help="Blank lines between each image."
    ),
    click.option("--debug", is_flag=True, help="Emit per-image debug info."),
]


def add_image_options(func):
    """Decorator to add all image processing options to a command"""
    for option in reversed(image_options):
        func = option(func)
    return func


def create_image_config(**kwargs) -> ImageProcessingConfig:
    """Create ImageProcessingConfig from click arguments"""
    scales = parse_comma_separated(kwargs["scale"], float)
    aligns = parse_comma_separated(kwargs["align"], str)
    methods = parse_comma_separated(kwargs["method"], str)
    dithers = (
        parse_comma_separated(kwargs["dither"], str) if kwargs["dither"] else [None]
    )
    thresholds = parse_comma_separated(kwargs["threshold"], float)
    diffusions = parse_comma_separated(kwargs["diffusion"], float)

    # Validation
    if not all(validate_method(m) for m in methods):
        raise click.BadParameter("--method must be raster|column|graphics")
    if not all(validate_dither(d) for d in dithers):
        raise click.BadParameter("--dither invalid")
    if not all(validate_threshold(t) for t in thresholds):
        raise click.BadParameter("--threshold values must be 0–1")
    if not all(validate_diffusion(d) for d in diffusions):
        raise click.BadParameter("--diffusion must be >= 0")

    ts_fmt = None if kwargs["timestamp"].lower() == "none" else kwargs["timestamp"]

    return ImageProcessingConfig(
        scales=scales,
        aligns=[a.lower() for a in aligns],
        methods=[m.lower() for m in methods],
        dithers=[d.lower() if d and d != "none" else None for d in dithers],
        thresholds=thresholds,
        diffusions=diffusions,
        ts_fmt=ts_fmt,
        captions_str=kwargs["caption"],
        footer_text=kwargs["footer"],
        debug=kwargs["debug"],
        spacing=kwargs["spacing"],
    )


def print_with_images(config: ImageProcessingConfig, images, names=None, heading=None):
    """Print images with common workflow"""
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


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Print text or images to a receipt printer."""
    if ctx.invoked_subcommand is None:
        if not sys.stdin.isatty():
            print_text(sys.stdin.read())
        else:
            click.echo(ctx.get_help())
            sys.exit(1)


@cli.command()
@click.argument("text", nargs=-1, required=True)
@click.option(
    "-l", "--lines", is_flag=True, help="Join args with newlines instead of spaces."
)
def text(text, lines):
    """Print literal text."""
    txt = "\n".join(text) if lines else " ".join(text)
    print_text(txt)


@cli.command()
@click.argument("urls", nargs=-1, required=True)
@click.option("--qr/--no-qr", default=True, help="Include article QR codes.")
@click.option(
    "--qr-position",
    type=click.Choice(["top-left", "top-right", "bottom-left", "bottom-right"], case_sensitive=False),
    default="bottom-left",
    show_default=True,
    help="Placement for article QR codes.",
)
def wiki(urls, qr, qr_position):
    """Print one or more Wikipedia articles."""

    articles = []
    for raw in urls:
        try:
            articles.append(load_article(raw))
        except WikipediaError as exc:
            click.echo(f"Failed to load '{raw}': {exc}", err=True)
            sys.exit(1)
    print_articles(articles, qr=qr, qr_position=qr_position.lower())


@cli.command()
@click.argument("files", nargs=-1, required=True)
def cat(files):
    """Print files' contents."""
    cat_files(files)


@cli.command()
@click.argument("files", nargs=-1)
def count(files):
    """Count printed lines."""
    if files:
        buf = ""
        for f in files:
            try:
                buf += open(f).read()
            except Exception as e:
                click.echo(f"Error reading {f}: {e}", err=True)
                sys.exit(1)
    else:
        if not sys.stdin.isatty():
            buf = sys.stdin.read()
        else:
            click.echo("No input provided for counting.", err=True)
            sys.exit(1)
    click.echo(count_lines(buf, CHAR_WIDTH))


@cli.command()
@click.argument("commands", nargs=-1, required=True)
@click.option("--no-wrap", is_flag=True, help="Standard capture instead of PTY wrap.")
def shell(commands, no_wrap):
    """Run shell commands and print output."""
    out = run_shell_commands(commands, wrap_tty=not no_wrap, columns=CHAR_WIDTH)
    print_text(out)


@cli.command()
@click.argument("files", nargs=-1)
@add_image_options
def image(files, **kwargs):
    """Print one or more images."""
    img_bytes = None
    if not files:
        if not sys.stdin.isatty():
            data = sys.stdin.buffer.read()
            try:
                txt = data.decode().strip()
            except Exception:
                txt = ""
            parts = shlex.split(txt)
            if parts and all(os.path.exists(os.path.expanduser(p)) for p in parts):
                files = [os.path.expanduser(p) for p in parts]
            else:
                img_bytes = data if data else None
        else:
            click.echo("No image paths provided.", err=True)
            sys.exit(1)

    if img_bytes is not None:
        try:
            images = [Image.open(io.BytesIO(img_bytes))]
            names = ["stdin"]
        except Exception as e:
            click.echo(f"Invalid image data: {e}", err=True)
            sys.exit(1)
    else:
        exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
        img_files = []
        for p in files:
            if os.path.isdir(p):
                for e in sorted(os.listdir(p)):
                    if e.lower().endswith(exts):
                        img_files.append(os.path.join(p, e))
            else:
                if not os.path.isfile(p):
                    click.echo(f"missing file: {p}", err=True)
                    sys.exit(1)
                if not p.lower().endswith(exts):
                    click.echo(f"unsupported file type: {p}", err=True)
                    sys.exit(1)
                img_files.append(p)

        if not img_files:
            click.echo("No usable images found.", err=True)
            sys.exit(1)

        images = [Image.open(f) for f in img_files]
        names = img_files

    config = create_image_config(**kwargs)
    print_with_images(config, images, names=names, heading=kwargs["heading"])


@cli.command()
@click.argument("files", nargs=-1)
@click.option(
    "--format",
    type=click.Choice(["image", "text"]),
    default="image",
    help="Render as images or markdown text",
)
@click.option("--range", help="Page range: '1,5' (pages 1-5), '5,-1' (page 5 to end)")
@click.option("--pages", help="Specific pages: '1,2,5' (pages 1, 2, and 5)")
@add_image_options
def pdf(files, format, range, pages, **kwargs):
    """Print PDF files."""
    pdf_bytes = None
    if not files:
        if not sys.stdin.isatty():
            data = sys.stdin.buffer.read()
            try:
                txt = data.decode().strip()
            except Exception:
                txt = ""
            parts = shlex.split(txt)
            if parts and all(os.path.exists(os.path.expanduser(p)) for p in parts):
                files = [os.path.expanduser(p) for p in parts]
            else:
                pdf_bytes = data if data else None
        else:
            click.echo("No PDF paths provided.", err=True)
            sys.exit(1)

    # Parse page filter
    page_filter = None
    if range and pages:
        click.echo("Cannot specify both --range and --pages", err=True)
        sys.exit(1)

    if range:
        try:
            parts = range.split(",")
            if len(parts) != 2:
                raise ValueError("Range must be two comma-separated values")
            start, end = parts
            start = int(start)
            end = int(end) if end != "-1" else -1
            if start < 1:
                raise ValueError("Start page must be >= 1")
            page_filter = ("range", start, end)
        except ValueError as e:
            click.echo(f"Invalid --range: {e}", err=True)
            sys.exit(1)
    elif pages:
        try:
            page_list = [int(p) for p in pages.split(",")]
            if any(p < 1 for p in page_list):
                raise ValueError("Page numbers must be >= 1")
            page_filter = ("pages", page_list)
        except ValueError as e:
            click.echo(f"Invalid --pages: {e}", err=True)
            sys.exit(1)

    if format == "text":
        from .pdf_utils import pdf_to_text

        if pdf_bytes is not None:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            try:
                txt = pdf_to_text([tmp_path], page_filter)
            finally:
                os.unlink(tmp_path)
        else:
            txt = pdf_to_text(files, page_filter)
        print_text(txt)
        return

    from .pdf_utils import pdf_to_images

    if pdf_bytes is not None:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            images, names = pdf_to_images([tmp_path], page_filter)
        finally:
            os.unlink(tmp_path)
    else:
        images, names = pdf_to_images(files, page_filter)

    if not images:
        click.echo("No pages to print.", err=True)
        return

    config = create_image_config(**kwargs)
    print_with_images(config, images, names=names, heading=kwargs["heading"])


@cli.group()
def action():
    """Perform printer actions."""
    pass


@action.command()
def cut():
    """Cut the paper."""
    p = connect_printer()
    p.cut()
    p.close()


def main():
    cli()


if __name__ == "__main__":
    main()
