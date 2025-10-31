#!/usr/bin/env python3
import io
import os
import random
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from tempfile import NamedTemporaryFile

import click
from PIL import Image
from escpos.escpos import QR_ECLEVEL_H, QR_ECLEVEL_L, QR_ECLEVEL_M, QR_ECLEVEL_Q

from .arena import (
    ArenaClient,
    ArenaDownloadError,
    ArenaError,
    ArenaNotFound,
    ArenaPrintJob,
    ArenaUnauthorized,
    ChannelIterator,
    block_attachment,
    block_class,
    block_connected_at,
    block_description,
    block_preview_urls,
    block_text_content,
    block_title,
    block_user_name,
    canonical_block_url,
    canonical_channel_url,
    load_image_from_bytes,
    parse_block_identifier,
    parse_channel_identifier,
    parse_timestamp,
    pdf_bytes_to_images,
    format_timestamp,
    should_include_block,
    video_frame_from_bytes,
)
from .printer import CHAR_WIDTH, cat_files, connect_printer, count_lines, print_text, sanitize_output
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


@dataclass
class MediaHandlingOptions:
    """Options for handling attachments/media from Are.na blocks."""

    video_mode: str
    ffmpeg_path: Optional[str]
    pdf_mode: str
    pdf_page_filter: Optional[Tuple]


@dataclass
class QRConfig:
    """Configuration for optional QR code printing."""

    enabled: bool
    size: int
    ec: int


QR_LEVELS = {"L": QR_ECLEVEL_L, "M": QR_ECLEVEL_M, "Q": QR_ECLEVEL_Q, "H": QR_ECLEVEL_H}
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WS_CLEANER = re.compile(r"\s{2,}")


def collapse_spaces(value: str) -> str:
    return WS_CLEANER.sub(" ", value).strip()


def clean_heading_text(value: str) -> str:
    return collapse_spaces(
        sanitize_output(value or "").encode("ascii", "ignore").decode("ascii")
    )


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


def create_arena_image_config(
    method: str,
    dither: Optional[str],
    threshold: str,
    diffusion: str,
    caption: Optional[str],
    footer: Optional[str],
    spacing: int,
    debug: bool,
) -> ImageProcessingConfig:
    """Build an ImageProcessingConfig for Are.na printing with defaults."""
    kwargs = {
        "scale": "1.0",
        "align": "center",
        "method": method,
        "timestamp": "none",
        "dither": dither or "",
        "threshold": threshold,
        "diffusion": diffusion,
        "caption": caption,
        "footer": footer,
        "debug": debug,
        "spacing": spacing,
        "heading": None,
    }
    return create_image_config(**kwargs)


def parse_pdf_options(
    pdf_mode: str, pdf_pages: Optional[str], pdf_range: Optional[str]
) -> Tuple[str, Optional[Tuple]]:
    """Parse PDF handling options into mode + page filter."""
    pdf_mode = (pdf_mode or "first").lower()
    if pdf_mode not in {"first", "all"}:
        raise click.BadParameter("--pdf must be 'first' or 'all'")
    if pdf_pages and pdf_range:
        raise click.BadParameter("Cannot use both --pdf-pages and --pdf-range")

    page_filter: Optional[Tuple] = None
    if pdf_range:
        try:
            parts = [p.strip() for p in pdf_range.split(",")]
            if len(parts) != 2:
                raise ValueError
            start = int(parts[0])
            if start < 1:
                raise ValueError
            end_str = parts[1]
            end = int(end_str) if end_str != "-1" else -1
            page_filter = ("range", start, end)
        except ValueError as exc:
            raise click.BadParameter(f"Invalid --pdf-range: {pdf_range}") from exc
    elif pdf_pages:
        try:
            pages = [int(p.strip()) for p in pdf_pages.split(",") if p.strip()]
            if not pages or any(p < 1 for p in pages):
                raise ValueError
            page_filter = ("pages", pages)
        except ValueError as exc:
            raise click.BadParameter(f"Invalid --pdf-pages: {pdf_pages}") from exc

    return pdf_mode, page_filter


def resolve_ffmpeg_path(video_mode: str, explicit: Optional[str]) -> Tuple[Optional[str], bool]:
    """Resolve ffmpeg binary path for video frame extraction."""
    if video_mode != "frame":
        return None, False

    candidates = []
    if explicit:
        candidates.append(explicit)
    env_ffmpeg = os.getenv("FFMPEG")
    if env_ffmpeg:
        candidates.append(env_ffmpeg)
    auto = shutil.which("ffmpeg")
    if auto:
        candidates.append(auto)

    for cand in candidates:
        if not cand:
            continue
        expanded = os.path.expanduser(cand)
        resolved = shutil.which(expanded) if not os.path.isabs(expanded) else expanded
        if resolved and os.path.exists(resolved):
            return resolved, False
        if os.path.exists(expanded):
            return expanded, False
    return None, True


def build_media_options(
    video_mode: str,
    ffmpeg_option: Optional[str],
    pdf_mode: str,
    pdf_pages: Optional[str],
    pdf_range: Optional[str],
) -> Tuple[MediaHandlingOptions, bool]:
    pdf_mode_resolved, page_filter = parse_pdf_options(pdf_mode, pdf_pages, pdf_range)
    ffmpeg_path, missing_ffmpeg = resolve_ffmpeg_path(video_mode, ffmpeg_option)
    opts = MediaHandlingOptions(
        video_mode=video_mode,
        ffmpeg_path=ffmpeg_path,
        pdf_mode=pdf_mode_resolved,
        pdf_page_filter=page_filter,
    )
    return opts, missing_ffmpeg


def parse_list_option(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_since_option(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = parse_timestamp(value)
    if not dt:
        raise click.BadParameter(f"Invalid ISO timestamp: {value}")
    return dt


def gather_images_for_block(
    block: Dict[str, Any], client: ArenaClient, media_opts: MediaHandlingOptions
) -> Tuple[List[Image.Image], List[str]]:
    """Download images for a block based on its class/attachment."""

    images: List[Image.Image] = []
    names: List[str] = []
    attachment = block_attachment(block)
    preview_urls = block_preview_urls(block)

    def fetch_preview() -> Tuple[Optional[Image.Image], Optional[str]]:
        for url in preview_urls:
            try:
                data = client.download_media(url)
                img = load_image_from_bytes(data)
                return img, url
            except ArenaDownloadError as exc:
                sys.stderr.write(f"Warning: Could not download preview {url}: {exc}\n")
        return None, None

    if attachment:
        url = attachment.get("url")
        content_type = (attachment.get("content_type") or "").lower()
        if url and content_type.startswith("image/"):
            try:
                data = client.download_media(url)
                images.append(load_image_from_bytes(data))
                names.append(url)
                return images, names
            except ArenaDownloadError as exc:
                sys.stderr.write(f"Warning: Could not load image attachment {url}: {exc}\n")
        elif url and content_type.startswith("video/"):
            if media_opts.video_mode == "frame":
                if media_opts.ffmpeg_path:
                    try:
                        data = client.download_media(url)
                        frame = video_frame_from_bytes(data, media_opts.ffmpeg_path)
                        images.append(frame)
                        names.append(url)
                        return images, names
                    except ArenaDownloadError as exc:
                        sys.stderr.write(
                            f"Warning: Failed video frame extraction for {url}: {exc}\n"
                        )
                else:
                    sys.stderr.write(
                        "Warning: ffmpeg unavailable; using preview image for video block.\n"
                    )
            # fallback to preview if available
        elif url and content_type == "application/pdf":
            try:
                data = client.download_media(url)
                pdf_images = pdf_bytes_to_images(data, media_opts.pdf_page_filter)
                if not pdf_images:
                    sys.stderr.write(f"Warning: PDF attachment {url} yielded no pages.\n")
                else:
                    if (
                        media_opts.pdf_mode == "first"
                        and media_opts.pdf_page_filter is None
                    ):
                        pdf_images = pdf_images[:1]
                    images.extend(pdf_images)
                    names.extend([f"{url}#page{i+1}" for i in range(len(pdf_images))])
                    return images, names
            except ArenaDownloadError as exc:
                sys.stderr.write(f"Warning: Could not download PDF {url}: {exc}\n")
        # other attachment types fall back to preview/text

    klass = block_class(block)
    if (preview_urls and not images) or klass in {"image", "link", "media", "attachment", "channel"}:
        preview_img, used_url = fetch_preview()
        if preview_img:
            images.append(preview_img)
            names.append(used_url or "preview")

    return images, names


def build_metadata_lines(
    block: Dict[str, Any], added_override: Optional[datetime] = None
) -> List[str]:
    lines: List[str] = []
    author = block_user_name(block)
    if author:
        lines.append(f"By: {author}")
    added_dt = added_override or parse_timestamp(block.get("created_at"))
    added_str = format_timestamp(added_dt)
    if added_str:
        lines.append(f"Added: {added_str}")
    modified_str = format_timestamp(parse_timestamp(block.get("updated_at")))
    if modified_str:
        lines.append(f"Modified: {modified_str}")
    return [collapse_spaces(line.encode("ascii", "ignore").decode("ascii")) for line in lines if line]


def print_arena_block(
    job: ArenaPrintJob,
    block: Dict[str, Any],
    block_id: str,
    client: ArenaClient,
    media_opts: MediaHandlingOptions,
    qr_cfg: QRConfig,
    clean: bool,
    added_override: Optional[datetime] = None,
) -> bool:
    """Print a single block following the Are.na spec."""

    content_printed = False
    title = block_title(block)
    title_clean = collapse_spaces(title.encode("ascii", "ignore").decode("ascii"))

    klass = block_class(block)
    text_content = block_text_content(block) if klass == "text" else None

    images, image_names = gather_images_for_block(block, client, media_opts)

    if klass == "text" and text_content:
        text_content = collapse_spaces(LINK_PATTERN.sub(lambda m: m.group(1), text_content))
        job.print_text(text_content, align="left", font="a")
        content_printed = True
    elif images:
        job.print_images(images, names=image_names)
        content_printed = True
    elif text_content:
        text_content = collapse_spaces(LINK_PATTERN.sub(lambda m: m.group(1), text_content))
        job.print_text(text_content, align="left", font="a")
        content_printed = True

    if not content_printed:
        preview_urls = block_preview_urls(block)
        fallback = preview_urls[0] if preview_urls else canonical_block_url(block_id)
        fallback = collapse_spaces(fallback.encode("ascii", "ignore").decode("ascii"))
        job.print_text(fallback, align="left", font="a")
        content_printed = True

    if not clean:
        if title_clean:
            job.print_text(title_clean, align="left", font="b", bold=True)

        description = block_description(block)
        if description:
            desc_clean = collapse_spaces(
                LINK_PATTERN.sub(
                    lambda m: m.group(1), description.encode("ascii", "ignore").decode("ascii")
                )
            )
            job.print_text(desc_clean, align="left", font="a", bold=False)
        metadata_lines = build_metadata_lines(block, added_override)
        for meta in metadata_lines:
            job.print_text(collapse_spaces(meta), align="right", font="a")

    if qr_cfg.enabled:
        url = canonical_block_url(block_id)
        job.printer.set(align="right")
        try:
            job.printer.qr(url, size=qr_cfg.size, ec=qr_cfg.ec)
        except Exception as exc:
            sys.stderr.write(f"Warning: Failed to print QR for {url}: {exc}\n")

    job.line_break(1 if qr_cfg.enabled else 2)
    return content_printed


def compute_channel_heading(meta: Optional[Dict[str, Any]]) -> List[str]:
    if not meta:
        return []
    title = clean_heading_text(meta.get("title") or "Untitled Channel")
    user = meta.get("user") or {}
    user_name = clean_heading_text(
        user.get("full_name") or user.get("username") or "Unknown"
    )
    combined = f"{user_name} / {title}".strip()
    if len(combined) <= HEADING_CHAR_WIDTH:
        return [combined]
    return [f"{user_name} /", title]


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


@cli.group(name="are.na")
def arena_group():
    """Interact with Are.na blocks and channels."""
    pass


@arena_group.command("block")
@click.argument("blocks", nargs=-1, required=True)
@click.option("--method", default="raster", help="Comma-separated: raster,column,graphics.")
@click.option("--dither", help="Comma-separated: none,thresh,floyd,atkinson.")
@click.option("--threshold", default="0.5", help="Comma-separated cutoff 0–1 for thresh/diffusion.")
@click.option("--diffusion", default="1.0", help="Comma-separated diffusion strength (repeat-last).")
@click.option("--spacing", type=int, default=1, help="Blank lines between images.")
@click.option("--heading", help="Optional heading printed once before blocks.")
@click.option("--caption", help="Comma-separated list of per-image captions.")
@click.option("--footer", help="Footer text printed after all output.")
@click.option("--clean", is_flag=True, help="Print content only (no title/description/metadata).")
@click.option(
    "--video",
    type=click.Choice(["frame", "preview"]),
    default="frame",
    help="Video attachments: 'frame' extracts first frame (requires ffmpeg), 'preview' uses preview image.",
)
@click.option("--ffmpeg", help="Explicit ffmpeg path for --video=frame.")
@click.option(
    "--pdf",
    type=click.Choice(["first", "all"]),
    default="first",
    help="PDF attachments: print first page or all pages.",
)
@click.option("--pdf-pages", help="Specific PDF pages (1-indexed), e.g. '1,3,5'.")
@click.option("--pdf-range", help="PDF page range 'start,end'; use -1 for end.")
@click.option("--qr", is_flag=True, help="Append a QR code with the block URL.")
@click.option(
    "--qr-size",
    type=int,
    default=4,
    help="QR module size (printer dependent). Default: 4.",
)
@click.option(
    "--qr-correction",
    type=click.Choice(["L", "M", "Q", "H"]),
    default="M",
    help="QR error correction level.",
)
@click.option("--no-cache", is_flag=True, help="Bypass local Are.na cache for this run.")
@click.option("--debug", is_flag=True, help="Emit per-image debug info during printing.")
def arena_block(
    blocks,
    method,
    dither,
    threshold,
    diffusion,
    spacing,
    heading,
    caption,
    footer,
    clean,
    video,
    ffmpeg,
    pdf,
    pdf_pages,
    pdf_range,
    qr,
    qr_size,
    qr_correction,
    no_cache,
    debug,
):
    """Print one or more Are.na blocks."""
    media_opts, missing_ffmpeg = build_media_options(video, ffmpeg, pdf, pdf_pages, pdf_range)
    qr_cfg = QRConfig(enabled=qr, size=qr_size, ec=QR_LEVELS.get(qr_correction, QR_ECLEVEL_M))
    config = create_arena_image_config(method, dither, threshold, diffusion, caption, footer, spacing, debug)

    client = ArenaClient(cache_enabled=not no_cache)
    printer = connect_printer()
    job = ArenaPrintJob(
        printer,
        config.scales,
        config.aligns,
        config.methods,
        config.ts_fmt,
        config.dithers,
        config.thresholds,
        config.diffusions,
        config.captions_str,
        config.spacing,
        footer,
        debug=config.debug,
    )

    if missing_ffmpeg and video == "frame":
        sys.stderr.write("Warning: ffmpeg not found; using preview images for videos.\n")

    printed_any = False
    heading_printed = False

    try:
        for ident in blocks:
            try:
                block_id = parse_block_identifier(ident)
            except ArenaError as exc:
                sys.stderr.write(f"Error: {exc}\n")
                continue

            try:
                block = client.fetch_block(block_id)
            except ArenaNotFound:
                if not os.getenv("ARENA_TOKEN"):
                    sys.stderr.write(
                        f"Error: Unauthorized for {ident} (set ARENA_TOKEN?).\n"
                    )
                else:
                    sys.stderr.write(f"Error: Not found: {ident}\n")
                continue
            except ArenaUnauthorized:
                sys.stderr.write(
                    f"Error: Unauthorized for {ident} (set ARENA_TOKEN?).\n"
                )
                continue
            except ArenaError as exc:
                sys.stderr.write(f"Error: {exc}\n")
                continue

            if heading and not heading_printed:
                job.print_heading(heading, trailing_blank=True)
                heading_printed = True


            success = print_arena_block(
                job,
                block,
                str(block.get("id", block_id)),
                client,
                media_opts,
                qr_cfg,
                clean,
            )
            if success:
                printed_any = True

        if printed_any:
            job.print_footer()
            printer.cut()
    finally:
        printer.close()
        client.close()

    if not printed_any:
        sys.exit(1)


@arena_group.command("channel")
@click.argument("channel", required=True)
@click.option("--method", default="raster", help="Comma-separated: raster,column,graphics.")
@click.option("--dither", help="Comma-separated: none,thresh,floyd,atkinson.")
@click.option("--threshold", default="0.5", help="Comma-separated cutoff 0–1 for thresh/diffusion.")
@click.option("--diffusion", default="1.0", help="Comma-separated diffusion strength (repeat-last).")
@click.option("--spacing", type=int, default=1, help="Blank lines between images.")
@click.option("--heading", help="Override heading text for the channel.")
@click.option("--caption", help="Comma-separated list of per-image captions.")
@click.option("--footer", help="Footer text printed after all output.")
@click.option("--clean", is_flag=True, help="Print content only (no title/description/metadata).")
@click.option(
    "--video",
    type=click.Choice(["frame", "preview"]),
    default="frame",
    help="Video attachments: 'frame' extracts first frame (requires ffmpeg), 'preview' uses preview image.",
)
@click.option("--ffmpeg", help="Explicit ffmpeg path for --video=frame.")
@click.option(
    "--pdf",
    type=click.Choice(["first", "all"]),
    default="first",
    help="PDF attachments: print first page or all pages.",
)
@click.option("--pdf-pages", help="Specific PDF pages (1-indexed), e.g. '1,3,5'.")
@click.option("--pdf-range", help="PDF page range 'start,end'; use -1 for end.")
@click.option("--filter", help="Comma-separated block classes to include (text,image,link,media,attachment,channel).")
@click.option("--exclude", help="Comma-separated block classes to skip.")
@click.option("--limit", type=int, help="Stop after printing N blocks.")
@click.option("--since", help="Only include blocks with connected_at >= ISO timestamp.")
@click.option("--include-channels", is_flag=True, help="Include nested channels within contents.")
@click.option("--qr", is_flag=True, help="Print a QR code of the channel URL beneath the heading.")
@click.option(
    "--qr-size",
    type=int,
    default=4,
    help="QR module size (printer dependent). Default: 4.",
)
@click.option(
    "--qr-correction",
    type=click.Choice(["L", "M", "Q", "H"]),
    default="M",
    help="QR error correction level.",
)
@click.option(
    "--sort",
    type=click.Choice(["asc", "desc", "random"]),
    default="desc",
    help="Ordering for channel contents: ascending, descending, or random (default desc).",
)
@click.option("--no-cache", is_flag=True, help="Bypass local Are.na cache for this run.")
@click.option("--debug", is_flag=True, help="Emit per-image debug info during printing.")
def arena_channel(
    channel,
    method,
    dither,
    threshold,
    diffusion,
    spacing,
    heading,
    caption,
    footer,
    clean,
    video,
    ffmpeg,
    pdf,
    pdf_pages,
    pdf_range,
    filter,
    exclude,
    limit,
    since,
    include_channels,
    qr,
    qr_size,
    qr_correction,
    sort,
    no_cache,
    debug,
):
    """Print all blocks in an Are.na channel."""
    try:
        ref = parse_channel_identifier(channel)
    except ArenaError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    media_opts, missing_ffmpeg = build_media_options(video, ffmpeg, pdf, pdf_pages, pdf_range)
    config = create_arena_image_config(method, dither, threshold, diffusion, caption, footer, spacing, debug)
    qr_cfg = QRConfig(enabled=qr, size=qr_size, ec=QR_LEVELS.get(qr_correction, QR_ECLEVEL_M))
    block_qr_cfg = QRConfig(enabled=False, size=qr_size, ec=qr_cfg.ec)

    since_dt = parse_since_option(since)
    filter_list = parse_list_option(filter)
    exclude_list = parse_list_option(exclude)
    limit_val = limit if limit and limit > 0 else None

    client = ArenaClient(cache_enabled=not no_cache)

    # Fetch metadata for heading/URL upfront
    meta_preview: Optional[Dict[str, Any]] = None
    try:
        if ref.slug:
            meta_preview = client.fetch_channel_meta_by_slug(ref.slug, 1, 1)
        else:
            meta_preview = client.fetch_channel_meta_by_id(ref.channel_id) or {}
    except ArenaNotFound:
        client.close()
        if not os.getenv("ARENA_TOKEN"):
            sys.stderr.write(
                f"Error: Unauthorized for {channel} (set ARENA_TOKEN?).\n"
            )
        else:
            sys.stderr.write(f"Error: Not found: {channel}\n")
        sys.exit(1)
    except ArenaUnauthorized:
        client.close()
        sys.stderr.write(f"Error: Unauthorized for {channel} (set ARENA_TOKEN?).\n")
        sys.exit(1)
    except ArenaError as exc:
        client.close()
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    channel_url = canonical_channel_url(ref, meta_preview)
    if heading is not None:
        heading_lines = [
            collapse_spaces(part)
            for part in heading.splitlines()
            if part.strip()
        ]
    else:
        heading_lines = compute_channel_heading(meta_preview)

    printer = connect_printer()
    job = ArenaPrintJob(
        printer,
        config.scales,
        config.aligns,
        config.methods,
        config.ts_fmt,
        config.dithers,
        config.thresholds,
        config.diffusions,
        config.captions_str,
        config.spacing,
        footer,
        debug=config.debug,
    )

    if missing_ffmpeg and video == "frame":
        sys.stderr.write("Warning: ffmpeg not found; using preview images for videos.\n")

    def block_sort_key(item: Dict[str, Any]) -> float:
        ts = block_connected_at(item) or parse_timestamp(item.get("created_at"))
        if ts:
            try:
                return ts.timestamp()
            except Exception:
                pass
        return 0.0

    blocks = list(ChannelIterator(client, ref))
    if sort == "random":
        random.shuffle(blocks)
    else:
        reverse = sort == "desc"
        blocks.sort(key=block_sort_key, reverse=reverse)
    printed_blocks = 0
    heading_printed = False

    try:
        for block in blocks:
            if not should_include_block(block, include_channels, filter_list, exclude_list):
                continue

            added_dt = block_connected_at(block)
            if since_dt and (not added_dt or added_dt < since_dt):
                continue

            if not heading_printed and heading_lines:
                job.print_heading(heading_lines, trailing_blank=True)
                heading_printed = True


            block_id_val = str(block.get("id") or "unknown")
            success = print_arena_block(
                job,
                block,
                block_id_val,
                client,
                media_opts,
                block_qr_cfg,
                clean,
                added_override=added_dt,
            )
            if success:
                printed_blocks += 1
                if limit_val and printed_blocks >= limit_val:
                    break

        if printed_blocks == 0:
            sys.stderr.write("No blocks to print.\n")
            sys.exit(0)

        if qr_cfg.enabled and channel_url:
            printer.set(align="right", font="a")
            try:
                printer.qr(channel_url, size=qr_cfg.size, ec=qr_cfg.ec)
            except Exception as exc:
                sys.stderr.write(f"Warning: Failed to print QR for {channel_url}: {exc}\n")
            printer.set(align="left")

        job.print_footer()
        printer.cut()
    finally:
        printer.close()
        client.close()

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
HEADING_CHAR_WIDTH = max(1, CHAR_WIDTH // 2)
