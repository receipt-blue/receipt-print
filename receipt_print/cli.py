#!/usr/bin/env python3
import io
import os
import platform
import random
import re
import shlex
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Sequence
from tempfile import NamedTemporaryFile

import click
from click.core import ParameterSource
import requests
from PIL import Image
from escpos.escpos import QR_ECLEVEL_H, QR_ECLEVEL_L, QR_ECLEVEL_M, QR_ECLEVEL_Q

from .arena import (
    ArenaClient,
    ArenaDownloadError,
    ArenaError,
    ArenaNotFound,
    ArenaPrintJob,
    ArenaUnauthorized,
    MEDIA_TIMEOUT,
    USER_AGENT,
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
    ensure_trailing_newline,
    load_image_from_bytes,
    parse_block_identifier,
    parse_channel_identifier,
    parse_timestamp,
    pdf_bytes_to_images,
    format_timestamp,
    should_include_block,
    video_frame_from_bytes,
)
from .contact import (
    ContactInfo,
    normalize_contact_value,
    parse_vcards,
    print_contact_card,
)
from .image_utils import parse_caption_csv
from .printer import (
    CHAR_WIDTH,
    cat_files,
    connect_printer,
    count_lines,
    enforce_line_limit,
    env_no_cut,
    maybe_cut,
    print_text,
    scaled_char_width,
    sanitize_output,
    wrap_text,
)
from .shell import run_shell_commands


class GroupedOption(click.Option):
    def __init__(self, *args, group: Optional[str] = None, **kwargs):
        self.group = group
        super().__init__(*args, **kwargs)


def _write_bold_section(formatter: click.HelpFormatter, title: str, records: List[tuple[str, str]]) -> None:
    if not records:
        return
    formatter.write("\n")
    formatter.write(click.style(title, bold=True) + "\n")
    formatter.indent()
    formatter.write_dl(records)
    formatter.dedent()


PRINTING_COMMANDS = {
    "image",
    "pdf",
    "are.na",
    "imessage",
    "text",
    "cat",
    "shell",
    "contact",
}


class GroupedCommand(click.Command):
    def format_options(self, ctx, formatter):
        grouped: "OrderedDict[Optional[str], List[tuple[str, str]]]" = OrderedDict()
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if not record:
                continue
            group = getattr(param, "group", None)
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(record)

        if not grouped:
            return

        for group, records in grouped.items():
            if not records:
                continue
            section_name = (group or "Options").upper()
            _write_bold_section(formatter, section_name, records)


class GroupedGroup(click.Group):
    command_class = GroupedCommand

    def format_options(self, ctx, formatter):
        option_records: List[tuple[str, str]] = []
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record:
                option_records.append(record)
        _write_bold_section(formatter, "OPTIONS", option_records)

        command_records: List[tuple[str, str]] = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            if subcommand in PRINTING_COMMANDS:
                group_name = "Print"
            else:
                group_name = "Utility"
            command_records.append((group_name, subcommand, cmd.get_short_help_str()))

        if not command_records:
            return

        groups: "OrderedDict[str, List[tuple[str, str]]]" = OrderedDict()
        for group, name, help_str in command_records:
            groups.setdefault(group, []).append((name, help_str))

        for title, records in groups.items():
            _write_bold_section(formatter, title.upper(), records)

    def command(self, *args, **kwargs):
        kwargs.setdefault("cls", self.command_class)
        return super().command(*args, **kwargs)

    def group(self, *args, **kwargs):
        kwargs.setdefault("cls", type(self))
        return super().group(*args, **kwargs)


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
    brightness: List[float] = field(default_factory=lambda: [1.0])
    contrast: List[float] = field(default_factory=lambda: [1.0])
    gamma: List[float] = field(default_factory=lambda: [1.0])
    autocontrast: bool = False
    captions_str: Optional[str] = None
    footer_text: Optional[str] = None
    debug: bool = False
    spacing: int = 1
    wrap_mode: str = "hyphen"


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

PDF_IMAGE_TUNING_DEFAULTS: Dict[str, Any] = {
    "dither": "floyd",
    "threshold": "0.40",
    "brightness": "1.10",
    "contrast": "1.50",
    "gamma": "0.75",
    "autocontrast": True,
}

IMAGE_OPTION_FALLBACK_DEFAULTS: Dict[str, Any] = {
    "dither": None,
    "threshold": "0.5",
    "diffusion": "1.0",
    "brightness": "1.0",
    "contrast": "1.0",
    "gamma": "1.0",
    "autocontrast": None,
}


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


def resolve_no_cut(ctx: Optional[click.Context], no_cut: bool) -> bool:
    if env_no_cut():
        return True
    if no_cut:
        return True
    parent = ctx.parent if ctx else None
    while parent:
        if parent.params.get("no_cut"):
            return True
        parent = parent.parent
    return False


def resolve_wrap(ctx: Optional[click.Context], wrap: Optional[str]) -> str:
    if wrap is not None:
        return wrap
    parent = ctx.parent if ctx else None
    while parent:
        if "wrap" in parent.params and parent.params.get("wrap") is not None:
            return parent.params["wrap"]
        parent = parent.parent
    return "hyphen"


def validate_method(value):
    return value.lower() in {"raster", "column", "graphics"}


def validate_dither(value):
    return value is None or value.lower() in {"none", "thresh", "floyd", "atkinson"}


def validate_threshold(value):
    return 0.0 <= value <= 1.0


def validate_diffusion(value):
    return value >= 0.0


# Common image processing options
def _apply_options(func, options):
    for option in reversed(options):
        func = option(func)
    return func


IMAGE_TUNING_GROUP = "image tuning"
IMAGE_LAYOUT_GROUP = "layout"
DIAGNOSTICS_GROUP = "diagnostics"
OUTPUT_GROUP = "output control"
TEXT_GROUP = "text formatting"
VIDEO_GROUP = "video handling"
PDF_GROUP = "PDF handling"
ARENA_CONTENT_GROUP = "content filters"
QR_GROUP = "QR codes"
NETWORK_GROUP = "cache, networking"
IMESSAGE_GROUP = "imessage listener"
CONTACT_GROUP = "contact card"


def add_no_cut_option(func):
    return click.option(
        "--no-cut",
        is_flag=True,
        help="Skip cutting the paper after printing.",
        cls=GroupedOption,
        group=OUTPUT_GROUP,
    )(func)


def add_wrap_option(func):
    return click.option(
        "--wrap",
        type=click.Choice(["hyphen", "word", "none"], case_sensitive=False),
        default=None,
        show_default=False,
        help="Wrap text: hyphen, word, or none. Default: hyphen.",
        cls=GroupedOption,
        group=TEXT_GROUP,
    )(func)


core_image_options = [
    click.option(
        "--method",
        default="raster",
        help="Comma-separated: raster,column,graphics.",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--dither",
        help="Comma-separated: none,thresh,floyd,atkinson.",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--threshold",
        default="0.5",
        help="Comma-separated cutoff 0–1 for thresh/diffusion.",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--diffusion",
        default="1.0",
        help="Comma-separated diffusion strength (0=no spread,1=classic).",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--brightness",
        default="1.0",
        help="Comma-separated brightness multipliers (1.0=no change).",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--contrast",
        default="1.0",
        help="Comma-separated contrast multipliers (1.0=no change).",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--gamma",
        default="1.0",
        help="Comma-separated gamma values (<1 brightens, >1 darkens).",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--autocontrast",
        is_flag=True,
        flag_value=True,
        default=None,
        help="Run auto-contrast pass before applying --contrast multipliers.",
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--no-autocontrast",
        "autocontrast",
        is_flag=True,
        flag_value=False,
        hidden=True,
        cls=GroupedOption,
        group=IMAGE_TUNING_GROUP,
    ),
    click.option(
        "--spacing",
        type=int,
        default=1,
        help="Blank lines between each image.",
        cls=GroupedOption,
        group=IMAGE_LAYOUT_GROUP,
    ),
    click.option(
        "--debug",
        is_flag=True,
        help="Emit per-image debug info.",
        cls=GroupedOption,
        group=DIAGNOSTICS_GROUP,
    ),
]


sugar_image_options = [
    click.option(
        "--scale",
        default="1.0",
        help="Comma-separated floats for per-image scale.",
        cls=GroupedOption,
        group=IMAGE_LAYOUT_GROUP,
    ),
    click.option(
        "--align",
        default="center",
        help="Comma-separated: left,right,center,p-center,l-top,l-bottom,l-center.",
        cls=GroupedOption,
        group=IMAGE_LAYOUT_GROUP,
    ),
    click.option(
        "--timestamp",
        default="none",
        help="strftime string or 'none' to skip timestamp.",
        cls=GroupedOption,
        group=IMAGE_LAYOUT_GROUP,
    ),
    click.option(
        "--heading",
        help="Optional heading before images.",
        cls=GroupedOption,
        group=IMAGE_LAYOUT_GROUP,
    ),
    click.option(
        "--caption",
        help="Comma-separated list of per-image captions.",
        cls=GroupedOption,
        group=IMAGE_LAYOUT_GROUP,
    ),
    click.option(
        "--footer",
        help="Global footer text to print after all images.",
        cls=GroupedOption,
        group=IMAGE_LAYOUT_GROUP,
    ),
]


def add_core_image_options(func):
    """Decorator to add common image tuning options"""
    return _apply_options(func, core_image_options)


def add_image_options(func):
    """Decorator to add the full image option set to a command"""
    return _apply_options(func, core_image_options + sugar_image_options)


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
    brightness = parse_comma_separated(kwargs.get("brightness"), float)
    contrast = parse_comma_separated(kwargs.get("contrast"), float)
    gamma_vals = parse_comma_separated(kwargs.get("gamma"), float)
    autocontrast_val = kwargs.get("autocontrast", None)
    autocontrast = bool(autocontrast_val) if autocontrast_val not in (None, "") else False

    if not brightness:
        brightness = [1.0]
    if not contrast:
        contrast = [1.0]
    if not gamma_vals:
        gamma_vals = [1.0]

    # Validation
    if not all(validate_method(m) for m in methods):
        raise click.BadParameter("--method must be raster|column|graphics")
    if not all(validate_dither(d) for d in dithers):
        raise click.BadParameter("--dither invalid")
    if not all(validate_threshold(t) for t in thresholds):
        raise click.BadParameter("--threshold values must be 0–1")
    if not all(validate_diffusion(d) for d in diffusions):
        raise click.BadParameter("--diffusion must be >= 0")
    if not all(v > 0 for v in brightness):
        raise click.BadParameter("--brightness must be > 0")
    if not all(v > 0 for v in contrast):
        raise click.BadParameter("--contrast must be > 0")
    if not all(v > 0 for v in gamma_vals):
        raise click.BadParameter("--gamma must be > 0")

    wrap_mode = (kwargs.get("wrap") or "hyphen").lower()
    ts_fmt = None if kwargs["timestamp"].lower() == "none" else kwargs["timestamp"]

    return ImageProcessingConfig(
        scales=scales,
        aligns=[a.lower() for a in aligns],
        methods=[m.lower() for m in methods],
        dithers=[d.lower() if d and d != "none" else None for d in dithers],
        thresholds=thresholds,
        diffusions=diffusions,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma_vals,
        autocontrast=autocontrast,
        ts_fmt=ts_fmt,
        captions_str=kwargs["caption"],
        footer_text=kwargs["footer"],
        debug=kwargs["debug"],
        spacing=kwargs["spacing"],
        wrap_mode=wrap_mode,
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
    brightness: str,
    contrast: str,
    gamma: str,
    autocontrast: bool,
    wrap: Optional[str],
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
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "autocontrast": autocontrast,
        "wrap": wrap,
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


def resolve_ffmpeg_path(
    video_mode: str, explicit: Optional[str]
) -> Tuple[Optional[str], bool]:
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


def is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_contact_images(sources: Sequence[str]) -> Tuple[List[Image.Image], List[str]]:
    if not sources:
        return [], []
    exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
    images: List[Image.Image] = []
    names: List[str] = []

    for source in sources:
        if is_http_url(source):
            try:
                response = requests.get(
                    source, timeout=MEDIA_TIMEOUT, headers={"User-Agent": USER_AGENT}
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise click.UsageError(f"Could not download image {source}: {exc}")
            try:
                images.append(load_image_from_bytes(response.content))
            except Exception as exc:
                raise click.UsageError(f"Invalid image data from {source}: {exc}")
            names.append(source)
            continue

        path = os.path.expanduser(source)
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                if entry.lower().endswith(exts):
                    img_path = os.path.join(path, entry)
                    img = Image.open(img_path)
                    img.load()
                    images.append(img)
                    names.append(img_path)
            continue

        if not os.path.isfile(path):
            raise click.UsageError(f"missing file: {source}")
        if not path.lower().endswith(exts):
            raise click.UsageError(f"unsupported file type: {source}")
        img = Image.open(path)
        img.load()
        images.append(img)
        names.append(path)

    if not images:
        raise click.UsageError("No usable images found.")
    return images, names


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
                sys.stderr.write(
                    f"Warning: Could not load image attachment {url}: {exc}\n"
                )
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
                    sys.stderr.write(
                        f"Warning: PDF attachment {url} yielded no pages.\n"
                    )
                else:
                    if (
                        media_opts.pdf_mode == "first"
                        and media_opts.pdf_page_filter is None
                    ):
                        pdf_images = pdf_images[:1]
                    images.extend(pdf_images)
                    names.extend([f"{url}#page{i + 1}" for i in range(len(pdf_images))])
                    return images, names
            except ArenaDownloadError as exc:
                sys.stderr.write(f"Warning: Could not download PDF {url}: {exc}\n")
        # other attachment types fall back to preview/text

    klass = block_class(block)
    if (preview_urls and not images) or klass in {
        "image",
        "link",
        "media",
        "attachment",
        "channel",
    }:
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
    return [
        collapse_spaces(line.encode("ascii", "ignore").decode("ascii"))
        for line in lines
        if line
    ]


def print_arena_block(
    job: ArenaPrintJob,
    block: Dict[str, Any],
    block_id: str,
    client: ArenaClient,
    media_opts: MediaHandlingOptions,
    qr_cfg: QRConfig,
    clean: bool,
    images_only: bool = False,
    cut_between: bool = False,
    added_override: Optional[datetime] = None,
) -> bool:
    """Print a single block following the Are.na spec."""

    content_printed = False
    title = block_title(block)
    title_clean = collapse_spaces(title.encode("ascii", "ignore").decode("ascii"))

    klass = block_class(block)
    text_content = block_text_content(block) if klass == "text" else None

    images, image_names = gather_images_for_block(block, client, media_opts)

    if images:
        job.print_images(images, names=image_names)
        content_printed = True
    elif images_only:
        return False
    elif klass == "text" and text_content:
        text_content = collapse_spaces(
            LINK_PATTERN.sub(lambda m: m.group(1), text_content)
        )
        job.print_text(text_content, align="left", font="a")
        content_printed = True
    elif text_content:
        text_content = collapse_spaces(
            LINK_PATTERN.sub(lambda m: m.group(1), text_content)
        )
        job.print_text(text_content, align="left", font="a")
        content_printed = True

    if not content_printed:
        preview_urls = block_preview_urls(block)
        fallback = preview_urls[0] if preview_urls else canonical_block_url(block_id)
        fallback = collapse_spaces(fallback.encode("ascii", "ignore").decode("ascii"))
        job.print_text(fallback, align="left", font="a")
        content_printed = True

    if not clean and not images_only:
        if title_clean:
            job.print_text(title_clean, align="left", font="b", bold=True)

        description = block_description(block)
        if description:
            desc_clean = collapse_spaces(
                LINK_PATTERN.sub(
                    lambda m: m.group(1),
                    description.encode("ascii", "ignore").decode("ascii"),
                )
            )
            job.print_text(desc_clean, align="left", font="a", bold=False)
        metadata_lines = build_metadata_lines(block, added_override)
        for meta in metadata_lines:
            job.print_text(collapse_spaces(meta), align="right", font="a")

    if qr_cfg.enabled and not images_only:
        url = canonical_block_url(block_id)
        job.printer.set(align="right")
        try:
            job.printer.qr(url, size=qr_cfg.size, ec=qr_cfg.ec)
        except Exception as exc:
            sys.stderr.write(f"Warning: Failed to print QR for {url}: {exc}\n")

    if not cut_between:
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


def print_with_images(
    config: ImageProcessingConfig, images, names=None, heading=None, no_cut: bool = False
):
    """Print images with common workflow"""
    printer = connect_printer()

    if heading:
        printer.set(align="center", double_height=True, double_width=True)
        heading_width = scaled_char_width(CHAR_WIDTH, 2)
        wrapped_heading = wrap_text(heading, heading_width, config.wrap_mode)
        if wrapped_heading and not wrapped_heading.endswith("\n"):
            wrapped_heading += "\n"
        printer.text(wrapped_heading)
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
        brightness_list=config.brightness,
        contrast_list=config.contrast,
        gamma_list=config.gamma,
        autocontrast=config.autocontrast,
        captions_str=config.captions_str,
        footer_text=config.footer_text,
        debug=config.debug,
        spacing=config.spacing,
        names=names,
        wrap_mode=config.wrap_mode,
        no_cut=no_cut,
    )

    maybe_cut(printer, no_cut=no_cut)
    printer.close()


@click.group(cls=GroupedGroup, invoke_without_command=True)
@add_wrap_option
@add_no_cut_option
@click.pass_context
def cli(ctx, wrap, no_cut):
    """Print text or images to a receipt printer."""
    if ctx.invoked_subcommand is None:
        if not sys.stdin.isatty():
            print_text(
                sys.stdin.read(),
                no_cut=resolve_no_cut(ctx, no_cut),
                wrap_mode=resolve_wrap(ctx, wrap),
            )
        else:
            click.echo(ctx.get_help())
            sys.exit(1)


@cli.command()
@click.argument("text", nargs=-1, required=True)
@click.option(
    "-l", "--lines", is_flag=True, help="Join args with newlines instead of spaces."
)
@click.option(
    "--size",
    type=click.IntRange(1, 8),
    help="Text size multiplier (1-8). Sets width and height together.",
    cls=GroupedOption,
    group=TEXT_GROUP,
)
@click.option(
    "--width",
    "text_width",
    type=click.IntRange(1, 8),
    help="Text width multiplier (1-8).",
    cls=GroupedOption,
    group=TEXT_GROUP,
)
@click.option(
    "--height",
    "text_height",
    type=click.IntRange(1, 8),
    help="Text height multiplier (1-8).",
    cls=GroupedOption,
    group=TEXT_GROUP,
)
@add_wrap_option
@add_no_cut_option
@click.pass_context
def text(ctx, text, lines, size, text_width, text_height, wrap, no_cut):
    """Print literal text."""
    if size is not None and (text_width is not None or text_height is not None):
        raise click.UsageError("--size is mutually exclusive with --width/--height.")
    if size is not None:
        text_width = size
        text_height = size
    size_active = text_width is not None or text_height is not None
    use_lines = lines or (size_active and len(text) > 1)
    txt = "\n".join(text) if use_lines else " ".join(text)
    print_text(
        txt,
        no_cut=resolve_no_cut(ctx, no_cut),
        text_width=text_width,
        text_height=text_height,
        wrap_mode=resolve_wrap(ctx, wrap),
    )


@cli.command()
@click.argument("files", nargs=-1, required=True)
@add_wrap_option
@add_no_cut_option
@click.pass_context
def cat(ctx, files, wrap, no_cut):
    """Print files' contents."""
    cat_files(
        files,
        no_cut=resolve_no_cut(ctx, no_cut),
        wrap_mode=resolve_wrap(ctx, wrap),
    )


@cli.command()
@click.argument("files", nargs=-1)
@add_wrap_option
@click.pass_context
def count(ctx, files, wrap):
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
    wrap_mode = resolve_wrap(ctx, wrap)
    wrapped = wrap_text(buf, CHAR_WIDTH, wrap_mode)
    click.echo(count_lines(wrapped, CHAR_WIDTH))


@cli.command()
@click.argument("commands", nargs=-1, required=True)
@click.option("--no-wrap", is_flag=True, help="Standard capture instead of PTY wrap.")
@add_wrap_option
@add_no_cut_option
@click.pass_context
def shell(ctx, commands, no_wrap, wrap, no_cut):
    """Run shell commands and print output."""
    out = run_shell_commands(commands, wrap_tty=not no_wrap, columns=CHAR_WIDTH)
    print_text(
        out,
        no_cut=resolve_no_cut(ctx, no_cut),
        wrap_mode=resolve_wrap(ctx, wrap),
    )


@cli.command()
@click.argument("files", nargs=-1)
@add_image_options
@add_wrap_option
@add_no_cut_option
@click.pass_context
def image(ctx, files, no_cut, **kwargs):
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
            if parts and all(
                is_http_url(p) or os.path.exists(os.path.expanduser(p))
                for p in parts
            ):
                files = [
                    p if is_http_url(p) else os.path.expanduser(p) for p in parts
                ]
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
        images = []
        names = []
        for p in files:
            if is_http_url(p):
                try:
                    response = requests.get(
                        p, timeout=MEDIA_TIMEOUT, headers={"User-Agent": USER_AGENT}
                    )
                    response.raise_for_status()
                except requests.RequestException as exc:
                    click.echo(f"Could not download image {p}: {exc}", err=True)
                    sys.exit(1)
                try:
                    images.append(load_image_from_bytes(response.content))
                except Exception as exc:
                    click.echo(f"Invalid image data from {p}: {exc}", err=True)
                    sys.exit(1)
                names.append(p)
                continue

            if os.path.isdir(p):
                for e in sorted(os.listdir(p)):
                    if e.lower().endswith(exts):
                        img_path = os.path.join(p, e)
                        images.append(Image.open(img_path))
                        names.append(img_path)
                continue

            if not os.path.isfile(p):
                click.echo(f"missing file: {p}", err=True)
                sys.exit(1)
            if not p.lower().endswith(exts):
                click.echo(f"unsupported file type: {p}", err=True)
                sys.exit(1)
            images.append(Image.open(p))
            names.append(p)

        if not images:
            click.echo("No usable images found.", err=True)
            sys.exit(1)

    kwargs["wrap"] = resolve_wrap(ctx, kwargs.get("wrap"))
    config = create_image_config(**kwargs)
    print_with_images(
        config,
        images,
        names=names,
        heading=kwargs["heading"],
        no_cut=resolve_no_cut(ctx, no_cut),
    )


@cli.command()
@click.argument("vcards", nargs=-1)
@click.option(
    "--name",
    help="Contact name.",
    cls=GroupedOption,
    group=CONTACT_GROUP,
)
@click.option(
    "--email",
    help="Contact email.",
    cls=GroupedOption,
    group=CONTACT_GROUP,
)
@click.option(
    "--phone",
    help="Contact phone number.",
    cls=GroupedOption,
    group=CONTACT_GROUP,
)
@click.option(
    "--image",
    "image_sources",
    multiple=True,
    help="Contact photo path or URL (repeatable).",
    cls=GroupedOption,
    group=CONTACT_GROUP,
)
@click.option(
    "--caption",
    help="Comma-separated list of per-contact captions.",
    cls=GroupedOption,
    group=CONTACT_GROUP,
)
@click.option(
    "--align",
    type=click.Choice(
        ["left", "right", "center", "p-center", "l-top", "l-bottom", "l-center"],
        case_sensitive=False,
    ),
    default="center",
    show_default=True,
    help="Image alignment; use l-top/l-bottom/l-center for landscape cards.",
    cls=GroupedOption,
    group=CONTACT_GROUP,
)
@click.option(
    "--qr-size",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="QR module size (printer dependent).",
    cls=GroupedOption,
    group=QR_GROUP,
)
@add_wrap_option
@add_no_cut_option
@click.pass_context
def contact(
    ctx,
    vcards,
    name,
    email,
    phone,
    image_sources,
    caption,
    align,
    qr_size,
    wrap,
    no_cut,
):
    """Print a contact card from fields or vCard files."""
    manual_values = any([name, email, phone])
    if vcards and manual_values:
        raise click.UsageError(
            "vCard arguments are mutually exclusive with --name/--email/--phone."
        )
    if not vcards and not manual_values:
        raise click.UsageError(
            "Provide a vCard path or at least one of --name/--email/--phone."
        )

    contacts: List[ContactInfo] = []
    if vcards:
        for raw_path in vcards:
            path = Path(raw_path).expanduser()
            if not path.is_file():
                raise click.UsageError(f"missing file: {raw_path}")
            try:
                raw = path.read_text(errors="replace")
            except Exception as exc:
                raise click.UsageError(f"Error reading {raw_path}: {exc}")
            parsed = parse_vcards(raw)
            if len(parsed) != 1:
                raise click.UsageError(
                    f"Expected a single vCard in {raw_path}, found {len(parsed)}."
                )
            contacts.append(parsed[0])
    else:
        contact_info = ContactInfo(
            name=normalize_contact_value(name),
            email=normalize_contact_value(email),
            phone=normalize_contact_value(phone),
        )
        if not contact_info.has_data():
            raise click.UsageError("Contact fields are empty after normalization.")
        contacts.append(contact_info)

    images, _ = load_contact_images(image_sources) if image_sources else ([], [])
    if vcards:
        contacts_missing_photo = [c for c in contacts if c.photo is None]
        if images and len(images) != len(contacts_missing_photo):
            raise click.UsageError(
                "Provide one --image per vCard without an embedded photo."
            )
    elif len(images) > 1:
        raise click.UsageError("Only one --image is allowed without vCards.")

    wrap_mode = resolve_wrap(ctx, wrap)
    effective_no_cut = resolve_no_cut(ctx, no_cut)
    align = align.lower()
    captions = parse_caption_csv(caption) if caption else []

    printer = connect_printer()
    try:
        image_idx = 0
        for idx, contact_info in enumerate(contacts):
            photo = contact_info.photo
            if photo is None and image_idx < len(images):
                photo = images[image_idx]
                image_idx += 1
            caption_text = None
            if captions:
                caption_text = captions[idx] if idx < len(captions) else captions[-1]
            try:
                print_contact_card(
                    printer,
                    contact_info,
                    photo=photo,
                    wrap_mode=wrap_mode,
                    align=align,
                    qr_size=qr_size,
                    caption=caption_text,
                )
            except RuntimeError as exc:
                raise click.ClickException(str(exc))

            if idx < len(contacts) - 1:
                printer.text("\n")

        maybe_cut(printer, no_cut=effective_no_cut)
    finally:
        printer.close()


@cli.group(name="are.na", cls=GroupedGroup)
def arena_group():
    """Interact with Are.na blocks and channels."""
    pass


@arena_group.command("block")
@click.argument("blocks", nargs=-1, required=True)
@add_core_image_options
@click.option(
    "--heading",
    help="Optional heading printed once before blocks.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@click.option(
    "--caption",
    help="Comma-separated list of per-image captions.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@click.option(
    "--footer",
    help="Footer text printed after all output.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@click.option(
    "--clean",
    is_flag=True,
    help="Print content only (no title/description/metadata).",
    cls=GroupedOption,
    group=OUTPUT_GROUP,
)
@add_wrap_option
@add_no_cut_option
@click.option(
    "--video",
    type=click.Choice(["frame", "preview"]),
    default="frame",
    help="Video attachments: 'frame' extracts first frame (requires ffmpeg), 'preview' uses preview image.",
    cls=GroupedOption,
    group=VIDEO_GROUP,
)
@click.option(
    "--ffmpeg",
    help="Explicit ffmpeg path for --video=frame.",
    cls=GroupedOption,
    group=VIDEO_GROUP,
)
@click.option(
    "--pdf",
    type=click.Choice(["first", "all"]),
    default="first",
    help="PDF attachments: print first page or all pages.",
    cls=GroupedOption,
    group=PDF_GROUP,
)
@click.option(
    "--pdf-pages",
    help="Specific PDF pages (1-indexed), e.g. '1,3,5'.",
    cls=GroupedOption,
    group=PDF_GROUP,
)
@click.option(
    "--pdf-range",
    help="PDF page range 'start,end'; use -1 for end.",
    cls=GroupedOption,
    group=PDF_GROUP,
)
@click.option(
    "--qr",
    is_flag=True,
    help="Append a QR code with the block URL.",
    cls=GroupedOption,
    group=QR_GROUP,
)
@click.option(
    "--qr-size",
    type=int,
    default=4,
    help="QR module size (printer dependent). Default: 4.",
    cls=GroupedOption,
    group=QR_GROUP,
)
@click.option(
    "--qr-correction",
    type=click.Choice(["L", "M", "Q", "H"]),
    default="M",
    help="QR error correction level.",
    cls=GroupedOption,
    group=QR_GROUP,
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Bypass local Are.na cache for this run.",
    cls=GroupedOption,
    group=NETWORK_GROUP,
)
@click.pass_context
def arena_block(
    ctx,
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
    brightness,
    contrast,
    gamma,
    autocontrast,
    video,
    ffmpeg,
    pdf,
    pdf_pages,
    pdf_range,
    qr,
    qr_size,
    qr_correction,
    no_cache,
    wrap,
    debug,
    no_cut,
):
    """Print one or more Are.na blocks."""
    effective_no_cut = resolve_no_cut(ctx, no_cut)
    media_opts, missing_ffmpeg = build_media_options(
        video, ffmpeg, pdf, pdf_pages, pdf_range
    )
    qr_cfg = QRConfig(
        enabled=qr, size=qr_size, ec=QR_LEVELS.get(qr_correction, QR_ECLEVEL_M)
    )
    wrap_mode = resolve_wrap(ctx, wrap)
    config = create_arena_image_config(
        method,
        dither,
        threshold,
        diffusion,
        caption,
        footer,
        spacing,
        debug,
        brightness,
        contrast,
        gamma,
        autocontrast,
        wrap_mode,
    )

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
        config.brightness,
        config.contrast,
        config.gamma,
        config.autocontrast,
        config.captions_str,
        config.spacing,
        footer,
        debug=config.debug,
        wrap_mode=wrap_mode,
        no_cut=effective_no_cut,
    )

    if missing_ffmpeg and video == "frame":
        sys.stderr.write(
            "Warning: ffmpeg not found; using preview images for videos.\n"
        )

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
                clean=clean,
            )
            if success:
                printed_any = True

        if printed_any:
            job.print_footer()
            maybe_cut(printer, no_cut=effective_no_cut)
    finally:
        printer.close()
        client.close()

    if not printed_any:
        sys.exit(1)


@arena_group.command("channel")
@click.argument("channel", required=True)
@add_core_image_options
@click.option(
    "--heading",
    help="Override heading text for the channel.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@click.option(
    "--caption",
    help="Comma-separated list of per-image captions.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@click.option(
    "--footer",
    help="Footer text printed after all output.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@click.option(
    "--clean",
    is_flag=True,
    help="Print content only (no title/description/metadata).",
    cls=GroupedOption,
    group=OUTPUT_GROUP,
)
@click.option(
    "--video",
    type=click.Choice(["frame", "preview"]),
    default="frame",
    help="Video attachments: 'frame' extracts first frame (requires ffmpeg), 'preview' uses preview image.",
    cls=GroupedOption,
    group=VIDEO_GROUP,
)
@click.option(
    "--ffmpeg",
    help="Explicit ffmpeg path for --video=frame.",
    cls=GroupedOption,
    group=VIDEO_GROUP,
)
@click.option(
    "--pdf",
    type=click.Choice(["first", "all"]),
    default="first",
    help="PDF attachments: print first page or all pages.",
    cls=GroupedOption,
    group=PDF_GROUP,
)
@click.option(
    "--pdf-pages",
    help="Specific PDF pages (1-indexed), e.g. '1,3,5'.",
    cls=GroupedOption,
    group=PDF_GROUP,
)
@click.option(
    "--pdf-range",
    help="PDF page range 'start,end'; use -1 for end.",
    cls=GroupedOption,
    group=PDF_GROUP,
)
@click.option(
    "--filter",
    help="Comma-separated block classes to include (text,image,link,media,attachment,channel).",
    cls=GroupedOption,
    group=ARENA_CONTENT_GROUP,
)
@click.option(
    "--exclude",
    help="Comma-separated block classes to skip.",
    cls=GroupedOption,
    group=ARENA_CONTENT_GROUP,
)
@click.option(
    "--limit",
    type=int,
    help="Stop after printing N blocks.",
    cls=GroupedOption,
    group=ARENA_CONTENT_GROUP,
)
@click.option(
    "--since",
    help="Only include blocks with connected_at >= ISO timestamp.",
    cls=GroupedOption,
    group=ARENA_CONTENT_GROUP,
)
@click.option(
    "--include-channels",
    is_flag=True,
    help="Include nested channels within contents.",
    cls=GroupedOption,
    group=ARENA_CONTENT_GROUP,
)
@click.option(
    "--cut-between",
    is_flag=True,
    help="Cut between each printed image; outputs images only (incompatible with --no-cut).",
    cls=GroupedOption,
    group=OUTPUT_GROUP,
)
@add_wrap_option
@add_no_cut_option
@click.option(
    "--qr",
    is_flag=True,
    help="Print a QR code of the channel URL beneath the heading.",
    cls=GroupedOption,
    group=QR_GROUP,
)
@click.option(
    "--qr-size",
    type=int,
    default=4,
    help="QR module size (printer dependent). Default: 4.",
    cls=GroupedOption,
    group=QR_GROUP,
)
@click.option(
    "--qr-correction",
    type=click.Choice(["L", "M", "Q", "H"]),
    default="M",
    help="QR error correction level.",
    cls=GroupedOption,
    group=QR_GROUP,
)
@click.option(
    "--sort",
    type=click.Choice(["asc", "desc", "random"]),
    default="desc",
    help="Ordering for channel contents: ascending, descending, or random (default desc).",
    cls=GroupedOption,
    group=ARENA_CONTENT_GROUP,
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Bypass local Are.na cache for this run.",
    cls=GroupedOption,
    group=NETWORK_GROUP,
)
@click.pass_context
def arena_channel(
    ctx,
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
    brightness,
    contrast,
    gamma,
    autocontrast,
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
    cut_between,
    qr,
    qr_size,
    qr_correction,
    sort,
    no_cache,
    wrap,
    debug,
    no_cut,
):
    """Print all blocks in an Are.na channel."""
    if cut_between and env_no_cut():
        raise click.UsageError(
            "RP_NO_CUT=1 disables cutting; --cut-between is incompatible."
        )
    effective_no_cut = resolve_no_cut(ctx, no_cut)
    if cut_between and effective_no_cut:
        raise click.UsageError("--no-cut is mutually exclusive with --cut-between.")
    try:
        ref = parse_channel_identifier(channel)
    except ArenaError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    media_opts, missing_ffmpeg = build_media_options(
        video, ffmpeg, pdf, pdf_pages, pdf_range
    )
    wrap_mode = resolve_wrap(ctx, wrap)
    config = create_arena_image_config(
        method,
        dither,
        threshold,
        diffusion,
        caption,
        footer,
        spacing,
        debug,
        brightness,
        contrast,
        gamma,
        autocontrast,
        wrap_mode,
    )
    qr_cfg = QRConfig(
        enabled=qr, size=qr_size, ec=QR_LEVELS.get(qr_correction, QR_ECLEVEL_M)
    )
    block_qr_cfg = QRConfig(enabled=False, size=qr_size, ec=qr_cfg.ec)

    cut_mode = cut_between
    clean_mode = clean or cut_mode
    if cut_mode:
        config.spacing = 0
        config.captions_str = None
        config.footer_text = None
        qr_cfg.enabled = False

    since_dt = parse_since_option(since)
    filter_list = parse_list_option(filter)
    exclude_list = parse_list_option(exclude)
    limit_val = limit if limit and limit > 0 else None

    if cut_mode and not filter_list:
        filter_list = ["image"]

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
            sys.stderr.write(f"Error: Unauthorized for {channel} (set ARENA_TOKEN?).\n")
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
            collapse_spaces(part) for part in heading.splitlines() if part.strip()
        ]
    else:
        heading_lines = compute_channel_heading(meta_preview)

    if cut_mode:
        heading_lines = []

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
        config.brightness,
        config.contrast,
        config.gamma,
        config.autocontrast,
        config.captions_str,
        config.spacing,
        None if cut_mode else footer,
        debug=config.debug,
        wrap_mode=wrap_mode,
        auto_orient=cut_mode,
        cut_between=cut_mode,
        no_cut=effective_no_cut,
    )

    if missing_ffmpeg and video == "frame":
        sys.stderr.write(
            "Warning: ffmpeg not found; using preview images for videos.\n"
        )

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
            if not should_include_block(
                block, include_channels, filter_list, exclude_list
            ):
                continue

            added_dt = block_connected_at(block)
            if since_dt and (not added_dt or added_dt < since_dt):
                continue

            if not cut_mode and not heading_printed and heading_lines:
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
                clean=clean_mode,
                images_only=cut_mode,
                cut_between=cut_mode,
                added_override=added_dt,
            )
            if success:
                printed_blocks += 1
                if limit_val and printed_blocks >= limit_val:
                    break

        if printed_blocks == 0:
            sys.stderr.write("No blocks to print.\n")
            sys.exit(0)

        if not cut_mode and qr_cfg.enabled and channel_url:
            printer.set(align="right", font="a")
            try:
                printer.qr(channel_url, size=qr_cfg.size, ec=qr_cfg.ec)
            except Exception as exc:
                sys.stderr.write(
                    f"Warning: Failed to print QR for {channel_url}: {exc}\n"
                )
            printer.set(align="left")

        if not cut_mode:
            job.print_footer()
            maybe_cut(printer, no_cut=effective_no_cut)
    finally:
        printer.close()
        client.close()


@cli.group(cls=GroupedGroup)
def imessage():
    """Listen for incoming iMessage printer-emoji triggers (macOS only)."""
    pass


@imessage.command("listen")
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    help="Messages chat.db path (default: ~/Library/Messages/chat.db).",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--attachments",
    type=click.Path(path_type=Path),
    help="Attachments root (default: ~/Library/Messages/Attachments).",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--state",
    type=click.Path(path_type=Path),
    help="State file path for last processed message id.",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--poll",
    type=click.FloatRange(min=0.1),
    default=2.0,
    show_default=True,
    help="Polling interval in seconds.",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--backfill",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Process the last N incoming messages on first run.",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--batch",
    "batch_size",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Max messages to process per poll.",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--once",
    is_flag=True,
    help="Process once and exit.",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--reset-state",
    is_flag=True,
    help="Ignore any existing state file on startup.",
    cls=GroupedOption,
    group=IMESSAGE_GROUP,
)
@click.option(
    "--scale",
    default="1.0",
    help="Comma-separated floats for per-image scale.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@click.option(
    "--align",
    default="center",
    help="Comma-separated: left,right,center,p-center,l-top,l-bottom,l-center.",
    cls=GroupedOption,
    group=IMAGE_LAYOUT_GROUP,
)
@add_core_image_options
@add_wrap_option
@add_no_cut_option
@click.pass_context
def imessage_listen(
    ctx,
    db,
    attachments,
    state,
    poll,
    backfill,
    batch_size,
    once,
    reset_state,
    scale,
    align,
    method,
    dither,
    threshold,
    diffusion,
    brightness,
    contrast,
    gamma,
    autocontrast,
    spacing,
    debug,
    wrap,
    no_cut,
):
    """Listen for incoming iMessage messages containing the printer emoji (U+1F5A8)."""
    if platform.system() != "Darwin":
        click.echo("Error: iMessage listener only runs on macOS.", err=True)
        sys.exit(1)

    get_source = getattr(ctx, "get_parameter_source", None)
    sentinel = object()
    option_values = {
        "dither": dither,
        "threshold": threshold,
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "autocontrast": autocontrast,
    }

    def user_supplied(param_name: str) -> bool:
        if callable(get_source):
            try:
                source = get_source(param_name)
            except Exception:
                source = None
            if source is not None:
                if source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP):
                    return False
                return True

        default_val = IMAGE_OPTION_FALLBACK_DEFAULTS.get(param_name, sentinel)
        if default_val is sentinel:
            return False
        return option_values.get(param_name, sentinel) != default_val

    if not user_supplied("dither"):
        dither = PDF_IMAGE_TUNING_DEFAULTS["dither"]
    if not user_supplied("threshold"):
        threshold = PDF_IMAGE_TUNING_DEFAULTS["threshold"]
    if not user_supplied("brightness"):
        brightness = PDF_IMAGE_TUNING_DEFAULTS["brightness"]
    if not user_supplied("contrast"):
        contrast = PDF_IMAGE_TUNING_DEFAULTS["contrast"]
    if not user_supplied("gamma"):
        gamma = PDF_IMAGE_TUNING_DEFAULTS["gamma"]
    if not user_supplied("autocontrast"):
        autocontrast = bool(PDF_IMAGE_TUNING_DEFAULTS["autocontrast"])

    wrap_mode = resolve_wrap(ctx, wrap)
    config = create_image_config(
        scale=scale,
        align=align,
        method=method,
        timestamp="none",
        dither=dither or "",
        threshold=threshold,
        diffusion=diffusion,
        caption=None,
        footer=None,
        debug=debug,
        spacing=spacing,
        heading=None,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
        autocontrast=autocontrast,
        wrap=wrap_mode,
    )

    from .imessage import (
        default_attachments_path,
        default_db_path,
        default_state_path,
        listen,
    )

    listen(
        db_path=db or default_db_path(),
        attachments_path=attachments or default_attachments_path(),
        state_path=state or default_state_path(),
        poll_interval=poll,
        backfill=backfill,
        batch_size=batch_size,
        once=once,
        reset_state=reset_state,
        image_config=config,
        wrap_mode=wrap_mode,
        no_cut=resolve_no_cut(ctx, no_cut),
    )


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
@add_wrap_option
@add_no_cut_option
@click.pass_context
def pdf(ctx, files, format, range, pages, no_cut, **kwargs):
    """Print PDF files by rendering each page to images before sending them to the printer."""
    pdf_bytes = None
    effective_no_cut = resolve_no_cut(ctx, no_cut)
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

    wrap_mode = resolve_wrap(ctx, kwargs.get("wrap"))
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
        print_text(txt, no_cut=effective_no_cut, wrap_mode=wrap_mode)
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

    get_source = getattr(ctx, "get_parameter_source", None)
    sentinel = object()

    def user_supplied(param_name: str) -> bool:
        if callable(get_source):
            try:
                source = get_source(param_name)
            except Exception:
                source = None
            if source is not None:
                if source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP):
                    return False
                return True

        default_val = IMAGE_OPTION_FALLBACK_DEFAULTS.get(param_name, sentinel)
        if default_val is sentinel:
            return False
        return kwargs.get(param_name, sentinel) != default_val

    for key, value in PDF_IMAGE_TUNING_DEFAULTS.items():
        if key == "autocontrast":
            if not user_supplied(key):
                kwargs[key] = bool(value)
            continue
        if not user_supplied(key):
            kwargs[key] = value

    kwargs["wrap"] = wrap_mode
    config = create_image_config(**kwargs)
    print_with_images(
        config,
        images,
        names=names,
        heading=kwargs["heading"],
        no_cut=effective_no_cut,
    )


for param in pdf.params:
    if isinstance(param, click.Option) and param.name == "autocontrast":
        if "--autocontrast" in param.opts:
            param.hidden = True
        if "--no-autocontrast" in param.opts:
            param.hidden = False
            param.help = "Skip the auto-contrast pre-pass; only --contrast multipliers are applied."


@cli.group(cls=GroupedGroup)
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
