#!/usr/bin/env python3
"""
Markdown rendering to images for thermal receipt printers.

Parses markdown and rasterizes it to PIL images, supporting:
- Headings (h1-h6)
- Bold, italic, strikethrough
- Code blocks and inline code
- Lists (ordered and unordered)
- Tables
- Blockquotes
- Horizontal rules
- Links (rendered as text)
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.request import urlretrieve

from PIL import Image, ImageDraw, ImageFont

UNIFONT_VERSION = "16.0.02"
UNIFONT_URL = f"https://unifoundry.com/pub/unifont/unifont-{UNIFONT_VERSION}/font-builds/unifont-{UNIFONT_VERSION}.otf"
NOTO_EMOJI_URL = "https://fonts.gstatic.com/s/notoemoji/v62/bMrnmSyK7YY-MEu6aWjPDs-ar6uWaGWuob-r0jwv.ttf"

PRINTER_WIDTH = 576
DEFAULT_FONT_SIZE = 24
HEADING_SIZES = {1: 48, 2: 40, 3: 32, 4: 28, 5: 24, 6: 22}
CODE_FONT_SIZE = 20
TABLE_FONT_SIZE = 20
MIN_COLUMN_WIDTH = 60


def get_cache_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "receipt-print"
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return Path(local) / "receipt-print" / "cache"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        return Path(xdg) / "receipt-print"


def get_unifont_path() -> Path:
    return get_cache_dir() / f"unifont-{UNIFONT_VERSION}.otf"


def get_emoji_path() -> Path:
    return get_cache_dir() / "noto-emoji.ttf"


def ensure_font_cached(url: str, path: Path, name: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"[markdown] Downloading {name}...\n")
    urlretrieve(url, path)
    sys.stderr.write(f"[markdown] Cached {name} at {path}\n")
    return path


def ensure_fonts() -> Tuple[Path, Path]:
    unifont = ensure_font_cached(UNIFONT_URL, get_unifont_path(), "Unifont")
    emoji = ensure_font_cached(NOTO_EMOJI_URL, get_emoji_path(), "Noto Emoji")
    return unifont, emoji


_font_cache: dict = {}


def get_font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold, italic)
    if key in _font_cache:
        return _font_cache[key]
    unifont_path, _ = ensure_fonts()
    font = ImageFont.truetype(str(unifont_path), size)
    _font_cache[key] = font
    return font


def round_up_to_8(n: int) -> int:
    return ((n + 7) // 8) * 8


def apply_threshold(img: Image.Image, threshold: int = 200) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img.point(lambda x: 255 if x > threshold else 0, mode="1")


def is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3040 <= cp <= 0x309F
        or 0x30A0 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


@dataclass
class StyledSpan:
    start: int
    end: int
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    code: bool = False


@dataclass
class TextStyle:
    font_size: int = DEFAULT_FONT_SIZE
    bold: bool = False
    italic: bool = False
    line_height_mult: float = 1.3


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    if not text:
        return []
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_width:
        return [text]

    lines = []
    current_line = ""

    for char in text:
        test_line = current_line + char
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]

        if test_width > max_width and current_line:
            if is_cjk(char) or char == " ":
                lines.append(current_line)
                current_line = "" if char == " " else char
            else:
                last_space = current_line.rfind(" ")
                if last_space > 0:
                    lines.append(current_line[:last_space])
                    current_line = current_line[last_space + 1 :] + char
                else:
                    lines.append(current_line)
                    current_line = char
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    return [line for line in lines if line.strip()]


def render_text_block(
    text: str,
    style: TextStyle,
    max_width: int = PRINTER_WIDTH,
    spans: Optional[List[StyledSpan]] = None,
) -> Image.Image:
    font = get_font(style.font_size, style.bold, style.italic)
    line_height = int(style.font_size * style.line_height_mult)
    padding = max(2, style.font_size // 10)

    measure_img = Image.new("L", (max_width * 2, style.font_size * 2), 255)
    measure_draw = ImageDraw.Draw(measure_img)

    wrapped_lines = wrap_text(measure_draw, text, font, max_width - padding * 2)
    if not wrapped_lines:
        wrapped_lines = [""]

    width = round_up_to_8(max_width)
    height = round_up_to_8(len(wrapped_lines) * line_height + padding * 2)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    char_positions = []
    offset = 0
    for line_idx, line in enumerate(wrapped_lines):
        for char_idx, char in enumerate(line):
            char_positions.append((line_idx, char_idx, offset))
            offset += 1
        offset += 1

    for line_idx, line in enumerate(wrapped_lines):
        y = padding + line_idx * line_height
        if spans:
            x = padding
            line_start = sum(len(wrapped_lines[i]) + 1 for i in range(line_idx))
            for char_idx, char in enumerate(line):
                global_idx = line_start + char_idx
                in_span = None
                for span in spans:
                    if span.start <= global_idx < span.end:
                        in_span = span
                        break

                char_bold = style.bold or (in_span and in_span.bold)
                char_font = get_font(style.font_size, char_bold, style.italic)
                char_bbox = draw.textbbox((0, 0), char, font=char_font)
                char_width = char_bbox[2] - char_bbox[0]

                draw.text((x, y), char, fill=0, font=char_font)
                if char_bold and not style.bold:
                    draw.text((x + 1, y), char, fill=0, font=char_font)

                if in_span and in_span.underline:
                    underline_y = y + style.font_size + 2
                    draw.line([(x, underline_y), (x + char_width, underline_y)], fill=0, width=1)

                if in_span and in_span.strikethrough:
                    strike_y = y + style.font_size // 2
                    draw.line([(x, strike_y), (x + char_width, strike_y)], fill=0, width=1)

                x += char_width
        else:
            draw.text((padding, y), line, fill=0, font=font)
            if style.bold:
                draw.text((padding + 1, y), line, fill=0, font=font)

    if style.font_size <= 32:
        img = apply_threshold(img, 200)

    return img


def render_code_block(code: str, max_width: int = PRINTER_WIDTH) -> Image.Image:
    font = get_font(CODE_FONT_SIZE)
    line_height = int(CODE_FONT_SIZE * 1.2)
    padding = 8
    border = 1

    lines = code.split("\n")

    width = round_up_to_8(max_width)
    content_height = len(lines) * line_height + padding * 2
    height = round_up_to_8(content_height + border * 2)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [(0, 0), (width - 1, height - 1)],
        outline=0,
        width=border,
    )

    for i, line in enumerate(lines):
        y = border + padding + i * line_height
        draw.text((border + padding, y), line, fill=0, font=font)

    return apply_threshold(img, 220)


def render_table(
    headers: List[str],
    rows: List[List[str]],
    max_width: int = PRINTER_WIDTH,
) -> Image.Image:
    font = get_font(TABLE_FONT_SIZE)
    bold_font = get_font(TABLE_FONT_SIZE, bold=True)
    line_height = int(TABLE_FONT_SIZE * 1.3)
    padding_x = 4
    padding_y = 2

    measure_img = Image.new("L", (max_width * 2, 100), 255)
    measure_draw = ImageDraw.Draw(measure_img)

    header_widths = []
    for h in headers:
        bbox = measure_draw.textbbox((0, 0), h, font=bold_font)
        header_widths.append(bbox[2] - bbox[0] + padding_x * 2)

    total_header_width = sum(header_widths)
    if total_header_width <= max_width:
        extra = max_width - total_header_width
        per_column = extra // len(headers)
        col_widths = [w + per_column for w in header_widths]
    else:
        col_widths = [max(MIN_COLUMN_WIDTH, int(w / total_header_width * max_width)) for w in header_widths]

    row_heights = []
    for row in rows:
        max_lines = 1
        for i, cell in enumerate(row):
            cell_width = col_widths[i] - padding_x * 2 if i < len(col_widths) else MIN_COLUMN_WIDTH
            wrapped = wrap_text(measure_draw, cell, font, cell_width)
            max_lines = max(max_lines, min(len(wrapped), 3))
        row_heights.append(max_lines * line_height + padding_y * 2)

    header_height = line_height + padding_y * 2
    total_height = header_height + 1 + sum(row_heights) + len(rows)

    width = round_up_to_8(max_width)
    height = round_up_to_8(total_height)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    y = 0
    x = 0
    for i, header in enumerate(headers):
        cell_width = col_widths[i] if i < len(col_widths) else MIN_COLUMN_WIDTH
        bbox = draw.textbbox((0, 0), header, font=bold_font)
        text_width = bbox[2] - bbox[0]
        if text_width > cell_width - padding_x * 2:
            header = header[: int((cell_width - padding_x * 2) / (TABLE_FONT_SIZE * 0.6))] + "…"
        draw.text((x + padding_x, y + padding_y + TABLE_FONT_SIZE), header, fill=0, font=bold_font)
        x += cell_width
    y += header_height

    draw.line([(0, y), (max_width, y)], fill=0, width=1)
    y += 1

    for row_idx, row in enumerate(rows):
        row_height = row_heights[row_idx]
        x = 0
        for col_idx, cell in enumerate(row):
            cell_width = col_widths[col_idx] if col_idx < len(col_widths) else MIN_COLUMN_WIDTH
            content_width = cell_width - padding_x * 2
            wrapped = wrap_text(draw, cell, font, content_width)[:3]

            if len(wrapped) == 3:
                last = wrapped[2]
                if len(last) > 1:
                    wrapped[2] = last[:-1] + "…"

            for line_idx, line in enumerate(wrapped):
                draw.text(
                    (x + padding_x, y + padding_y + TABLE_FONT_SIZE + line_idx * line_height),
                    line,
                    fill=0,
                    font=font,
                )
            x += cell_width

        y += row_height
        if row_idx < len(rows) - 1:
            draw.line([(0, y), (max_width, y)], fill=0, width=1)
            y += 1

    return apply_threshold(img, 220)


def render_horizontal_rule(max_width: int = PRINTER_WIDTH) -> Image.Image:
    width = round_up_to_8(max_width)
    height = 16
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    y = height // 2
    draw.line([(0, y), (width, y)], fill=0, width=1)
    return img


def render_blockquote(text: str, max_width: int = PRINTER_WIDTH) -> Image.Image:
    indent = 16
    bar_width = 3
    bar_margin = 8

    style = TextStyle(font_size=DEFAULT_FONT_SIZE, italic=True)
    content_width = max_width - indent - bar_margin

    font = get_font(style.font_size, style.bold, style.italic)
    line_height = int(style.font_size * style.line_height_mult)
    padding = max(2, style.font_size // 10)

    measure_img = Image.new("L", (max_width * 2, style.font_size * 2), 255)
    measure_draw = ImageDraw.Draw(measure_img)

    wrapped_lines = wrap_text(measure_draw, text, font, content_width - padding * 2)
    if not wrapped_lines:
        wrapped_lines = [""]

    width = round_up_to_8(max_width)
    height = round_up_to_8(len(wrapped_lines) * line_height + padding * 2)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (bar_width, height - 1)], fill=0)

    for line_idx, line in enumerate(wrapped_lines):
        y = padding + line_idx * line_height
        draw.text((indent, y), line, fill=0, font=font)

    if style.font_size <= 32:
        img = apply_threshold(img, 200)

    return img


def render_list_item(
    text: str,
    bullet: str,
    indent: int = 0,
    max_width: int = PRINTER_WIDTH,
) -> Image.Image:
    indent_px = indent * 16
    bullet_width = 24

    style = TextStyle(font_size=DEFAULT_FONT_SIZE)
    content_width = max_width - indent_px - bullet_width

    font = get_font(style.font_size)
    line_height = int(style.font_size * style.line_height_mult)
    padding = max(2, style.font_size // 10)

    measure_img = Image.new("L", (max_width * 2, style.font_size * 2), 255)
    measure_draw = ImageDraw.Draw(measure_img)

    wrapped_lines = wrap_text(measure_draw, text, font, content_width - padding * 2)
    if not wrapped_lines:
        wrapped_lines = [""]

    width = round_up_to_8(max_width)
    height = round_up_to_8(len(wrapped_lines) * line_height + padding * 2)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    draw.text((indent_px, padding), bullet, fill=0, font=font)

    for line_idx, line in enumerate(wrapped_lines):
        y = padding + line_idx * line_height
        draw.text((indent_px + bullet_width, y), line, fill=0, font=font)

    if style.font_size <= 32:
        img = apply_threshold(img, 200)

    return img


@dataclass
class MarkdownToken:
    type: str
    content: str = ""
    level: int = 0
    children: List["MarkdownToken"] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    ordered: bool = False
    spans: List[StyledSpan] = field(default_factory=list)


def parse_inline_formatting(text: str) -> Tuple[str, List[StyledSpan]]:
    spans = []
    result = text

    patterns = [
        (r"\*\*\*(.+?)\*\*\*", "bold_italic"),
        (r"___(.+?)___", "bold_italic"),
        (r"\*\*(.+?)\*\*", "bold"),
        (r"__(.+?)__", "bold"),
        (r"\*(.+?)\*", "italic"),
        (r"_(.+?)_", "italic"),
        (r"~~(.+?)~~", "strikethrough"),
        (r"`(.+?)`", "code"),
        (r"\[([^\]]+)\]\([^)]+\)", "link"),
    ]

    for pattern, style_type in patterns:
        offset = 0
        for match in re.finditer(pattern, text):
            inner = match.group(1)
            start_in_original = match.start()
            end_in_original = match.end()
            marker_len = len(match.group(0)) - len(inner)

            result_before = result[:start_in_original - offset]
            result_after = result[end_in_original - offset:]
            result = result_before + inner + result_after

            span = StyledSpan(
                start=start_in_original - offset,
                end=start_in_original - offset + len(inner),
            )
            if style_type == "bold":
                span.bold = True
            elif style_type == "italic":
                span.italic = True
            elif style_type == "bold_italic":
                span.bold = True
                span.italic = True
            elif style_type == "strikethrough":
                span.strikethrough = True
            elif style_type == "code":
                span.code = True
            elif style_type == "link":
                span.underline = True

            spans.append(span)
            offset += marker_len

    return result, spans


def parse_markdown(text: str) -> List[MarkdownToken]:
    tokens = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        if re.match(r"^#{1,6}\s", line):
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                content, spans = parse_inline_formatting(match.group(2))
                tokens.append(MarkdownToken(type="heading", content=content, level=level, spans=spans))
            i += 1
            continue

        if re.match(r"^```", line):
            lang = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            tokens.append(MarkdownToken(type="code_block", content="\n".join(code_lines)))
            i += 1
            continue

        if re.match(r"^[-*_]{3,}\s*$", line):
            tokens.append(MarkdownToken(type="hr"))
            i += 1
            continue

        if re.match(r"^>\s*", line):
            quote_lines = []
            while i < len(lines) and (lines[i].startswith(">") or (lines[i].strip() and quote_lines)):
                quote_lines.append(re.sub(r"^>\s*", "", lines[i]))
                i += 1
            content, spans = parse_inline_formatting(" ".join(quote_lines))
            tokens.append(MarkdownToken(type="blockquote", content=content, spans=spans))
            continue

        if re.match(r"^\|.+\|", line):
            table_lines = []
            while i < len(lines) and re.match(r"^\|.+\|", lines[i]):
                table_lines.append(lines[i])
                i += 1

            if len(table_lines) >= 2:
                headers = [cell.strip() for cell in table_lines[0].split("|")[1:-1]]
                rows = []
                for row_line in table_lines[2:]:
                    cells = [cell.strip() for cell in row_line.split("|")[1:-1]]
                    rows.append(cells)
                tokens.append(MarkdownToken(type="table", headers=headers, rows=rows))
            continue

        list_match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
        if list_match:
            indent = len(list_match.group(1)) // 2
            list_items = []
            while i < len(lines):
                item_match = re.match(r"^(\s*)[-*+]\s+(.+)$", lines[i])
                if item_match:
                    item_indent = len(item_match.group(1)) // 2
                    content, spans = parse_inline_formatting(item_match.group(2))
                    list_items.append(MarkdownToken(type="list_item", content=content, level=item_indent, spans=spans))
                    i += 1
                elif lines[i].strip() == "":
                    i += 1
                    break
                else:
                    break
            tokens.append(MarkdownToken(type="list", children=list_items, ordered=False))
            continue

        ordered_match = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
        if ordered_match:
            indent = len(ordered_match.group(1)) // 2
            list_items = []
            item_num = 1
            while i < len(lines):
                item_match = re.match(r"^(\s*)(\d+)\.\s+(.+)$", lines[i])
                if item_match:
                    item_indent = len(item_match.group(1)) // 2
                    content, spans = parse_inline_formatting(item_match.group(3))
                    list_items.append(
                        MarkdownToken(type="list_item", content=content, level=item_indent, spans=spans)
                    )
                    item_num += 1
                    i += 1
                elif lines[i].strip() == "":
                    i += 1
                    break
                else:
                    break
            tokens.append(MarkdownToken(type="list", children=list_items, ordered=True))
            continue

        if line.strip():
            para_lines = []
            while i < len(lines) and lines[i].strip() and not any([
                re.match(r"^#{1,6}\s", lines[i]),
                re.match(r"^```", lines[i]),
                re.match(r"^[-*_]{3,}\s*$", lines[i]),
                re.match(r"^>\s*", lines[i]),
                re.match(r"^\|.+\|", lines[i]),
                re.match(r"^[-*+]\s+", lines[i]),
                re.match(r"^\d+\.\s+", lines[i]),
            ]):
                para_lines.append(lines[i])
                i += 1
            content, spans = parse_inline_formatting(" ".join(para_lines))
            tokens.append(MarkdownToken(type="paragraph", content=content, spans=spans))
            continue

        i += 1

    return tokens


def render_markdown(markdown_text: str, max_width: int = PRINTER_WIDTH) -> List[Image.Image]:
    tokens = parse_markdown(markdown_text)
    images = []

    for token in tokens:
        if token.type == "heading":
            font_size = HEADING_SIZES.get(token.level, DEFAULT_FONT_SIZE)
            style = TextStyle(font_size=font_size, bold=True)
            img = render_text_block(token.content, style, max_width, token.spans if token.spans else None)
            images.append(img)

        elif token.type == "paragraph":
            style = TextStyle(font_size=DEFAULT_FONT_SIZE)
            img = render_text_block(token.content, style, max_width, token.spans if token.spans else None)
            images.append(img)

        elif token.type == "code_block":
            img = render_code_block(token.content, max_width)
            images.append(img)

        elif token.type == "hr":
            img = render_horizontal_rule(max_width)
            images.append(img)

        elif token.type == "blockquote":
            img = render_blockquote(token.content, max_width)
            images.append(img)

        elif token.type == "table":
            if token.headers and token.rows:
                img = render_table(token.headers, token.rows, max_width)
                images.append(img)

        elif token.type == "list":
            for idx, item in enumerate(token.children):
                if token.ordered:
                    bullet = f"{idx + 1}."
                else:
                    bullet = "•"
                img = render_list_item(item.content, bullet, item.level, max_width)
                images.append(img)

    return images


def render_markdown_to_single_image(
    markdown_text: str,
    max_width: int = PRINTER_WIDTH,
    spacing: int = 4,
) -> Image.Image:
    block_images = render_markdown(markdown_text, max_width)

    if not block_images:
        img = Image.new("L", (round_up_to_8(max_width), 8), 255)
        return img

    total_height = sum(img.height for img in block_images) + spacing * (len(block_images) - 1)
    total_height = round_up_to_8(total_height)

    combined = Image.new("L", (round_up_to_8(max_width), total_height), 255)

    y = 0
    for img in block_images:
        combined.paste(img, (0, y))
        y += img.height + spacing

    return combined
