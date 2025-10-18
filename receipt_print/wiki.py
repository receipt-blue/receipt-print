import io
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image

USER_AGENT = os.getenv(
    "RP_WIKI_USER_AGENT",
    "receipt-print/0.1 (+https://github.com/jmpaz/receipt-print)",
)
API_TIMEOUT = float(os.getenv("RP_WIKI_TIMEOUT", "20"))


class WikipediaError(RuntimeError):
    """Raised when a Wikipedia article cannot be loaded."""


@dataclass
class WikiSegment:
    kind: Literal["text", "image"]
    text: Optional[str] = None
    image_url: Optional[str] = None
    caption: Optional[str] = None


@dataclass
class WikiArticle:
    title: str
    url: str
    segments: List[WikiSegment]
    heading: str


def _canonicalize_article_url(lang: str, title: str) -> str:
    normalized = title.replace(" ", "_")
    return f"https://{lang}.wikipedia.org/wiki/{quote(normalized)}"


def _parse_wikipedia_url(raw_url: str) -> tuple[str, str]:
    parsed = urlparse(raw_url)
    if not parsed.scheme:
        raise WikipediaError(f"Invalid URL (missing scheme): {raw_url}")

    host = parsed.netloc.lower()
    if not host.endswith(".wikipedia.org"):
        raise WikipediaError(f"Not a Wikipedia host: {raw_url}")

    parts = host.split(".")
    lang = parts[0] if parts[0] not in {"www", "m"} else "en"

    title: Optional[str] = None
    if parsed.path.startswith("/wiki/"):
        title = parsed.path[len("/wiki/") :]
    elif parsed.path == "/w/index.php":
        qs = parse_qs(parsed.query)
        title = qs.get("title", [""])[0]
    if not title:
        raise WikipediaError(f"Unsupported Wikipedia URL: {raw_url}")

    title = title.strip("/")
    if not title:
        raise WikipediaError(f"Missing article title in URL: {raw_url}")

    return lang, title


def _request_json(lang: str, params: dict) -> dict:
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=API_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise WikipediaError(str(exc)) from exc
    data = resp.json()
    if "error" in data:
        info = data["error"].get("info") or "unknown error"
        raise WikipediaError(info)
    return data


def load_article(url: str) -> WikiArticle:
    lang, title = _parse_wikipedia_url(url)
    data = _request_json(
        lang,
        {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text",
            "formatversion": "2",
        },
    )
    parse = data.get("parse")
    if not parse:
        raise WikipediaError("Malformed response from Wikipedia")
    html = parse.get("text")
    if not html:
        raise WikipediaError("No article content returned")
    if isinstance(html, dict):
        html = html.get("*", "")
    article_title = parse.get("title", title)
    canonical_url = _canonicalize_article_url(lang, article_title)
    segments, heading = _html_to_segments(html, article_title)
    return WikiArticle(
        title=article_title, url=canonical_url, segments=segments, heading=heading
    )


class _ReferenceManager:
    def __init__(self, reference_texts: dict[str, str]):
        self._reference_texts = reference_texts
        self._id_to_label: dict[str, str] = {}
        self._order: List[str] = []

    @staticmethod
    def _is_reference_sup(tag: Tag) -> bool:
        if tag.name != "sup":
            return False
        classes = tag.get("class", [])
        return any("reference" in cls or cls.startswith("mw-ref") for cls in classes)

    def _register_anchor(self, anchor: Tag) -> Optional[str]:
        href = anchor.get("href", "")
        if not href.startswith("#"):
            return None
        ref_id = href[1:]
        label_text = anchor.get_text(strip=True)
        label = label_text.strip("[]") or str(len(self._id_to_label) + 1)
        if ref_id not in self._id_to_label:
            self._id_to_label[ref_id] = label
            self._order.append(ref_id)
        return self._id_to_label[ref_id]

    def marker_for(self, sup: Tag) -> str:
        if not self._is_reference_sup(sup):
            return ""
        markers: List[str] = []
        anchors = [a for a in sup.find_all("a", href=True)]
        if not anchors:
            text = sup.get_text(strip=True)
            if text:
                label = text.strip("[]")
                return f"[^{label}]"
            return ""
        for anchor in anchors:
            label = self._register_anchor(anchor)
            if label:
                markers.append(f"[^{label}]")
        return "".join(markers)

    def references(self) -> List[tuple[str, str]]:
        collected: List[tuple[str, str]] = []
        for ref_id in self._order:
            label = self._id_to_label.get(ref_id)
            text = self._reference_texts.get(ref_id)
            if label and text:
                collected.append((label, text))
        return collected


def _collect_reference_texts(root: Tag) -> dict[str, str]:
    ref_texts: dict[str, str] = {}
    for ol in root.select("ol.references"):
        for li in ol.find_all("li", recursive=False):
            ref_id = li.get("id")
            if not ref_id:
                continue
            backlink = li.find("span", class_="mw-cite-backlink")
            if backlink:
                backlink.decompose()
            text = _render_inline(li, None)
            cleaned = _normalize_block(text)
            if cleaned:
                ref_texts[ref_id] = cleaned
        ol.decompose()
    return ref_texts


_SKIP_CLASSES = {
    "mw-editsection",
    "noprint",
    "metadata",
    "refbegin",
    "refend",
    "navbox",
    "vertical-navbox",
    "navbox-inner",
    "toc",
    "sisterproject",
    "hatnote",
    "plainlist",
    "shortdescription",
    "succession-box",
    "sistersitebox",
}

_SKIP_HEADINGS = {
    "references",
    "notes",
    "citations",
    "further reading",
    "external links",
    "sources",
    "bibliography",
}


def _should_skip_element(tag: Tag) -> bool:
    classes = tag.get("class", [])
    if any(cls in _SKIP_CLASSES for cls in classes):
        return True
    if tag.name == "table":
        return True
    if tag.name == "div" and tag.get("role") in {"note", "navigation"}:
        return True
    return False


def _normalize_block(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")


def _limit_heading_lines(
    text: str,
    remaining: Optional[int],
    exhausted: bool,
) -> Tuple[str, Optional[int], bool]:
    if remaining is None or exhausted:
        return text, remaining, exhausted
    rem = max(remaining, 0)
    lines_out: List[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if _HEADING_LINE_RE.match(stripped):
            if rem <= 0:
                exhausted = True
                break
            rem -= 1
            lines_out.append(line)
        else:
            lines_out.append(line)
    return "\n".join(lines_out), rem, exhausted


def _render_inline(node, ref_manager: Optional[_ReferenceManager]) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""

    if node.name == "sup":
        if ref_manager:
            marker = ref_manager.marker_for(node)
            if marker:
                return marker
        content = "".join(_render_inline(child, ref_manager) for child in node.children)
        return f"^{content}" if content else ""
    if node.name == "sub":
        return "".join(_render_inline(child, ref_manager) for child in node.children)
    if node.name == "br":
        return "\n"
    if node.name in {"img", "script", "style"}:
        return ""

    if node.name in {"ul", "ol"}:
        return _render_block(node, ref_manager)

    parts: List[str] = []
    for child in node.children:
        parts.append(_render_inline(child, ref_manager))
    return "".join(parts)


def _render_block(tag: Tag, ref_manager: Optional[_ReferenceManager]) -> str:
    name = tag.name
    if name == "p":
        text = _render_inline(tag, ref_manager)
        return _normalize_block(text)
    if name == "ul":
        lines = []
        for li in tag.find_all("li", recursive=False):
            content = _normalize_block(_render_inline(li, ref_manager))
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)
    if name == "ol":
        lines = []
        for idx, li in enumerate(tag.find_all("li", recursive=False), start=1):
            content = _normalize_block(_render_inline(li, ref_manager))
            if content:
                lines.append(f"{idx}. {content}")
        return "\n".join(lines)
    if name == "dl":
        lines = []
        terms = list(tag.find_all("dt", recursive=False))
        for term in terms:
            desc = term.find_next_sibling("dd")
            term_text = _normalize_block(_render_inline(term, ref_manager))
            desc_text = _normalize_block(_render_inline(desc, ref_manager)) if desc else ""
            if term_text and desc_text:
                lines.append(f"{term_text}: {desc_text}")
            elif term_text:
                lines.append(term_text)
        return "\n".join(lines)
    if name == "blockquote":
        text = _normalize_block(_render_inline(tag, ref_manager))
        if not text:
            return ""
        return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
    if name == "pre":
        return tag.get_text()
    text = _render_inline(tag, ref_manager)
    return _normalize_block(text)


def _figure_to_segment(fig: Tag) -> Optional[WikiSegment]:
    img = fig.find("img")
    if not img:
        return None
    src = img.get("src") or ""
    srcset = img.get("srcset") or ""
    url = src
    if srcset:
        last = srcset.split(",")[-1].strip()
        if " " in last:
            url = last.split(" ")[0]
        else:
            url = last
    if url.startswith("//"):
        url = "https:" + url
    caption_tag = fig.find("figcaption")
    caption = _normalize_block(_render_inline(caption_tag, None)) if caption_tag else None
    return WikiSegment(kind="image", image_url=url, caption=caption or None)


def _html_to_segments(html: str, title: str) -> Tuple[List[WikiSegment], str]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", class_="mw-parser-output")
    if not root:
        raise WikipediaError("Unable to locate article content")

    for span in root.select("span.mw-editsection"):
        span.decompose()

    ref_texts = _collect_reference_texts(root)
    ref_manager = _ReferenceManager(ref_texts)

    segments: List[WikiSegment] = []
    current_blocks: List[str] = []
    heading_text = title
    heading_set = False

    def flush_blocks():
        if not current_blocks:
            return
        block_text = "\n\n".join(filter(None, current_blocks)).strip()
        if block_text:
            segments.append(WikiSegment(kind="text", text=block_text))
        current_blocks.clear()

    skip_level: Optional[int] = None

    def process_node(node: Tag):
        nonlocal skip_level
        if _should_skip_element(node):
            return
        if node.name in {"style", "script"}:
            return
        if node.name and node.name.startswith("h") and len(node.name) == 2 and node.name[1].isdigit():
            level = int(node.name[1])
            heading_line = node.get_text(" ", strip=True)
            lower = heading_line.lower()
            if lower in _SKIP_HEADINGS:
                flush_blocks()
                skip_level = level
                return
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is not None:
                return
            if level == 1 and not heading_set:
                flush_blocks()
                if heading_line:
                    heading_text = heading_line
                heading_set = True
                return
            flush_blocks()
            hashes = "#" * level
            current_blocks.append(f"{hashes} {heading_line}")
            return
        if skip_level is not None:
            if node.name and node.name.startswith("h") and node.name[1].isdigit():
                level = int(node.name[1])
                if level <= skip_level:
                    skip_level = None
            if skip_level is not None:
                return
        if node.name == "figure":
            flush_blocks()
            seg = _figure_to_segment(node)
            if seg:
                segments.append(seg)
            return
        if node.name in {"p", "ul", "ol", "dl", "blockquote", "pre"}:
            block = _render_block(node, ref_manager)
            if block:
                current_blocks.append(block)
            return
        if node.name in {"div", "section"}:
            for child in node.children:
                if isinstance(child, Tag):
                    process_node(child)
            return
        text = _render_block(node, ref_manager)
        if text:
            current_blocks.append(text)

    for child in root.children:
        if isinstance(child, Tag):
            process_node(child)

    flush_blocks()

    references = ref_manager.references()
    if references:
        lines = ["## References"]
        for label, text in references:
            lines.append(f"[^{label}]: {text}")
        ref_block = "\n".join(lines)
        segments.append(WikiSegment(kind="text", text=ref_block))

    return segments, heading_text


def _load_image(url: str) -> Optional[Image.Image]:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=API_TIMEOUT)
    resp.raise_for_status()
    data = resp.content
    return Image.open(io.BytesIO(data)).convert("RGB")


def _max_printer_width(printer) -> int:
    default = 576
    try:
        profile = getattr(printer, "profile", None)
        if profile:
            media = profile.profile_data.get("media", {})
            width = media.get("width", {}).get("pixels")
            if width and width != "Unknown":
                return int(width)
    except Exception:
        pass
    return default


def _resize_for_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    ratio = max_width / image.width
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _print_image(printer, image: Image.Image, caption: Optional[str]) -> None:
    from .printer import sanitize_output

    printer.set(align="center")
    printer.image(image, impl="bitImageColumn")
    if caption:
        printer.text(sanitize_output(caption) + "\n")
    printer.text("\n")
    printer.set(align="left")


def _print_text(printer, text: str) -> None:
    from .printer import sanitize_output

    printer.set(normal_textsize=True, double_width=False, double_height=False)
    sanitized = sanitize_output(text)
    printer.set(
        align="left",
        font="a",
        bold=False,
        width=1,
        height=1,
        double_height=False,
        double_width=False,
    )
    lines = sanitized.splitlines()
    if not lines:
        printer.text("\n")
        return
    for line in lines:
        if _HEADING_LINE_RE.match(line):
            printer.set(bold=True, font="a", width=1, height=1, normal_textsize=True)
            printer.text(line + "\n")
            printer.set(
                bold=False,
                font="a",
                width=1,
                height=1,
                normal_textsize=True,
                double_height=False,
                double_width=False,
            )
        else:
            printer.text(line + "\n")
    printer.text("\n")


def _print_qr(printer, url: str, position: str) -> None:
    align = "left" if position.endswith("left") else "right"
    printer.set(align=align, normal_textsize=True, double_width=False, double_height=False)
    printer.qr(url, size=4)
    printer.set(align="left", normal_textsize=True, double_width=False, double_height=False)


def print_articles(
    articles: Iterable[WikiArticle],
    qr: bool,
    qr_position: str,
    max_headings: Optional[int] = None,
) -> None:
    from .printer import (
        CHAR_WIDTH,
        DOTS_PER_LINE,
        MAX_LINES,
        connect_printer,
        count_lines,
        sanitize_output,
    )

    article_list = list(articles)
    if not article_list:
        return

    qr_pos = qr_position.lower()
    printer = connect_printer()
    try:
        max_width = _max_printer_width(printer)
        plan: List[tuple] = []
        total_lines = 0

        for idx, article in enumerate(article_list):
            heading_remaining = max_headings if max_headings is not None else None
            heading_limit_reached = False
            if idx:
                plan.append(("blank",))
                total_lines += 1

            heading_value = article.heading.strip() if article.heading else article.title
            heading_text = sanitize_output(heading_value)
            plan.append(("header", heading_text))
            total_lines += count_lines(heading_text, CHAR_WIDTH)
            if heading_remaining is not None and heading_remaining > 0:
                heading_remaining = max(heading_remaining - 1, 0)

            top = qr and qr_pos.startswith("top")
            bottom = qr and qr_pos.startswith("bottom")
            if top:
                plan.append(("qr", article.url, qr_pos))
                plan.append(("blank",))
                total_lines += 1

            for segment in article.segments:
                if heading_limit_reached:
                    break
                if segment.kind == "text" and segment.text:
                    segment_text = segment.text
                    limited_text, heading_remaining, exhausted = _limit_heading_lines(
                        segment_text, heading_remaining, heading_limit_reached
                    )
                    if exhausted:
                        heading_limit_reached = True
                    sanitized_text = sanitize_output(limited_text)
                    if sanitized_text:
                        plan.append(("text", sanitized_text))
                        total_lines += count_lines(sanitized_text, CHAR_WIDTH) + 1
                elif segment.kind == "image" and segment.image_url:
                    try:
                        image = _load_image(segment.image_url)
                    except Exception:
                        continue
                    image = _resize_for_width(image, max_width)
                    caption_text = segment.caption
                    sanitized_caption = sanitize_output(caption_text) if caption_text else None
                    image_lines = math.ceil(image.height / DOTS_PER_LINE)
                    caption_lines = 1
                    if sanitized_caption:
                        caption_lines += count_lines(sanitized_caption, CHAR_WIDTH)
                    total_lines += image_lines + caption_lines
                    plan.append(("image", image, caption_text))

            if bottom:
                plan.append(("blank",))
                total_lines += 1
                if heading_limit_reached:
                    plan.append(("ellipsis", "..."))
                    total_lines += 1
                    plan.append(("blank",))
                    total_lines += 1
                plan.append(("qr", article.url, qr_pos))
            elif heading_limit_reached:
                plan.append(("blank",))
                total_lines += 1
                plan.append(("ellipsis", "..."))
                total_lines += 1
                plan.append(("blank",))
                total_lines += 1

        if total_lines > MAX_LINES:
            try:
                with open("/dev/tty") as tty:
                    sys.stdout.write(
                        f"Warning: {total_lines} lines > limit {MAX_LINES}. Continue? [y/N] "
                    )
                    sys.stdout.flush()
                    if tty.readline().strip().lower() not in ("y", "yes"):
                        sys.exit(1)
            except Exception:
                sys.stderr.write("No TTY available; aborting.\n")
                sys.exit(1)

        for action in plan:
            kind = action[0]
            if kind == "blank":
                printer.text("\n")
            elif kind == "header":
                printer.set(
                    align="left",
                    font="b",
                    bold=True,
                    normal_textsize=False,
                    double_height=True,
                    double_width=False,
                )
                printer.text(action[1] + "\n\n")
                printer.set(
                    align="left",
                    font="a",
                    bold=False,
                    width=1,
                    height=1,
                    double_height=False,
                    double_width=False,
                    normal_textsize=True,
                )
            elif kind == "qr":
                _print_qr(printer, action[1], action[2])
            elif kind == "text":
                _print_text(printer, action[1])
            elif kind == "ellipsis":
                printer.set(align="center", normal_textsize=True, bold=True)
                printer.text(action[1] + "\n")
                printer.set(
                    align="left",
                    font="a",
                    bold=False,
                    width=1,
                    height=1,
                    double_height=False,
                    double_width=False,
                    normal_textsize=True,
                )
            elif kind == "image":
                _print_image(printer, action[1], action[2])

        printer.cut()
    finally:
        printer.close()
