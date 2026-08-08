from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from .markdown_render import MarkdownToken, StyledSpan, parse_markdown


def _mathematical_style(char: str) -> Optional[str]:
    name = unicodedata.name(char, "")
    if "MATHEMATICAL" not in name:
        return None
    bold = "BOLD" in name
    italic = "ITALIC" in name
    if bold and italic:
        return "bold_italic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return None


def _styled_unicode_markdown(value: str) -> str:
    markers = {
        "bold": "**",
        "italic": "*",
        "bold_italic": "***",
    }
    rendered: list[str] = []
    index = 0
    while index < len(value):
        style = _mathematical_style(value[index])
        if not style:
            rendered.append(unicodedata.normalize("NFKC", value[index]))
            index += 1
            continue

        segment: list[str] = []
        cursor = index
        while cursor < len(value):
            if _mathematical_style(value[cursor]) == style:
                segment.append(unicodedata.normalize("NFKC", value[cursor]))
                cursor += 1
                continue
            if value[cursor].isspace():
                next_content = cursor
                while next_content < len(value) and value[next_content].isspace():
                    next_content += 1
                if (
                    next_content < len(value)
                    and _mathematical_style(value[next_content]) == style
                ):
                    segment.append(value[cursor:next_content])
                    cursor = next_content
                    continue
            break

        marker = markers[style]
        rendered.extend((marker, "".join(segment), marker))
        index = cursor

    return "".join(rendered)


def _arena_markdown_source(value: str) -> str:
    prepared = _styled_unicode_markdown(value).replace("\r\n", "\n").replace("\r", "\n")
    prepared = re.sub(r"<br\s*/?>", "\n", prepared, flags=re.IGNORECASE)
    return html.unescape(prepared.replace("\u2060", ""))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value).strip()
    if isinstance(value, dict):
        for key in ("markdown", "plain", "html"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return unicodedata.normalize("NFKC", candidate).strip()
    return ""


def _markdown_text(value: Any) -> str:
    if isinstance(value, str):
        return _styled_unicode_markdown(value).strip()
    if isinstance(value, dict):
        for key in ("markdown", "plain", "html"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _styled_unicode_markdown(candidate).strip()
    return ""


def _href(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    if isinstance(value, dict):
        for key in ("url", "href", "src"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.startswith(
                ("http://", "https://")
            ):
                return candidate
    return None


def _person_name(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in ("full_name", "name", "username", "slug"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _connection(block: dict[str, Any]) -> dict[str, Any]:
    value = block.get("connection")
    return value if isinstance(value, dict) else {}


def _block_kind(block: dict[str, Any]) -> str:
    return str(
        block.get("type")
        or block.get("class")
        or block.get("base_type")
        or block.get("base_class")
        or "Block"
    ).lower()


def _preview_url(block: dict[str, Any]) -> Optional[str]:
    image = block.get("image")
    if isinstance(image, dict):
        for key in ("display", "large", "original", "medium", "small", "thumb"):
            candidate = _href(image.get(key))
            if candidate:
                return candidate
        candidate = _href(image)
        if candidate:
            return candidate
    return _href(block.get("thumb_url"))


def _attachment_url(block: dict[str, Any]) -> Optional[str]:
    for key in ("attachment", "file"):
        candidate = _href(block.get(key))
        if candidate:
            return candidate
    return None


def _source_url(block: dict[str, Any]) -> Optional[str]:
    for key in ("source", "source_url"):
        candidate = _href(block.get(key))
        if candidate:
            return candidate
    return None


def _channel_url(block: dict[str, Any]) -> Optional[str]:
    self_link = _href(block.get("self"))
    if self_link and "www.are.na" in urlparse(self_link).netloc:
        return self_link
    links = block.get("_links")
    if isinstance(links, dict):
        candidate = _href(links.get("self"))
        if candidate and "www.are.na" in urlparse(candidate).netloc:
            return candidate
    slug = block.get("slug")
    owner = block.get("owner") or block.get("user")
    owner_slug = owner.get("slug") if isinstance(owner, dict) else None
    if slug and owner_slug:
        return f"https://www.are.na/{owner_slug}/{slug}"
    if slug:
        return f"https://www.are.na/channel/{slug}"
    return None


@dataclass(frozen=True)
class ArenaPlacement:
    position: Optional[int]
    pinned: bool
    connected_at: Optional[str]
    connected_by: Optional[str]


@dataclass(frozen=True)
class ArenaBlock:
    id: str
    kind: str
    title: str
    content: str
    description: str
    state: str
    creator: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    canonical_url: str
    external_url: Optional[str]
    attachment_url: Optional[str]
    preview_url: Optional[str]
    linked_channel_url: Optional[str]
    placement: ArenaPlacement

    @classmethod
    def from_api(cls, block: dict[str, Any]) -> "ArenaBlock":
        block_id_value = block.get("id")
        block_id = "unknown" if block_id_value is None else str(block_id_value)
        connection = _connection(block)
        kind = _block_kind(block)
        return cls(
            id=block_id,
            kind=kind,
            title=_text(block.get("title") or block.get("generated_title")),
            content=_markdown_text(block.get("content")),
            description=_markdown_text(block.get("description")),
            state=str(block.get("state") or "available").lower(),
            creator=_person_name(block.get("user") or block.get("owner")),
            created_at=block.get("created_at"),
            updated_at=block.get("updated_at"),
            canonical_url=f"https://www.are.na/block/{block_id}",
            external_url=_source_url(block),
            attachment_url=_attachment_url(block),
            preview_url=_preview_url(block),
            linked_channel_url=_channel_url(block) if kind == "channel" else None,
            placement=ArenaPlacement(
                position=connection.get("position"),
                pinned=bool(connection.get("pinned")),
                connected_at=connection.get("connected_at"),
                connected_by=_person_name(connection.get("connected_by")),
            ),
        )


@dataclass(frozen=True)
class ArenaChannel:
    id: str
    slug: str
    title: str
    owner: str
    description: str
    canonical_url: str
    cover_url: Optional[str]
    total_count: int
    fetched_at: str
    blocks: tuple[ArenaBlock, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArenaChannel":
        blocks = []
        for item in value.get("blocks", []):
            placement = ArenaPlacement(**item["placement"])
            blocks.append(
                ArenaBlock(**{**{"updated_at": None}, **item, "placement": placement})
            )
        return cls(**{**value, "blocks": tuple(blocks)})


def normalize_channel(
    meta: dict[str, Any],
    raw_blocks: Iterable[dict[str, Any]],
    canonical_url: str,
    *,
    total_count: Optional[int] = None,
    fetched_at: Optional[str] = None,
) -> ArenaChannel:
    owner_value = meta.get("owner") or meta.get("user")
    counts = meta.get("counts") if isinstance(meta.get("counts"), dict) else {}
    count = total_count
    if count is None:
        count = counts.get("contents") or meta.get("length") or 0
    return ArenaChannel(
        id=str(meta.get("id") or meta.get("slug") or "unknown"),
        slug=str(meta.get("slug") or "unknown"),
        title=_text(meta.get("title")) or "Untitled channel",
        owner=_person_name(owner_value) or "Are.na",
        description=_markdown_text(meta.get("description")),
        canonical_url=canonical_url,
        cover_url=_href(
            meta.get("cover") or meta.get("image") or meta.get("thumbnail")
        ),
        total_count=int(count),
        fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
        blocks=tuple(ArenaBlock.from_api(block) for block in raw_blocks),
    )


def _same_destination(first: str, second: str) -> bool:
    def normalized(value: str) -> tuple[str, str]:
        parsed = urlparse(value)
        return parsed.netloc.lower(), parsed.path.rstrip("/")

    return normalized(first) == normalized(second)


def _readable_url(url: str, *, limit: int = 54) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path.rstrip("/")
    value = f"{host}{path}"
    return value if len(value) <= limit else f"{value[: limit - 3]}..."


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.removeprefix("www.") or _readable_url(url)


def block_destinations(block: ArenaBlock) -> list[dict[str, Any]]:
    if block.kind == "channel" and block.linked_channel_url:
        return [
            {
                "role": "primary",
                "kind": "channel",
                "url": block.linked_channel_url,
                "title": "source",
                "detail": _readable_url(block.linked_channel_url, limit=28),
            },
            {
                "role": "context",
                "kind": "arena",
                "url": block.canonical_url,
                "title": "block",
                "detail": block.id,
            },
        ]

    direct = block.external_url or block.attachment_url
    if direct and block.state not in {"failure", "deleted", "unavailable"}:
        destinations = [
            {
                "role": "primary",
                "kind": "source" if block.external_url else "attachment",
                "url": direct,
                "title": "source",
                "detail": _domain(direct),
            }
        ]
        if not _same_destination(direct, block.canonical_url):
            destinations.append(
                {
                    "role": "context",
                    "kind": "arena",
                    "url": block.canonical_url,
                    "title": "block",
                    "detail": block.id,
                }
            )
        return destinations

    return [
        {
            "role": "primary",
            "kind": "arena",
            "url": block.canonical_url,
            "title": "block",
            "detail": block.id,
        }
    ]


def _ref(block: ArenaBlock, note: Optional[str] = None) -> dict[str, str]:
    ref = {"source": "arena", "id": block.id}
    if note:
        ref["note"] = note
    return ref


def _text_block(
    text: str,
    *,
    bold: bool = False,
    font: Optional[str] = None,
    invert: bool = False,
    align: Optional[str] = None,
    marker: Optional[str] = None,
    ref: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    run: dict[str, Any] = {"text": unicodedata.normalize("NFKC", text)}
    if bold:
        run["bold"] = True
    if font:
        run["font"] = font
    if invert:
        run["invert"] = True
    block: dict[str, Any] = {"type": "text", "runs": [run], "wrap": "word"}
    if align:
        block["align"] = align
    if marker:
        block["marker"] = marker
    if ref:
        block["ref"] = ref
    return block


def _runs_block(
    runs: list[dict[str, Any]],
    *,
    align: Optional[str] = None,
    ref: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    normalized_runs = [
        {
            **run,
            "text": unicodedata.normalize("NFKC", str(run.get("text", ""))),
        }
        for run in runs
    ]
    block: dict[str, Any] = {
        "type": "text",
        "runs": normalized_runs,
        "wrap": "word",
    }
    if align:
        block["align"] = align
    if ref:
        block["ref"] = ref
    return block


def _span_runs(
    text: str,
    spans: list[StyledSpan],
    **base_style: Any,
) -> list[dict[str, Any]]:
    boundaries = {0, len(text)}
    for span in spans:
        boundaries.add(max(0, min(len(text), span.start)))
        boundaries.add(max(0, min(len(text), span.end)))
    points = sorted(boundaries)
    runs: list[dict[str, Any]] = []
    for start, end in zip(points, points[1:]):
        if start == end:
            continue
        active = [span for span in spans if span.start <= start and span.end >= end]
        run: dict[str, Any] = {"text": text[start:end], **base_style}
        if any(span.bold for span in active):
            run["bold"] = True
        if any(span.italic for span in active):
            run["italic"] = True
        if any(span.underline for span in active):
            run["bold"] = True
        if any(span.code for span in active):
            run["font"] = "b"
        runs.append(run)
    return runs or [{"text": text, **base_style}]


def _markdown_token_runs(token: MarkdownToken) -> list[dict[str, Any]]:
    if token.type == "heading":
        return _span_runs(token.content, token.spans, bold=True)
    if token.type == "paragraph":
        return _span_runs(token.content, token.spans)
    if token.type == "blockquote":
        return _span_runs(token.content, token.spans, quote=True)
    if token.type == "code_block":
        return [{"text": token.content, "font": "b"}]
    if token.type == "hr":
        return [{"text": "— — —", "font": "b"}]
    if token.type == "table":
        runs: list[dict[str, Any]] = []
        if token.headers:
            runs.append({"text": " | ".join(token.headers), "font": "b", "bold": True})
        for row in token.rows:
            if runs:
                runs.append({"text": "\n"})
            runs.append({"text": " | ".join(row), "font": "b"})
        return runs
    return []


def _markdown_runs(text: str) -> list[dict[str, Any]]:
    semantic_text = _arena_markdown_source(text)
    tokens = parse_markdown(semantic_text, preserve_line_breaks=True)
    runs: list[dict[str, Any]] = []
    for token_index, token in enumerate(tokens):
        if token.type == "list":
            for item_index, item in enumerate(token.children):
                if item_index:
                    runs.append({"text": "\n"})
                if item.task_state == "unchecked":
                    marker = "☐"
                elif item.task_state == "checked":
                    marker = "☑"
                elif item.task_state == "partial":
                    marker = "▣"
                elif token.ordered:
                    marker = f"{item_index + 1}."
                else:
                    marker = "•"
                runs.append({"text": f"{'  ' * item.level}{marker} ", "bold": True})
                runs.extend(_span_runs(item.content, item.spans))
        else:
            runs.extend(_markdown_token_runs(token))
        if token_index + 1 < len(tokens):
            runs.append({"text": "\n\n"})
    return runs or [{"text": unicodedata.normalize("NFKC", text)}]


def _markdown_text_block(
    text: str,
    *,
    ref: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    return _runs_block(_markdown_runs(text), ref=ref)


def _format_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value[:10]
    return f"{parsed.day} {parsed.strftime('%b %Y')}"


def _metadata_line(
    label: str,
    value: str,
    *,
    clickable: bool = False,
    ref: dict[str, str],
) -> dict[str, Any]:
    value_run: dict[str, Any] = {
        "text": value,
        "font": "b",
        "dock": "right",
    }
    if clickable:
        value_run["bold"] = True
    return _runs_block(
        [
            {"text": label, "font": "b"},
            value_run,
        ],
        ref=ref,
    )


def _source_name(block: ArenaBlock) -> str:
    direct = block.external_url or block.attachment_url
    if direct:
        return _domain(direct)
    return "Are.na"


def _metadata_blocks(block: ArenaBlock, ref: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    added = _format_date(block.placement.connected_at or block.created_at)
    modified = _format_date(block.updated_at)
    if added:
        rows.append(_metadata_line("Added", added, ref=ref))
    if modified:
        rows.append(_metadata_line("Modified", modified, ref=ref))
    if block.creator:
        rows.append(_metadata_line("By", block.creator, clickable=True, ref=ref))
    if block.placement.connected_by and block.placement.connected_by != block.creator:
        rows.append(
            _metadata_line(
                "Added by",
                block.placement.connected_by,
                clickable=True,
                ref=ref,
            )
        )
    rows.append(
        _metadata_line("Source", _source_name(block), clickable=True, ref=ref)
    )
    return rows


def _content_blocks(block: ArenaBlock, media: dict[str, str]) -> list[dict[str, Any]]:
    ref = _ref(block, "pinned connection" if block.placement.pinned else None)
    content: list[dict[str, Any]] = []
    if block.title or block.placement.pinned:
        content.append(
            _text_block(
                block.title or " ",
                bold=True,
                align="center",
                marker="pin" if block.placement.pinned else None,
                ref=ref,
            )
        )

    data_uri = media.get(block.id)
    if data_uri:
        content.append(
            {
                "type": "image",
                "source": data_uri,
                "dither": "atkinson",
                "width": "full",
                "spacing": {"beforeDots": 8, "afterDots": 8},
                "ref": ref,
            }
        )

    body = block.content or block.description
    if body:
        content.append(_markdown_text_block(body, ref=ref))

    content.extend(_metadata_blocks(block, ref))
    if block.state in {"failure", "deleted", "unavailable"}:
        content.append(
            _text_block("Source unavailable; Are.na copy retained", font="b", ref=ref)
        )
    return content or [_text_block(f"Block {block.id}", ref=ref)]


def _flow_lead_blocks(block: ArenaBlock, media: dict[str, str]) -> list[dict[str, Any]]:
    ref = _ref(block, "pinned connection" if block.placement.pinned else None)
    lead: list[dict[str, Any]] = [{"type": "rule", "weight": "light", "ref": ref}]
    if block.title or block.placement.pinned:
        lead.append(
            _text_block(
                block.title or " ",
                bold=True,
                align="center",
                marker="pin" if block.placement.pinned else None,
                ref=ref,
            )
        )
    data_uri = media.get(block.id)
    if data_uri:
        lead.append(
            {
                "type": "image",
                "source": data_uri,
                "dither": "atkinson",
                "width": "full",
                "spacing": {"beforeDots": 8, "afterDots": 16},
                "ref": ref,
            }
        )
    if block.kind == "text" and block.content:
        lead.append(_markdown_text_block(block.content, ref=ref))
        lead.append({"type": "feed", "dots": 8, "ref": ref})
    return lead


def _flow_body_runs(block: ArenaBlock) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    narrative = block.description
    if block.kind != "text" and not narrative:
        narrative = block.content
    if narrative:
        runs.extend([*_markdown_runs(narrative), {"text": "\n\n"}])

    rows: list[tuple[str, str, bool]] = []
    added = _format_date(block.placement.connected_at or block.created_at)
    modified = _format_date(block.updated_at)
    if added:
        rows.append(("Added", added, False))
    if modified:
        rows.append(("Modified", modified, False))
    if block.creator:
        rows.append(("By", block.creator, True))
    if block.placement.connected_by and block.placement.connected_by != block.creator:
        rows.append(("Added by", block.placement.connected_by, True))
    rows.append(("Source", _source_name(block), True))
    if block.state in {"failure", "deleted", "unavailable"}:
        rows.append(("State", "Source unavailable; Are.na copy retained", False))

    for index, (label, value, clickable) in enumerate(rows):
        value_run: dict[str, Any] = {
            "text": value,
            "font": "b",
            "dock": "right",
        }
        if clickable:
            value_run["bold"] = True
        runs.extend([{"text": label, "font": "b"}, value_run])
        if index + 1 < len(rows):
            runs.append({"text": "\n", "font": "b"})
    return runs


def _legacy_content_blocks(
    block: ArenaBlock, media: dict[str, str]
) -> list[dict[str, Any]]:
    ref = _ref(block, "pinned connection" if block.placement.pinned else None)
    content: list[dict[str, Any]] = []
    if block.title or block.placement.pinned:
        content.append(
            _text_block(
                block.title or " ",
                bold=True,
                align="center",
                marker="pin" if block.placement.pinned else None,
                ref=ref,
            )
        )
    if media.get(block.id):
        content.append(
            {
                "type": "image",
                "source": media[block.id],
                "dither": "atkinson",
                "width": "full",
                "spacing": {"beforeDots": 8, "afterDots": 8},
                "ref": ref,
            }
        )
    body = block.content or block.description
    if body:
        content.append(_markdown_text_block(body, ref=ref))
    if block.creator:
        content.append(_text_block(f"by {block.creator}", font="b", ref=ref))
    return content or [_text_block(f"Block {block.id}", ref=ref)]


def _destination_caption(destination: dict[str, Any]) -> list[dict[str, Any]]:
    if destination["kind"] == "arena":
        return [
            {"text": "block ", "font": "b", "bold": True},
            {"text": destination["detail"], "font": "b"},
        ]
    return [{"text": "source", "font": "b", "bold": True}]


def _qr_block(
    destination: dict[str, Any],
    *,
    placement: str = "block",
    purpose: Optional[str] = None,
    size: Optional[int] = None,
    ref: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "qr",
        "payload": destination["url"],
        "size": size or 4,
        "placement": placement,
        "ecc": "M",
        "caption": _destination_caption(destination),
    }
    if purpose:
        block["purpose"] = purpose
    if ref:
        block["ref"] = ref
    return block


def _flow_block(
    block: ArenaBlock,
    media: dict[str, str],
) -> dict[str, Any]:
    ref = _ref(block, "pinned connection" if block.placement.pinned else None)
    return {
        "type": "qr-flow",
        "side": "right",
        "lead": _flow_lead_blocks(block, media),
        "body": _runs_block(_flow_body_runs(block), ref=ref),
        "destinations": [
            {
                "role": destination["role"],
                "payload": destination["url"],
                "ecc": "M",
                "caption": _destination_caption(destination),
                "ref": _ref(block, destination["kind"]),
            }
            for destination in block_destinations(block)[:2]
        ],
        "ref": ref,
    }


def _channel_header_flow(channel: ArenaChannel) -> dict[str, Any]:
    ref = {"source": "arena", "id": channel.id, "note": "channel artifact"}
    runs: list[dict[str, Any]] = [
        {
            "text": f"{channel.owner} / {channel.title}",
            "bold": True,
            "size": {"w": 1.12, "h": 1.12},
        }
    ]
    return {
        "type": "qr-flow",
        "side": "left",
        "body": _runs_block(runs, ref=ref),
        "destinations": [
            {
                "role": "primary",
                "payload": channel.canonical_url,
                "ecc": "M",
                "ref": ref,
            }
        ],
        "ref": ref,
    }


VARIANTS = (
    "paired",
    "column",
    "minimal",
)


def compose_channel_document(
    channel: ArenaChannel,
    variant: str,
    *,
    media: Optional[dict[str, str]] = None,
    channel_qr: Optional[bool] = None,
    selection: str = "full",
    cut: Optional[str] = None,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Are.na layout variant: {variant}")
    media = media or {}
    blocks = list(channel.blocks)
    rendered: list[dict[str, Any]] = []
    selected = len(blocks)
    include_channel_qr = variant != "minimal" if channel_qr is None else channel_qr
    if include_channel_qr:
        rendered.append(_channel_header_flow(channel))
    else:
        rendered.append(
            _runs_block(
                [
                    {
                        "text": f"{channel.owner} / {channel.title}",
                        "bold": True,
                        "size": {"w": 1.12, "h": 1.12},
                    }
                ],
                ref={"source": "arena", "id": channel.id, "note": "channel artifact"},
            )
        )
    if channel.description:
        rendered.append(
            _markdown_text_block(
                channel.description,
                ref={
                    "source": "arena",
                    "id": channel.id,
                    "note": "channel description",
                },
            )
        )
    if media.get("__channel_cover__"):
        rendered.append(
            {
                "type": "image",
                "source": media["__channel_cover__"],
                "dither": "atkinson",
                "width": "full",
                "spacing": {"beforeDots": 8, "afterDots": 16},
                "ref": {"source": "arena", "id": channel.id, "note": "channel cover"},
            }
        )
    for block in blocks:
        if variant == "column":
            rendered.append(_flow_block(block, media))
        elif variant == "paired":
            rendered.append({"type": "rule", "weight": "light", "ref": _ref(block)})
            rendered.extend(_content_blocks(block, media))
            rendered.append({"type": "feed", "lines": 1, "ref": _ref(block)})
            rendered.extend(
                _qr_block(
                    destination,
                    placement="tile",
                    ref=_ref(block, destination["kind"]),
                )
                for destination in block_destinations(block)[:2]
            )
        else:
            rendered.append({"type": "rule", "weight": "light", "ref": _ref(block)})
            rendered.extend(_legacy_content_blocks(block, media))
        rendered.append({"type": "feed", "lines": 1})

    if cut:
        if cut not in {"full", "partial"}:
            raise ValueError(f"Unknown cut kind: {cut}")
        rendered.append({"type": "cut", "kind": cut})

    refs = [{"source": "arena", "id": block.id} for block in blocks]
    document: dict[str, Any] = {
        "format": "document/1",
        "class": "arena-channel",
        "strategy": "raster",
        "blocks": rendered,
        "provenance": {
            "primary": {"source": "arena", "id": channel.id},
            "refs": refs,
            "composed-at": channel.fetched_at,
            "composed-by": "receipt-print",
        },
        "realization": {
            "mode": "arena-channel",
            "params": {
                "layout": variant,
                "order": (
                    "pinned_first_position_desc"
                    if selection == "random"
                    else "position_desc"
                ),
                "selection": selection,
                "count": selected,
                "channelQr": include_channel_qr,
            },
            "overrides": {},
        },
    }
    return document


def expected_qr_payloads(document: dict[str, Any]) -> list[str]:
    payloads: list[str] = []

    def visit(block: dict[str, Any]) -> None:
        if block.get("type") == "qr":
            payloads.append(block["payload"])
        if block.get("type") in {"qr-rail", "qr-flow"}:
            payloads.extend(item["payload"] for item in block.get("destinations", []))
            for child in block.get("content", []):
                visit(child)
        for child in block.get("blocks", []):
            visit(child)

    for block in document.get("blocks", []):
        visit(block)
    return payloads
