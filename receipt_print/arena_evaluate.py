from __future__ import annotations

import base64
import io
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

from .arena import (
    ArenaClient,
    ChannelIterator,
    canonical_channel_url,
    parse_channel_identifier,
)
from .arena_document import (
    VARIANTS,
    ArenaChannel,
    compose_channel_document,
    expected_qr_payloads,
    normalize_channel,
)
from .receipt_core import PreviewResult, ReceiptCoreClient


@dataclass(frozen=True)
class EvaluationArtifact:
    channel: str
    variant: str
    directory: Path
    pages: tuple[Path, ...]
    crops: tuple[Path, ...]
    qr_report: Path


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "channel"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _channel_total(meta: dict[str, Any], iterator: ChannelIterator) -> int:
    counts = meta.get("counts")
    if isinstance(counts, dict):
        for key in ("contents", "blocks"):
            if isinstance(counts.get(key), int):
                return counts[key]
    if isinstance(meta.get("length"), int):
        return meta["length"]
    return 0


def _is_pinned(block: dict[str, Any]) -> bool:
    connection = block.get("connection")
    if isinstance(connection, dict) and connection.get("pinned") is not None:
        return bool(connection["pinned"])
    return bool(block.get("pinned"))


def fetch_channel_snapshot(
    client: ArenaClient,
    value: str,
    *,
    selection: str,
    limit: Optional[int],
    seed: Optional[int] = None,
) -> tuple[ArenaChannel, dict[str, Any]]:
    ref = parse_channel_identifier(value)
    if ref.slug:
        meta = client.fetch_channel_meta_by_slug(ref.slug, 1, 1)
    else:
        meta = client.fetch_channel_meta_by_id(ref.channel_id) or {}
    channel_url = canonical_channel_url(ref, meta)
    if not channel_url:
        raise ValueError(f"Could not determine canonical channel URL for {value}")

    if selection not in {"full", "top", "random"}:
        raise ValueError(f"Unknown channel selection: {selection}")
    effective_limit = 5 if selection in {"top", "random"} and limit is None else limit
    iterator = ChannelIterator(client, ref, sort="position_desc")
    raw_blocks: list[dict[str, Any]] = []
    for item in iterator:
        raw_blocks.append(item)
        if (
            selection != "random"
            and effective_limit
            and len(raw_blocks) >= effective_limit
        ):
            break

    total = _channel_total(meta, iterator) or len(raw_blocks)
    population_count = len(raw_blocks)
    if selection == "random" and effective_limit:
        sampled_indices = random.Random(seed).sample(
            range(population_count), min(effective_limit, population_count)
        )
        raw_blocks = [raw_blocks[index] for index in sorted(sampled_indices)]
        raw_blocks.sort(key=lambda block: not _is_pinned(block))
    fetched_at = datetime.now(timezone.utc).isoformat()
    channel = normalize_channel(
        meta,
        raw_blocks,
        channel_url,
        total_count=total,
        fetched_at=fetched_at,
    )
    source = {
        "channel": meta,
        "contents": raw_blocks,
        "selection": selection,
        "limit": effective_limit,
        "sort": "position_desc",
        "fetchedAt": fetched_at,
    }
    if selection == "random":
        source.update(
            randomSeed=seed,
            populationCount=population_count,
            pinOrder="pinned-first",
        )
    return channel, source


def _png_data_uri(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        frame = image.convert("RGB")
        output = io.BytesIO()
        frame.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def collect_media(client: ArenaClient, channel: ArenaChannel) -> dict[str, str]:
    media: dict[str, str] = {}
    if channel.cover_url:
        try:
            media["__channel_cover__"] = _png_data_uri(
                client.download_media(channel.cover_url)
            )
        except Exception:
            pass
    for block in channel.blocks:
        if not block.preview_url:
            continue
        try:
            media[block.id] = _png_data_uri(client.download_media(block.preview_url))
        except Exception:
            continue
    return media


def _page_paths(directory: Path, preview: PreviewResult) -> tuple[Path, ...]:
    pages = []
    for index, encoded in enumerate(preview.pngs, start=1):
        path = directory / f"page-{index:03d}.png"
        path.write_bytes(base64.b64decode(encoded))
        pages.append(path)
    return tuple(pages)


def _region_bounds(region: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
    bounds = region.get("bounds") if isinstance(region.get("bounds"), dict) else region
    try:
        x = int(bounds["x"])
        y = int(bounds["y"])
        width = int(bounds["width"])
        height = int(bounds["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return x, y, x + width, y + height


def _crop_regions(
    directory: Path,
    pages: tuple[Path, ...],
    regions: Iterable[dict[str, Any]],
) -> tuple[tuple[Path, ...], tuple[tuple[Path, dict[str, Any]], ...]]:
    crops_dir = directory / "crops"
    crops_dir.mkdir(exist_ok=True)
    crops: list[Path] = []
    mapped: list[tuple[Path, dict[str, Any]]] = []
    opened: dict[int, Image.Image] = {}
    try:
        for index, region in enumerate(regions, start=1):
            bounds = _region_bounds(region)
            page = int(region.get("page", 0))
            if not bounds or page < 0 or page >= len(pages):
                continue
            if page not in opened:
                opened[page] = Image.open(pages[page])
            name = _slug(str(region.get("path") or f"region-{index}"))
            path = crops_dir / f"{index:03d}-{name}.png"
            opened[page].crop(bounds).save(path)
            crops.append(path)
            mapped.append((path, region))
    finally:
        for image in opened.values():
            image.close()
    return tuple(crops), tuple(mapped)


def _qr_report(
    path: Path,
    document: dict[str, Any],
    preview: PreviewResult,
) -> None:
    results_by_payload: dict[str, list[dict[str, Any]]] = {}
    for result in preview.qr_results:
        payload = result.get("payload") or result.get("expectedPayload")
        results_by_payload.setdefault(str(payload), []).append(result)
    rows = []
    for payload in expected_qr_payloads(document):
        matches = results_by_payload.get(payload, [])
        decoded = (
            any(result.get("decoded") is True for result in matches)
            if matches
            else None
        )
        rows.append(
            {
                "payload": payload,
                "decoded": decoded,
                "reported": bool(matches),
                "status": (
                    "decoded"
                    if decoded is True
                    else "decode-failed"
                    if decoded is False
                    else "not-reported"
                ),
                "results": matches,
                "validation": "software-only",
            }
        )
    _write_json(path, {"codes": rows})


def _contact_sheet(
    output: Path,
    entries: list[tuple[str, Path]],
    *,
    filename: str = "contact-sheet.png",
) -> Optional[Path]:
    if not entries:
        return None
    columns = 2
    cell_width = 520
    label_height = 34
    images: list[tuple[str, Image.Image]] = []
    for label, path in entries:
        image = Image.open(path).convert("RGB")
        ratio = min(1.0, 480 / image.width, 850 / image.height)
        if ratio < 1:
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
            )
        images.append((label, image))
    cell_height = max(image.height for _, image in images) + label_height + 24
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(images):
        column = index % columns
        row = index // columns
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * cell_height + label_height
        draw.text(
            (column * cell_width + 16, row * cell_height + 10),
            label,
            fill="black",
            font=font,
        )
        sheet.paste(image, (x, y))
    path = output / filename
    sheet.save(path)
    for _, image in images:
        image.close()
    sheet.close()
    return path


def evaluate_channel(
    channel: ArenaChannel,
    source: dict[str, Any],
    core: ReceiptCoreClient,
    output: Path,
    *,
    variants: Iterable[str] = VARIANTS,
    media: Optional[dict[str, str]] = None,
    channel_qr: Optional[bool] = None,
) -> list[EvaluationArtifact]:
    channel_dir = output / _slug(channel.slug or channel.title)
    channel_dir.mkdir(parents=True, exist_ok=True)
    _write_json(channel_dir / "snapshot.json", channel.to_dict())
    _write_json(channel_dir / "source-raw.json", source)
    artifacts: list[EvaluationArtifact] = []
    contact_entries: list[tuple[str, Path]] = []
    pin_entries: list[tuple[str, Path]] = []
    pinned_ids = {block.id for block in channel.blocks if block.placement.pinned}
    manifest_variants = []

    for variant in variants:
        variant_dir = channel_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        document = compose_channel_document(
            channel,
            variant,
            media=media,
            channel_qr=channel_qr,
        )
        _write_json(variant_dir / "document.json", document)
        study = None
        preview = core.preview(document, study=study)
        pages = _page_paths(variant_dir, preview)
        crops, mapped_crops = _crop_regions(variant_dir, pages, preview.regions)
        pin_candidates: list[tuple[Path, dict[str, Any]]] = []
        for crop, region in mapped_crops:
            ref = region.get("ref")
            if (
                isinstance(ref, dict)
                and str(ref.get("id")) in pinned_ids
                and ref.get("note") == "pinned connection"
            ):
                pin_candidates.append((crop, region))
        if pin_candidates:
            crop, _ = min(
                pin_candidates,
                key=lambda item: (
                    int(item[1].get("height", 0)),
                    int(item[1].get("width", 0)),
                ),
            )
            pin_entries.append((variant, crop))
        qr_report = variant_dir / "qr-report.json"
        _qr_report(qr_report, document, preview)
        _write_json(
            variant_dir / "preview.json",
            {
                "economy": preview.economy,
                "regions": list(preview.regions),
                "qrResults": list(preview.qr_results),
                "text": preview.text,
                "study": study,
            },
        )
        artifacts.append(
            EvaluationArtifact(
                channel=channel.title,
                variant=variant,
                directory=variant_dir,
                pages=pages,
                crops=crops,
                qr_report=qr_report,
            )
        )
        if pages:
            contact_entries.append((variant, pages[0]))
        manifest_variants.append(
            {
                "name": variant,
                "pages": [str(path.relative_to(channel_dir)) for path in pages],
                "crops": [str(path.relative_to(channel_dir)) for path in crops],
                "economy": preview.economy,
            }
        )

    contact = _contact_sheet(channel_dir, contact_entries)
    pin_contact = _contact_sheet(
        channel_dir,
        pin_entries,
        filename="pin-treatment-sheet.png",
    )
    _write_json(
        channel_dir / "manifest.json",
        {
            "channel": channel.canonical_url,
            "title": channel.title,
            "order": "position_desc",
            "fetchedAt": channel.fetched_at,
            "variants": manifest_variants,
            "contactSheet": contact.name if contact else None,
            "pinTreatmentSheet": pin_contact.name if pin_contact else None,
        },
    )
    return artifacts


def load_snapshot(path: Path) -> ArenaChannel:
    return ArenaChannel.from_dict(json.loads(path.read_text(encoding="utf-8")))
