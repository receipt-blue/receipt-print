import json
import os
import platform
import plistlib
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import quote

from PIL import Image

from .arena import ArenaPrintJob, format_timestamp
from .contact import ContactInfo, parse_vcards, print_contact_card
from .image_utils import print_images_from_pil
from .pdf_utils import pdf_to_images
from .printer import connect_printer, maybe_cut, sanitize_output

PRINTER_EMOJI_RE = re.compile(r"\U0001F5A8(?:\ufe0f|\ufe0e)?")
APPLE_EPOCH_OFFSET = 978307200


@dataclass
class ListenerState:
    last_rowid: int = 0


@dataclass
class IMessageRow:
    rowid: int
    guid: Optional[str]
    text: Optional[str]
    attributed_body: Optional[bytes]
    date: Optional[int]
    handle_id: Optional[str]
    handle_uncanonicalized_id: Optional[str]
    chat_identifier: Optional[str]
    chat_display_name: Optional[str]


@dataclass
class AttachmentRow:
    rowid: int
    filename: Optional[str]
    mime_type: Optional[str]
    transfer_name: Optional[str]


@dataclass
class MessageMedia:
    attachment_images: List[Image.Image]
    attachment_names: List[str]
    pdf_images: List[Image.Image]
    pdf_names: List[str]
    contacts: List[ContactInfo]
    missing: List[str]
    unsupported: List[str]
    invalid: List[str]


@dataclass
class IncomingSummary:
    rowid: int
    text: Optional[str]
    attributed_body: Optional[bytes]
    date: Optional[int]


def default_db_path() -> Path:
    override = os.getenv("IMESSAGE_DB_PATH") or os.getenv("RP_IMESSAGE_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Messages" / "chat.db"


def default_attachments_path() -> Path:
    override = os.getenv("IMESSAGE_ATTACHMENTS_PATH") or os.getenv(
        "RP_IMESSAGE_ATTACHMENTS_PATH"
    )
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Messages" / "Attachments"


def default_state_path() -> Path:
    override = os.getenv("IMESSAGE_STATE_PATH") or os.getenv("RP_IMESSAGE_STATE_PATH")
    if override:
        return Path(override).expanduser()
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home()
    return base / "receipt-print" / "imessage-state.json"


def load_state(path: Path) -> ListenerState:
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return ListenerState()
    except Exception as exc:
        sys.stderr.write(f"Warning: Could not read state file {path}: {exc}\n")
        return ListenerState()
    try:
        data = json.loads(raw)
        return ListenerState(last_rowid=int(data.get("last_rowid", 0)))
    except Exception as exc:
        sys.stderr.write(f"Warning: Could not parse state file {path}: {exc}\n")
        return ListenerState()


def save_state(path: Path, state: ListenerState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"last_rowid": state.last_rowid})
        path.write_text(payload)
    except Exception as exc:
        sys.stderr.write(f"Warning: Could not write state file {path}: {exc}\n")


def open_chat_db(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_latest_incoming_rowid(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT m.ROWID AS rowid FROM message m "
        "ORDER BY m.ROWID DESC LIMIT 1"
    ).fetchone()
    if not row:
        return 0
    return int(row["rowid"])


def fetch_latest_incoming_message(
    conn: sqlite3.Connection,
) -> Optional[IncomingSummary]:
    row = conn.execute(
        "SELECT m.ROWID AS rowid, m.text AS text, "
        "m.attributedBody AS attributed_body, m.date AS date "
        "FROM message m "
        "ORDER BY m.ROWID DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return IncomingSummary(
        rowid=int(row["rowid"]),
        text=row["text"],
        attributed_body=row["attributed_body"],
        date=row["date"],
    )


def fetch_recent_incoming_messages(
    conn: sqlite3.Connection, limit: int
) -> List[IncomingSummary]:
    rows = conn.execute(
        "SELECT m.ROWID AS rowid, m.text AS text, "
        "m.attributedBody AS attributed_body, m.date AS date "
        "FROM message m "
        "ORDER BY m.ROWID DESC LIMIT ?",
        (limit,),
    ).fetchall()
    summaries: List[IncomingSummary] = []
    for row in rows:
        summaries.append(
            IncomingSummary(
                rowid=int(row["rowid"]),
                text=row["text"],
                attributed_body=row["attributed_body"],
                date=row["date"],
            )
        )
    return summaries


def bootstrap_last_rowid(conn: sqlite3.Connection, backfill: int) -> int:
    if backfill <= 0:
        return fetch_latest_incoming_rowid(conn)
    row = conn.execute(
        "SELECT m.ROWID AS rowid FROM message m "
        "ORDER BY m.ROWID DESC LIMIT 1 OFFSET ?",
        (backfill - 1,),
    ).fetchone()
    if not row:
        return 0
    return max(0, int(row["rowid"]) - 1)


def fetch_messages(
    conn: sqlite3.Connection, since_rowid: int, limit: int
) -> List[IMessageRow]:
    rows = conn.execute(
        "SELECT m.ROWID AS rowid, m.guid AS guid, m.text AS text, "
        "m.attributedBody AS attributed_body, m.date AS date, "
        "h.id AS handle_id, h.uncanonicalized_id AS handle_uncanonicalized_id, "
        "c.chat_identifier AS chat_identifier, c.display_name AS chat_display_name "
        "FROM message m "
        "JOIN chat_message_join cmj ON m.ROWID = cmj.message_id "
        "JOIN chat c ON cmj.chat_id = c.ROWID "
        "LEFT JOIN handle h ON m.handle_id = h.ROWID "
        "WHERE m.ROWID > ? "
        "ORDER BY m.ROWID ASC LIMIT ?",
        (since_rowid, limit),
    ).fetchall()
    messages: List[IMessageRow] = []
    for row in rows:
        messages.append(
            IMessageRow(
                rowid=int(row["rowid"]),
                guid=row["guid"],
                text=row["text"],
                attributed_body=row["attributed_body"],
                date=row["date"],
                handle_id=row["handle_id"],
                handle_uncanonicalized_id=row["handle_uncanonicalized_id"],
                chat_identifier=row["chat_identifier"],
                chat_display_name=row["chat_display_name"],
            )
        )
    return messages


def fetch_attachments(
    conn: sqlite3.Connection, message_id: int
) -> List[AttachmentRow]:
    rows = conn.execute(
        "SELECT a.ROWID AS rowid, a.filename AS filename, "
        "a.mime_type AS mime_type, a.transfer_name AS transfer_name "
        "FROM attachment a "
        "JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id "
        "WHERE maj.message_id = ? ORDER BY a.ROWID",
        (message_id,),
    ).fetchall()
    attachments: List[AttachmentRow] = []
    for row in rows:
        attachments.append(
            AttachmentRow(
                rowid=int(row["rowid"]),
                filename=row["filename"],
                mime_type=row["mime_type"],
                transfer_name=row["transfer_name"],
            )
        )
    return attachments


def extract_text_from_attributed_body(blob: Optional[bytes]) -> Optional[str]:
    if not blob:
        return None
    data = bytes(blob)
    payload = None
    if data.startswith(b"bplist"):
        payload = data
    else:
        idx = data.find(b"bplist")
        if idx != -1:
            payload = data[idx:]
    if not payload:
        return extract_text_fallback(data)
    try:
        plist = plistlib.loads(payload)
    except Exception:
        return extract_text_fallback(data)
    objects = plist.get("$objects")
    top = plist.get("$top")
    if isinstance(objects, list) and isinstance(top, dict):
        root = top.get("root")
        root_obj = _resolve_uid(root, objects)
        if isinstance(root_obj, dict):
            text_ref = root_obj.get("NS.string")
            text_obj = _resolve_uid(text_ref, objects)
            if isinstance(text_obj, str):
                return text_obj
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, str) and obj and not obj.startswith("NS"):
                return obj
    return extract_text_fallback(data)


def extract_text_fallback(data: bytes) -> Optional[str]:
    candidates: List[str] = []
    try:
        candidates.append(data.decode("utf-8", errors="ignore"))
    except Exception:
        pass

    null_ratio = data.count(b"\x00") / max(len(data), 1)
    if null_ratio > 0.2:
        for encoding in ("utf-16le", "utf-16be"):
            try:
                candidates.append(data.decode(encoding, errors="ignore"))
            except Exception:
                continue

    key_prefixes = (
        "__kIM",
        "NS",
        "NSMutable",
        "NSConcrete",
        "NSObject",
        "NSString",
        "NSAttributedString",
        "NSDictionary",
        "NSNumber",
    )

    best = None
    best_score = 0
    for candidate in candidates:
        if not candidate:
            continue
        candidate = candidate.replace("\x00", "")
        chunks: List[str] = []
        buf: List[str] = []
        for ch in candidate:
            if ch.isprintable():
                buf.append(ch)
            else:
                if buf:
                    chunks.append("".join(buf))
                    buf = []
        if buf:
            chunks.append("".join(buf))

        for chunk in chunks:
            cleaned = re.sub(r"\s{2,}", " ", chunk).strip()
            if len(cleaned) < 2:
                continue
            letters = sum(c.isalnum() for c in cleaned)
            score = len(cleaned) + letters
            if PRINTER_EMOJI_RE.search(cleaned):
                score += 1000
            if any(ord(c) >= 0x1F300 for c in cleaned):
                score += 100
            if cleaned.startswith(key_prefixes):
                score -= 200
            if cleaned.startswith("__") or cleaned.count("_") >= max(2, len(cleaned) // 4):
                score -= 100
            if score > best_score:
                best_score = score
                best = cleaned

    return best if best_score > 0 else None


def _resolve_uid(obj, objects):
    if isinstance(obj, plistlib.UID):
        idx = obj.data
        if 0 <= idx < len(objects):
            return objects[idx]
    return obj


def resolve_message_text(text: Optional[str], attributed_body: Optional[bytes]) -> str:
    text_val = text or ""
    if contains_printer_emoji(text_val):
        return text_val
    decoded = extract_text_from_attributed_body(attributed_body) or ""
    if decoded and contains_printer_emoji(decoded):
        return decoded
    return text_val or decoded


def contains_printer_emoji(text: str) -> bool:
    if not text:
        return False
    return bool(PRINTER_EMOJI_RE.search(text))


def strip_printer_emoji(text: str) -> str:
    if not text:
        return ""
    cleaned = PRINTER_EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def sanitize_caption(text: str) -> str:
    cleaned = sanitize_output(text)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def apple_time_to_datetime(value: Optional[int]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        raw = int(value)
    except Exception:
        return None
    if raw <= 0:
        return None
    if raw > 1e16:
        seconds = raw / 1_000_000_000
    elif raw > 1e13:
        seconds = raw / 1_000_000
    elif raw > 1e10:
        seconds = raw / 1_000
    else:
        seconds = raw
    return (
        datetime.fromtimestamp(seconds + APPLE_EPOCH_OFFSET, tz=timezone.utc)
        .astimezone()
    )


def resolve_sender(message: IMessageRow) -> str:
    for candidate in (
        message.handle_uncanonicalized_id,
        message.handle_id,
        message.chat_display_name,
        message.chat_identifier,
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    return "Unknown Sender"


def is_image_attachment(path: Path, mime_type: Optional[str]) -> bool:
    mime = (mime_type or "").lower()
    if mime.startswith("image/"):
        return True
    return path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".heic",
    }


def is_pdf_attachment(path: Path, mime_type: Optional[str]) -> bool:
    mime = (mime_type or "").lower()
    if mime == "application/pdf" or mime == "application/x-pdf":
        return True
    return path.suffix.lower() == ".pdf"


def is_vcard_attachment(path: Path, mime_type: Optional[str]) -> bool:
    mime = (mime_type or "").lower().split(";", 1)[0].strip()
    if mime in {
        "text/vcard",
        "text/x-vcard",
        "text/directory",
        "application/vcard",
        "application/x-vcard",
    }:
        return True
    return path.suffix.lower() in {".vcf", ".vcard"}


def resolve_attachment_path(
    filename: Optional[str], attachments_root: Path
) -> Optional[Path]:
    if not filename:
        return None
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = attachments_root / path
    return path


def load_message_media(
    attachments: Sequence[AttachmentRow], attachments_root: Path
) -> MessageMedia:
    attachment_images: List[Image.Image] = []
    attachment_names: List[str] = []
    pdf_images: List[Image.Image] = []
    pdf_names: List[str] = []
    contacts: List[ContactInfo] = []
    missing: List[str] = []
    unsupported: List[str] = []
    invalid: List[str] = []

    for att in attachments:
        path = resolve_attachment_path(att.filename, attachments_root)
        if not path:
            missing.append(att.transfer_name or f"attachment:{att.rowid}")
            continue
        if not path.exists():
            missing.append(str(path))
            continue
        if is_pdf_attachment(path, att.mime_type):
            pdf_pages, page_names = pdf_to_images([str(path)])
            if not pdf_pages:
                unsupported.append(str(path))
                continue
            pdf_images.extend(pdf_pages)
            pdf_names.extend(page_names)
            continue
        if is_image_attachment(path, att.mime_type):
            try:
                img = Image.open(path)
                img.load()
                attachment_images.append(img)
                attachment_names.append(str(path))
            except Exception:
                unsupported.append(str(path))
            continue
        if is_vcard_attachment(path, att.mime_type):
            try:
                raw = path.read_text(errors="replace")
            except Exception as exc:
                invalid.append(f"{path} ({exc})")
                continue
            parsed = parse_vcards(raw)
            if not parsed:
                invalid.append(f"{path} (no contacts)")
                continue
            contacts.extend(parsed)
            continue
        unsupported.append(str(path))

    return MessageMedia(
        attachment_images=attachment_images,
        attachment_names=attachment_names,
        pdf_images=pdf_images,
        pdf_names=pdf_names,
        contacts=contacts,
        missing=missing,
        unsupported=unsupported,
        invalid=invalid,
    )


def print_message(
    sender: str,
    body_text: str,
    contacts: Sequence[ContactInfo],
    contact_photos: Sequence[Optional[Image.Image]],
    images: Sequence[Image.Image],
    image_names: Sequence[str],
    timestamp: Optional[str],
    image_config,
    wrap_mode: str,
    no_cut: bool,
) -> None:
    printer = connect_printer()
    try:
        job = ArenaPrintJob(
            printer,
            image_config.scales,
            image_config.aligns,
            image_config.methods,
            image_config.ts_fmt,
            image_config.dithers,
            image_config.thresholds,
            image_config.diffusions,
            image_config.brightness,
            image_config.contrast,
            image_config.gamma,
            image_config.autocontrast,
            captions_str=None,
            spacing=image_config.spacing,
            footer_text=None,
            debug=image_config.debug,
            wrap_mode=wrap_mode,
            no_cut=no_cut,
        )

        printed_media = False
        caption_text = sanitize_caption(body_text) if body_text else ""

        if contacts:
            contact_align = (
                image_config.aligns[0]
                if image_config.aligns
                else "center"
            )
            for idx, contact in enumerate(contacts):
                photo = (
                    contact_photos[idx]
                    if idx < len(contact_photos)
                    else None
                )
                print_contact_card(
                    printer,
                    contact,
                    photo=photo,
                    wrap_mode=wrap_mode,
                    align=contact_align,
                    qr_size=3,
                    caption=caption_text or None,
                    spacing=image_config.spacing,
                )
                printed_media = True
                if idx < len(contacts) - 1:
                    job.line_break(1)

        if images:
            captions: List[str] = []
            if caption_text and not contacts:
                captions = [caption_text]
            print_images_from_pil(
                printer,
                images,
                image_config.scales,
                image_config.aligns,
                image_config.methods,
                image_config.ts_fmt,
                image_config.dithers,
                image_config.thresholds,
                image_config.diffusions,
                captions_list=captions,
                caption_start=0,
                footer_text=None,
                debug=image_config.debug,
                spacing=image_config.spacing,
                names=image_names,
                brightness_list=image_config.brightness,
                contrast_list=image_config.contrast,
                gamma_list=image_config.gamma,
                autocontrast=image_config.autocontrast,
                wrap_mode=wrap_mode,
                no_cut=no_cut,
            )
            printed_media = True

        if not printed_media and body_text:
            job.print_text(body_text, align="left", font="a")

        if not contacts and (sender or timestamp):
            job.line_break(1)
            if sender:
                job.print_text(
                    f"- {sender}",
                    align="right",
                    font="a",
                    bold=True,
                )
            if timestamp:
                job.print_text(timestamp, align="right", font="a")

        maybe_cut(printer, no_cut=no_cut)
    finally:
        printer.close()


def listen(
    *,
    db_path: Path,
    attachments_path: Path,
    state_path: Path,
    poll_interval: float,
    backfill: int,
    batch_size: int,
    once: bool,
    reset_state: bool,
    verbose: bool,
    image_config,
    wrap_mode: str,
    no_cut: bool,
) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("iMessage listener only runs on macOS.")

    db_path = db_path.expanduser()
    attachments_path = attachments_path.expanduser()
    state_path = state_path.expanduser()

    if reset_state:
        state = ListenerState()
    else:
        state = load_state(state_path)

    def preview(text: str, max_len: int = 80) -> str:
        if not text:
            return "<empty>"
        cleaned = text.replace("\n", "\\n")
        if len(cleaned) > max_len:
            return f"{cleaned[: max_len - 3]}..."
        return cleaned

    def log_line(message: str) -> None:
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    while True:
        try:
            conn = open_chat_db(db_path)
        except sqlite3.OperationalError as exc:
            sys.stderr.write(
                f"Error opening Messages database at {db_path}: {exc}\n"
                "Make sure your terminal has Full Disk Access in System Settings.\n"
            )
            return

        try:
            if state.last_rowid == 0:
                state.last_rowid = bootstrap_last_rowid(conn, backfill)

            while True:
                messages = fetch_messages(conn, state.last_rowid, batch_size)
                if not messages:
                    break
                for message in messages:
                    state.last_rowid = max(state.last_rowid, message.rowid)
                    raw_text = resolve_message_text(
                        message.text, message.attributed_body
                    )
                    if verbose:
                        sender = resolve_sender(message)
                        msg_dt = apple_time_to_datetime(message.date)
                        timestamp = format_timestamp(msg_dt) or "unknown"
                        log_line(
                            " ".join(
                                [
                                    f"rowid={message.rowid}",
                                    f"time={timestamp}",
                                    f"from={sender}",
                                    f"emoji={'yes' if contains_printer_emoji(raw_text) else 'no'}",
                                    f"text={preview(raw_text)}",
                                ]
                            )
                        )
                    if not contains_printer_emoji(raw_text):
                        continue
                    body_text = strip_printer_emoji(raw_text)
                    attachments = fetch_attachments(conn, message.rowid)
                    media = load_message_media(attachments, attachments_path)

                    if media.missing:
                        for path in media.missing:
                            sys.stderr.write(
                                f"Warning: Missing attachment for message {message.rowid}: {path}\n"
                            )
                    if media.unsupported:
                        for path in media.unsupported:
                            sys.stderr.write(
                                f"Warning: Unsupported attachment for message {message.rowid}: {path}\n"
                            )
                    if media.invalid:
                        for info in media.invalid:
                            sys.stderr.write(
                                "Warning: Invalid vCard attachment for message "
                                f"{message.rowid}: {info}\n"
                            )

                    contacts = media.contacts
                    contact_photos: List[Optional[Image.Image]] = []
                    if contacts:
                        image_idx = 0
                        for contact in contacts:
                            photo = contact.photo
                            if photo is None and image_idx < len(
                                media.attachment_images
                            ):
                                photo = media.attachment_images[image_idx]
                                image_idx += 1
                            contact_photos.append(photo)
                        images = (
                            media.pdf_images
                            + media.attachment_images[image_idx:]
                        )
                        names = (
                            media.pdf_names
                            + media.attachment_names[image_idx:]
                        )
                    else:
                        images = media.pdf_images + media.attachment_images
                        names = media.pdf_names + media.attachment_names

                    if not images and not contacts and not body_text:
                        sys.stderr.write(
                            f"Warning: No printable content for message {message.rowid}.\n"
                        )
                        continue

                    sender = resolve_sender(message)
                    msg_dt = apple_time_to_datetime(message.date)
                    timestamp = format_timestamp(msg_dt)
                    log_line(
                        " ".join(
                            [
                                "print",
                                f"rowid={message.rowid}",
                                f"time={timestamp or 'unknown'}",
                                f"from={sender}",
                                f"text={preview(body_text)}",
                                f"images={len(images)}",
                            ]
                        )
                    )
                    try:
                        print_message(
                            sender,
                            body_text,
                            contacts,
                            contact_photos,
                            images,
                            names,
                            timestamp,
                            image_config,
                            wrap_mode,
                            no_cut,
                        )
                    except Exception as exc:
                        sys.stderr.write(
                            f"Error printing message {message.rowid}: {exc}\n"
                        )

                if len(messages) < batch_size:
                    break
        finally:
            conn.close()
            save_state(state_path, state)

        if once:
            return

        time.sleep(poll_interval)
