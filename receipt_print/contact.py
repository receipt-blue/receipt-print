import base64
import binascii
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw, ImageFont

from .image_utils import desired_orientation
from .printer import (
    CHAR_WIDTH,
    count_lines,
    enforce_line_limit,
    sanitize_output,
    wrap_text,
)

WS_CLEANER = re.compile(r"\s+")
BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


@dataclass
class ContactInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    photo: Optional[Image.Image] = None

    def has_data(self) -> bool:
        return bool(self.name or self.email or self.phone)


@dataclass
class ContactPanel:
    lines: List[str]
    bold_line_indices: Set[int]
    width: int


def normalize_contact_value(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = sanitize_output(value)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return WS_CLEANER.sub(" ", cleaned).strip()


def _normalize_caption(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = sanitize_output(value)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return WS_CLEANER.sub(" ", cleaned).strip()


def _unfold_vcard_lines(raw: str) -> List[str]:
    lines = raw.splitlines()
    unfolded: List[str] = []
    for line in lines:
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _name_from_n(value: str) -> str:
    parts = value.split(";")
    parts += [""] * (5 - len(parts))
    last, first, additional, prefix, suffix = parts[:5]
    name = " ".join(p for p in [prefix, first, additional, last, suffix] if p)
    return name


def _parse_vcard_lines(lines: Sequence[str]) -> ContactInfo:
    name = ""
    email = ""
    phone = ""
    n_value = ""
    photo: Optional[Image.Image] = None

    for line in lines:
        if ":" not in line:
            continue
        prop, _, value = line.partition(":")
        prop_name = prop.split(";", 1)[0].strip().upper()
        if prop_name == "FN" and not name:
            name = value
        elif prop_name == "N" and not n_value:
            n_value = value
        elif prop_name == "EMAIL" and not email:
            email = value
        elif prop_name == "TEL" and not phone:
            phone = value
        elif prop_name == "PHOTO" and photo is None:
            photo = _parse_vcard_photo(prop, value)

    if not name and n_value:
        name = _name_from_n(n_value)

    contact = ContactInfo(
        name=normalize_contact_value(name),
        email=normalize_contact_value(email),
        phone=normalize_contact_value(phone),
        photo=photo,
    )
    return contact


def parse_vcards(raw: str) -> List[ContactInfo]:
    lines = _unfold_vcard_lines(raw)
    cards: List[ContactInfo] = []
    current: List[str] = []
    in_card = False

    for line in lines:
        upper = line.strip().upper()
        if upper == "BEGIN:VCARD":
            current = []
            in_card = True
            continue
        if upper == "END:VCARD":
            if in_card:
                card = _parse_vcard_lines(current)
                if card.has_data():
                    cards.append(card)
            in_card = False
            current = []
            continue
        if in_card:
            current.append(line)

    return cards


def build_vcard(contact: ContactInfo) -> str:
    lines = ["BEGIN:VCARD", "VERSION:3.0"]
    if contact.name:
        lines.append(f"FN:{contact.name}")
    if contact.email:
        lines.append(f"EMAIL:{contact.email}")
    if contact.phone:
        lines.append(f"TEL:{contact.phone}")
    lines.append("END:VCARD")
    return "\n".join(lines)


def _parse_vcard_photo(prop: str, value: str) -> Optional[Image.Image]:
    if not value:
        return None
    params = [item.strip() for item in prop.split(";")[1:] if item.strip()]
    param_map = {}
    for param in params:
        if "=" in param:
            key, val = param.split("=", 1)
            param_map[key.strip().upper()] = val.strip()
        else:
            param_map[param.upper()] = ""

    raw_value = value.strip()
    if not raw_value:
        return None

    if raw_value.lower().startswith("data:"):
        return _image_from_data_uri(raw_value)

    value_type = param_map.get("VALUE", "").upper()
    if value_type in {"URI", "URL"}:
        return _image_from_uri(raw_value)

    encoding = param_map.get("ENCODING", "").upper()
    if encoding in {"B", "BASE64"} or _looks_like_base64(raw_value):
        return _image_from_base64(raw_value)

    return None


def _looks_like_base64(value: str) -> bool:
    if len(value) < 32:
        return False
    return bool(BASE64_RE.match(value))


def _image_from_base64(value: str) -> Optional[Image.Image]:
    cleaned = re.sub(r"\s+", "", value)
    try:
        data = base64.b64decode(cleaned, validate=False)
    except (binascii.Error, ValueError):
        return None
    return _image_from_bytes(data)


def _image_from_data_uri(value: str) -> Optional[Image.Image]:
    header, _, data = value.partition(",")
    if not data:
        return None
    if ";base64" not in header.lower():
        return None
    return _image_from_base64(data)


def _image_from_uri(value: str) -> Optional[Image.Image]:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return None
    path = parsed.path if parsed.scheme == "file" else value
    try:
        photo_path = Path(unquote(path)).expanduser()
    except Exception:
        return None
    if not photo_path.is_file():
        return None
    try:
        img = Image.open(photo_path)
        img.load()
        return img
    except Exception:
        return None


def _image_from_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def _wrap_lines(value: str, width: int, wrap_mode: str) -> List[str]:
    if not value:
        return []
    wrapped = wrap_text(value, width, wrap_mode)
    return wrapped.splitlines() if wrapped else [""]


def _wrap_with_prefix(
    prefix: str, value: str, width: int, wrap_mode: str
) -> List[str]:
    if not value:
        return []
    if len(prefix) >= width:
        return _wrap_lines(prefix + value, width, wrap_mode)
    avail = width - len(prefix)
    wrapped = wrap_text(value, avail, wrap_mode)
    lines = wrapped.splitlines() if wrapped else [""]
    out = [prefix + lines[0]]
    pad = " " * len(prefix)
    for line in lines[1:]:
        out.append(pad + line)
    return out


def build_contact_panel(
    contact: ContactInfo, *, width: int, wrap_mode: str
) -> ContactPanel:
    width = max(4, width)
    inner_width = max(1, width - 4)
    content: List[str] = []
    bold_indices: Set[int] = set()

    name_lines = _wrap_lines(contact.name, inner_width, wrap_mode)
    if name_lines:
        start_idx = len(content)
        if len(name_lines) == 1:
            name_lines = [name_lines[0].center(inner_width)]
        content.extend(name_lines)
        bold_indices.update(range(start_idx, start_idx + len(name_lines)))

    if name_lines and (contact.email or contact.phone):
        content.append("")

    content.extend(
        _wrap_with_prefix("email: ", contact.email, inner_width, wrap_mode)
    )
    content.extend(
        _wrap_with_prefix("phone: ", contact.phone, inner_width, wrap_mode)
    )

    if not content:
        content.append("")

    border = "+" + "-" * (width - 2) + "+"
    lines = [border]
    for line in content:
        lines.append("| " + line.ljust(inner_width) + " |")
    lines.append(border)

    bold_lines = {1 + idx for idx in bold_indices}
    return ContactPanel(lines=lines, bold_line_indices=bold_lines, width=width)


def _measure_font(font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        bbox = font.getbbox("M")
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return font.getsize("M")


def _render_qr_image(payload: str, max_width: int) -> Image.Image:
    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M
    except ImportError as exc:
        raise RuntimeError(
            "Missing QR dependency. Install qrcode to render landscape contact cards."
        ) from exc

    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, border=1, box_size=1)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    if img.width > max_width:
        new_width = max_width
        new_height = max(1, int(img.height * (new_width / img.width)))
        img = img.resize((new_width, new_height), Image.NEAREST)
    elif img.width < max_width:
        scale = max_width // img.width
        if scale > 1:
            img = img.resize(
                (img.width * scale, img.height * scale), Image.NEAREST
            )
    return img


def render_landscape_card(
    contact: ContactInfo,
    photo: Image.Image,
    qr_payload: str,
    printer_width_px: int,
    *,
    wrap_mode: str,
    align: str,
    qr_size: int,
) -> Image.Image:
    font = ImageFont.load_default()
    font_width, font_height = _measure_font(font)
    line_height = font_height + 2
    gutter_px = font_width * 2

    text_chars = max(20, int(CHAR_WIDTH * 0.6))
    text_chars = min(text_chars, max(10, CHAR_WIDTH - 8))
    text_width_px = text_chars * font_width
    left_width_px = max(1, printer_width_px - text_width_px - gutter_px)
    if left_width_px < font_width * 8:
        text_chars = max(10, (printer_width_px // font_width) - 10)
        text_width_px = text_chars * font_width
        left_width_px = max(1, printer_width_px - text_width_px - gutter_px)

    panel = build_contact_panel(contact, width=text_chars, wrap_mode=wrap_mode)
    text_lines = panel.lines
    text_height = max(1, len(text_lines) * line_height)
    text_img = Image.new("RGB", (text_width_px, text_height), "white")
    draw = ImageDraw.Draw(text_img)
    for idx, line in enumerate(text_lines):
        y = idx * line_height
        draw.text((0, y), line, font=font, fill="black")
        if idx in panel.bold_line_indices:
            draw.text((1, y), line, font=font, fill="black")

    scale_ratio = min(1.0, qr_size / 3) if qr_size > 0 else 1.0
    qr_max_width = max(1, int(text_width_px * scale_ratio))
    qr_img = _render_qr_image(qr_payload, qr_max_width)
    right_height = text_height + line_height + qr_img.height

    work = photo.convert("RGB").copy()
    if desired_orientation(align) == "landscape":
        work = work.rotate(270, expand=True)
    if work.width > left_width_px:
        ratio = left_width_px / work.width
        work = work.resize(
            (left_width_px, max(1, int(work.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    total_height = max(right_height, work.height)
    card = Image.new("RGB", (printer_width_px, total_height), "white")

    img_x = max(0, (left_width_px - work.width) // 2)
    if align == "l-bottom":
        img_y = max(0, total_height - work.height)
    elif align == "l-center":
        img_y = max(0, (total_height - work.height) // 2)
    else:
        img_y = 0
    card.paste(work, (img_x, img_y))

    text_x = left_width_px + gutter_px
    card.paste(text_img, (text_x, 0))
    qr_x = text_x + max(0, text_width_px - qr_img.width)
    qr_y = max(0, total_height - qr_img.height)
    card.paste(qr_img, (qr_x, qr_y))
    return card


def print_contact_image(
    printer,
    image: Image.Image,
    *,
    wrap_mode: str,
    align: str,
    spacing: int = 1,
    caption: Optional[str] = None,
) -> None:
    from .image_utils import print_images_from_pil

    captions_list = [caption] if caption else None
    print_images_from_pil(
        printer,
        [image],
        [1.0],
        [align],
        ["raster"],
        None,
        [None],
        [0.5],
        [1.0],
        captions_list=captions_list,
        caption_start=0,
        footer_text=None,
        debug=False,
        spacing=spacing,
        names=["contact"],
        brightness_list=[1.0],
        contrast_list=[1.0],
        gamma_list=[1.0],
        autocontrast=False,
        wrap_mode=wrap_mode,
        no_cut=True,
    )
    printer.set(align="left")


def _print_caption_text(printer, caption: str, wrap_mode: str) -> None:
    cleaned = _normalize_caption(caption)
    if not cleaned:
        return
    printer.text("\n")
    printer.set(align="center", font="b", bold=True)
    wrapped = wrap_text(cleaned, CHAR_WIDTH, wrap_mode)
    if not wrapped.endswith("\n"):
        wrapped += "\n"
    printer.text(wrapped)
    printer.set(align="left")


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def print_contact_card(
    printer,
    contact: ContactInfo,
    *,
    photo: Optional[Image.Image],
    wrap_mode: str,
    align: str,
    qr_size: int,
    caption: Optional[str] = None,
    spacing: int = 1,
) -> None:
    from escpos.escpos import QR_ECLEVEL_M

    align = (align or "center").lower()
    photo_to_use = contact.photo or photo
    qr_payload = build_vcard(contact)
    landscape = desired_orientation(align) == "landscape"

    if landscape and photo_to_use is not None:
        card_img = render_landscape_card(
            contact,
            photo_to_use,
            qr_payload,
            printer_width_px=_printer_width_px(printer),
            wrap_mode=wrap_mode,
            align=align,
            qr_size=qr_size,
        )
        print_contact_image(
            printer,
            card_img,
            wrap_mode=wrap_mode,
            align="left",
            spacing=spacing,
            caption=_normalize_caption(caption),
        )
        return

    panel = build_contact_panel(contact, width=CHAR_WIDTH, wrap_mode=wrap_mode)
    enforce_line_limit(count_lines("\n".join(panel.lines), panel.width))
    printer.set(align="left", font="a", bold=False, normal_textsize=True)
    for line_idx, line in enumerate(panel.lines):
        printer.set(bold=line_idx in panel.bold_line_indices)
        printer.text(_ensure_trailing_newline(line))
    printer.set(align="left", font="a", bold=False, normal_textsize=True)

    if photo_to_use is not None:
        printer.text("\n")
        print_contact_image(
            printer,
            photo_to_use,
            wrap_mode=wrap_mode,
            align="center",
            spacing=spacing,
        )

    printer.set(align="right", font="a", bold=False)
    try:
        printer.qr(qr_payload, size=qr_size, ec=QR_ECLEVEL_M)
    except Exception as exc:
        sys.stderr.write(f"Warning: Failed to print QR for contact: {exc}\n")
    printer.set(align="left")

    if caption:
        _print_caption_text(printer, caption, wrap_mode)


def _printer_width_px(printer) -> int:
    max_width = 576
    try:
        width = printer.profile.profile_data["media"]["width"]["pixels"]
        if width != "Unknown":
            max_width = int(width)
    except Exception:
        pass
    return max_width
