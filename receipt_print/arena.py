import hashlib
import json
import mimetypes
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import quote, unquote, urlparse

import requests
from PIL import Image

from .image_loader import load_image_from_bytes
from .image_utils import parse_caption_csv, print_images_from_pil
from .pdf_utils import pdf_to_images
from .printer import (
    CHAR_WIDTH,
    count_lines,
    enforce_line_limit,
    sanitize_output,
    scaled_char_width,
    wrap_text,
)

USER_AGENT = "receipt-print/0.0.1 (+https://github.com/jmpaz/receipt-print)"
DEFAULT_API_BASE = os.getenv("ARENA_API_BASE", "https://api.are.na/v2")
JSON_TIMEOUT = 20
MEDIA_TIMEOUT = 60


class ArenaError(Exception):
    """Base exception for Are.na handling."""


class ArenaNotFound(ArenaError):
    """Raised when resource is not found."""


class ArenaUnauthorized(ArenaError):
    """Raised when resource requires auth."""


class ArenaDownloadError(ArenaError):
    """Raised when media download fails."""


@dataclass
class ChannelRef:
    slug: Optional[str] = None
    user: Optional[str] = None
    channel_id: Optional[str] = None
    original: str = ""


def default_cache_dir() -> Path:
    """Determine the default cache directory."""
    override = os.getenv("ARENA_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Windows":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "receipt-print" / "arena").expanduser()


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def parse_block_identifier(value: str) -> str:
    """Resolve block identifier to numeric ID string."""
    val = value.strip()
    if not val:
        raise ArenaError("Empty block identifier.")
    if val.isdigit():
        return val
    if val.startswith("http://") or val.startswith("https://"):
        parsed = urlparse(val)
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0] in {"block", "blocks"} and len(parts) >= 2:
            candidate = parts[1]
            if candidate.isdigit():
                return candidate
    raise ArenaError(f"Unable to parse block identifier: {value}")


def parse_channel_identifier(value: str) -> ChannelRef:
    """Parse channel input into ChannelRef."""
    raw = value.strip()
    if not raw:
        raise ArenaError("Empty channel identifier.")
    if raw.isdigit():
        return ChannelRef(channel_id=raw, original=value)
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            user = parts[0]
            channel_slug = parts[1]
            return ChannelRef(slug=channel_slug, user=user, original=value)
        raise ArenaError(f"Unrecognized channel URL: {value}")
    slug = raw.strip("/")
    if "/" in slug:
        user, channel_slug = slug.split("/", 1)
        if not user or not channel_slug:
            raise ArenaError("Channel slug must include username/slug.")
        return ChannelRef(slug=channel_slug, user=user, original=value)
    if not slug:
        raise ArenaError("Empty channel slug.")
    return ChannelRef(slug=slug, original=value)


def block_class(block: Dict[str, Any]) -> str:
    return (block.get("class") or block.get("base_class") or "").lower()


def block_title(block: Dict[str, Any]) -> str:
    return block.get("title") or block.get("generated_title") or ""


def block_description(block: Dict[str, Any]) -> str:
    return block.get("description") or ""


def block_text_content(block: Dict[str, Any]) -> Optional[str]:
    content = block.get("content")
    if content:
        return content
    # Some blocks provide content in content_html; fall back if it's plain text.
    html = block.get("content_html")
    if html and not any(tag in html.lower() for tag in ("<p", "<div", "<br")):
        return html
    return None


def block_preview_urls(block: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    image = block.get("image") or {}
    if isinstance(image, dict):
        for key in ("display", "original", "thumb"):
            maybe = image.get(key)
            if isinstance(maybe, dict):
                url = maybe.get("url")
                if url:
                    urls.append(url)
            elif isinstance(maybe, str):
                urls.append(maybe)
    thumb = block.get("thumb_url")
    if thumb:
        urls.append(thumb)
    return urls


def block_attachment(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attachment = block.get("attachment")
    if isinstance(attachment, dict) and attachment.get("url"):
        return attachment
    return None


def block_user_name(block: Dict[str, Any]) -> Optional[str]:
    user = block.get("user") or {}
    if not isinstance(user, dict):
        return None
    full = user.get("full_name")
    if full:
        return full
    return user.get("username")


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        txt = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            return dt
        return dt.astimezone()
    except Exception:
        return None


def format_timestamp(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.strftime("%Y-%m-%d %H:%M")


def canonical_block_url(block_id: str) -> str:
    return f"https://www.are.na/block/{block_id}"


def canonical_channel_url(
    ref: ChannelRef, meta: Optional[Dict[str, Any]]
) -> Optional[str]:
    if ref.slug and ref.user:
        return f"https://www.are.na/{ref.user}/{ref.slug}"
    if ref.original and ref.original.startswith("http"):
        return ref.original
    if meta:
        slug = meta.get("slug")
        user = meta.get("user", {})
        username = user.get("slug") or user.get("username")
        if slug and username:
            return f"https://www.are.na/{username}/{slug}"
    if ref.channel_id:
        return f"https://www.are.na/channel/{ref.channel_id}"
    return None


def ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


class ArenaClient:
    """HTTP client for Are.na API with caching support."""

    def __init__(self, cache_enabled: bool = True):
        token = os.getenv("ARENA_TOKEN")
        self.base_url = DEFAULT_API_BASE.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

        self.cache_enabled = cache_enabled
        self.cache_dir: Optional[Path] = None
        if cache_enabled:
            self.cache_dir = default_cache_dir()
            self.api_cache = self.cache_dir / "api"
            self.media_cache = self.cache_dir / "media"
            self.api_cache.mkdir(parents=True, exist_ok=True)
            self.media_cache.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.session.close()

    # -- caching helpers -------------------------------------------------
    def _cache_paths(self, kind: str, key: str) -> Tuple[Path, Path]:
        assert self.cache_dir is not None
        digest = _hash_key(key)
        base = self.api_cache if kind == "api" else self.media_cache
        return base / f"{digest}.data", base / f"{digest}.meta"

    @staticmethod
    def _read_meta(path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    @staticmethod
    def _store_meta(path: Path, response: requests.Response) -> None:
        meta = {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_type": response.headers.get("Content-Type"),
        }
        path.write_text(json.dumps(meta))

    # -- HTTP operations -------------------------------------------------
    def _request_with_retries(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: int,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> requests.Response:
        attempt = 0
        while True:
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    stream=stream,
                )
            except requests.RequestException as exc:
                attempt += 1
                if attempt >= 3:
                    raise ArenaError(f"Network error contacting Are.na: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if response.status_code in (429, 503):
                attempt += 1
                if attempt >= 3:
                    raise ArenaError("Are.na API rate limit exceeded.")
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else float(2**attempt)
                time.sleep(delay)
                continue

            return response

    def _request_json(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        cache_key = url
        if params:
            items = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            cache_key = f"{url}?{items}"

        headers: Dict[str, str] = {}
        data_path = meta_path = None
        if self.cache_enabled and self.cache_dir:
            data_path, meta_path = self._cache_paths("api", cache_key)
            meta = self._read_meta(meta_path)
            etag = meta.get("etag")
            last_modified = meta.get("last_modified")
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

        response = self._request_with_retries(
            url, params=params, timeout=JSON_TIMEOUT, headers=headers
        )

        if response.status_code == 304:
            if data_path and data_path.exists():
                return json.loads(data_path.read_text())
            raise ArenaError("Received 304 but no cached data available.")

        if response.status_code == 404:
            raise ArenaNotFound(url)
        if response.status_code in (401, 403):
            raise ArenaUnauthorized(url)
        if not response.ok:
            raise ArenaError(f"Are.na API error {response.status_code} for {url}")

        payload = response.json()
        if self.cache_enabled and data_path and meta_path:
            data_path.write_text(response.text)
            self._store_meta(meta_path, response)
        return payload

    def _download_binary(self, url: str) -> bytes:
        cache_key = url
        headers: Dict[str, str] = {}
        data_path = meta_path = None
        if self.cache_enabled and self.cache_dir:
            data_path, meta_path = self._cache_paths("media", cache_key)
            meta = self._read_meta(meta_path)
            etag = meta.get("etag")
            last_modified = meta.get("last_modified")
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

        response = self._request_with_retries(
            url, timeout=MEDIA_TIMEOUT, headers=headers
        )

        if response.status_code == 304:
            if data_path and data_path.exists():
                return data_path.read_bytes()
            raise ArenaDownloadError("Received 304 without cached media.")

        if not response.ok:
            raise ArenaDownloadError(f"Download failed with {response.status_code}")

        content = response.content
        if self.cache_enabled and data_path and meta_path:
            data_path.write_bytes(content)
            self._store_meta(meta_path, response)
        return content

    # -- public helpers --------------------------------------------------
    def fetch_block(self, block_id: str) -> Dict[str, Any]:
        return self._request_json(f"/blocks/{quote(block_id)}")

    def fetch_channel_meta_by_slug(
        self, slug: str, page: int, per: int
    ) -> Dict[str, Any]:
        params = {"page": page, "per": per}
        return self._request_json(f"/channels/{quote(slug)}", params=params)

    def fetch_channel_contents_by_id(
        self, channel_id: str, page: int, per: int
    ) -> Dict[str, Any]:
        params = {"page": page, "per": per}
        return self._request_json(
            f"/channels/{quote(channel_id)}/contents", params=params
        )

    def fetch_channel_meta_by_id(self, channel_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._request_json(f"/channels/{quote(channel_id)}")
        except ArenaNotFound:
            return None

    def download_media(self, url: str) -> bytes:
        return self._download_binary(url)


class ChannelIterator:
    """Iterator over channel contents, retaining metadata."""

    def __init__(self, client: ArenaClient, ref: ChannelRef, per: int = 100):
        self.client = client
        self.ref = ref
        self.per = per
        self.meta: Optional[Dict[str, Any]] = None
        self._page = 1

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while True:
            if self.ref.slug:
                data = self.client.fetch_channel_meta_by_slug(
                    self.ref.slug, self._page, self.per
                )
                if self.meta is None:
                    self.meta = data
                contents = data.get("contents") or []
            else:
                data = self.client.fetch_channel_contents_by_id(
                    self.ref.channel_id, self._page, self.per
                )
                if self.meta is None:
                    meta = self.client.fetch_channel_meta_by_id(self.ref.channel_id)
                    self.meta = meta or {}
                contents = data.get("contents", data if isinstance(data, list) else [])

            if not contents:
                break

            for item in contents:
                yield item

            if len(contents) < self.per:
                break
            self._page += 1


class ArenaPrintJob:
    def __init__(
        self,
        printer,
        scales: List[float],
        aligns: List[str],
        methods: List[str],
        ts_fmt: Optional[str],
        dithers: List[Optional[str]],
        thresholds: List[float],
        diffusions: List[float],
        brightness: List[float],
        contrast: List[float],
        gamma: List[float],
        autocontrast: bool,
        captions_str: Optional[str],
        spacing: int,
        footer_text: Optional[str],
        debug: bool = False,
        wrap_mode: str = "hyphen",
        auto_orient: bool = False,
        cut_between: bool = False,
        no_cut: bool = False,
    ):
        self.printer = printer
        self.scales = scales or [1.0]
        self.aligns = aligns or ["center"]
        self.methods = methods or ["raster"]
        self.ts_fmt = ts_fmt
        self.dithers = dithers or [None]
        self.thresholds = thresholds or [0.5]
        self.diffusions = diffusions or [1.0]
        self.brightness = brightness or [1.0]
        self.contrast = contrast or [1.0]
        self.gamma = gamma or [1.0]
        self.autocontrast = autocontrast
        self.captions = parse_caption_csv(captions_str)
        self.spacing = spacing
        self.footer_text = footer_text
        self.debug = debug
        self.wrap_mode = wrap_mode.lower() if wrap_mode else "hyphen"
        self.caption_index = 0
        self._footer_printed = False
        self.auto_orient_images = auto_orient
        self.cut_between_images = cut_between
        self.no_cut = no_cut

    def _wrap_and_count(
        self, text: str, width_mult: int = 1, height_mult: int = 1
    ) -> Tuple[str, int]:
        line_width = scaled_char_width(CHAR_WIDTH, width_mult)
        wrapped = wrap_text(text, line_width, self.wrap_mode)
        return wrapped, count_lines(wrapped, line_width) * height_mult

    def print_heading(
        self,
        text: Union[str, Iterable[str]],
        double: bool = True,
        trailing_blank: bool = False,
    ) -> None:
        if text is None:
            return
        if isinstance(text, str):
            raw_lines = text.splitlines()
        else:
            raw_lines = list(text)
        lines = [
            sanitize_output(line).encode("ascii", "ignore").decode("ascii")
            for line in raw_lines
            if line
        ]
        if not lines:
            return
        width_mult = 2 if double else 1
        height_mult = 2 if double else 1
        wrapped, line_count = self._wrap_and_count(
            "\n".join(lines), width_mult=width_mult, height_mult=height_mult
        )
        if wrapped:
            enforce_line_limit(line_count)
        for line in wrapped.splitlines():
            if not line:
                continue
            self.printer.set(
                align="left",
                font="b",
                bold=True,
                double_width=double,
                double_height=double,
                normal_textsize=True,
            )
            self.printer.text(ensure_trailing_newline(line))
        self.printer.set(
            align="left",
            font="a",
            bold=False,
            double_width=False,
            double_height=False,
            normal_textsize=True,
        )
        if trailing_blank:
            self.printer.text("\n")

    def print_center_bold(self, text: str) -> None:
        if not text:
            return
        sanitized = sanitize_output(text)
        sanitized = sanitized.encode("ascii", "ignore").decode("ascii")
        wrapped, line_count = self._wrap_and_count(sanitized)
        if wrapped:
            enforce_line_limit(line_count)
        self.printer.set(
            align="center",
            font="b",
            bold=True,
            double_width=False,
            double_height=False,
            normal_textsize=True,
        )
        self.printer.text(ensure_trailing_newline(wrapped))
        self.printer.set(
            align="left",
            font="a",
            bold=False,
            double_width=False,
            double_height=False,
            normal_textsize=True,
        )

    def print_text(
        self, text: str, *, align: str = "left", font: str = "a", bold: bool = False
    ) -> None:
        if not text:
            return
        sanitized = sanitize_output(text)
        sanitized = sanitized.encode("ascii", "ignore").decode("ascii")
        wrapped, line_count = self._wrap_and_count(sanitized)
        enforce_line_limit(line_count)
        self.printer.set(
            align=align,
            font=font,
            bold=bold,
            double_width=False,
            double_height=False,
            normal_textsize=True,
        )
        self.printer.text(ensure_trailing_newline(wrapped))
        self.printer.set(
            align="left",
            font="a",
            bold=False,
            double_width=False,
            double_height=False,
            normal_textsize=True,
        )

    def print_images(
        self, images: Iterable[Image.Image], names: Optional[Iterable[str]] = None
    ) -> None:
        img_list = list(images)
        if not img_list:
            return
        self.caption_index = print_images_from_pil(
            self.printer,
            img_list,
            self.scales,
            self.aligns,
            self.methods,
            self.ts_fmt,
            self.dithers,
            self.thresholds,
            self.diffusions,
            brightness_list=self.brightness,
            contrast_list=self.contrast,
            gamma_list=self.gamma,
            autocontrast=self.autocontrast,
            captions_list=self.captions,
            caption_start=self.caption_index,
            footer_text=None,
            debug=self.debug,
            spacing=0 if self.cut_between_images else self.spacing,
            names=names,
            wrap_mode=self.wrap_mode,
            auto_orient=self.auto_orient_images,
            cut_between=self.cut_between_images,
            no_cut=self.no_cut,
        )

    def line_break(self, count: int = 1) -> None:
        if count > 0:
            self.printer.text("\n" * count)

    def print_footer(self) -> None:
        if self.footer_text and not self._footer_printed:
            self.print_center_bold(self.footer_text)
            self._footer_printed = True


def video_frame_from_bytes(data: bytes, ffmpeg_path: Optional[str]) -> Image.Image:
    if not ffmpeg_path:
        raise ArenaDownloadError("ffmpeg not available for video frame extraction.")

    suffix = ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(data)
        input_path = tmp_in.name

    with NamedTemporaryFile(delete=False, suffix=".png") as tmp_out:
        output_path = tmp_out.name

    try:
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            output_path,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=MEDIA_TIMEOUT,
            check=True,
        )
        _ = proc  # silence unused warning
        img = Image.open(output_path)
        img.load()
        return img
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise ArenaDownloadError(f"ffmpeg failed to extract frame: {exc}") from exc
    finally:
        try:
            os.unlink(input_path)
        except Exception:
            pass
        try:
            os.unlink(output_path)
        except Exception:
            pass


def pdf_bytes_to_images(data: bytes, page_filter: Optional[Tuple]) -> List[Image.Image]:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        images, _ = pdf_to_images([tmp_path], page_filter)
        return images
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def guess_extension(content_type: Optional[str]) -> str:
    if not content_type:
        return ""
    ext = mimetypes.guess_extension(content_type)
    if ext:
        return ext
    if content_type.startswith("video/"):
        return ".mp4"
    if content_type == "application/pdf":
        return ".pdf"
    if content_type.startswith("image/"):
        return ".png"
    return ""


def block_connected_at(block: Dict[str, Any]) -> Optional[datetime]:
    ts = block.get("connected_at") or block.get("created_at")
    return parse_timestamp(ts)


def should_include_block(
    block: Dict[str, Any],
    include_channels: bool,
    filter_set: Optional[Iterable[str]] = None,
    exclude_set: Optional[Iterable[str]] = None,
) -> bool:
    klass = block_class(block)
    if klass == "channel" and not include_channels:
        return False
    if filter_set:
        normalized = {f.lower() for f in filter_set}
        if klass not in normalized:
            return False
    if exclude_set:
        normalized_excl = {e.lower() for e in exclude_set}
        if klass in normalized_excl:
            return False
    return True
