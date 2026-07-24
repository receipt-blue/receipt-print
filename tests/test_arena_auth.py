import hashlib
import json
from datetime import datetime, timedelta, timezone

from click.testing import CliRunner

from receipt_print import cli as cli_module
from receipt_print.arena import (
    ArenaClient,
    ChannelIterator,
    ChannelRef,
    block_class,
    block_connected_at,
    block_description,
    block_preview_urls,
    block_text_content,
    resolve_arena_access_token,
)


TOKEN_IDENTITY = "arena-user:active"


def store_contextualize_token(cache_root, token, expires_at):
    token_dir = cache_root / "token"
    token_dir.mkdir(parents=True)
    digest = hashlib.sha256(TOKEN_IDENTITY.encode("utf-8")).hexdigest()
    (token_dir / f"{digest}.json").write_text(
        json.dumps({"access_token": token}),
        encoding="utf-8",
    )
    (token_dir / f"{digest}.meta.json").write_text(
        json.dumps(
            {
                "cache_version": 1,
                "expires_at": expires_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def clear_arena_auth_env(monkeypatch):
    monkeypatch.delenv("ARENA_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ARENA_TOKEN", raising=False)


def test_arena_channel_exposes_max_blocks_alias():
    result = CliRunner().invoke(cli_module.cli, ["are.na", "channel", "--help"])

    assert result.exit_code == 0
    assert "--limit, --max-blocks INTEGER" in result.output
    assert "media for later" in result.output
    assert "blocks is not downloaded" in result.output


def test_arena_channel_max_blocks_stops_before_later_media(monkeypatch):
    media_blocks = []

    class Client:
        authenticated = True

        def __init__(self, cache_enabled=True):
            pass

        def fetch_channel_meta_by_slug(self, slug, page, per):
            return {
                "id": 7,
                "slug": slug,
                "title": "Private Channel",
                "owner": {"name": "Example User"},
            }

        def close(self):
            pass

    class Iterator:
        def __init__(self, client, ref):
            pass

        def __iter__(self):
            return iter(
                [
                    {"id": 1, "type": "Text", "content": "one"},
                    {"id": 2, "type": "Text", "content": "two"},
                    {"id": 3, "type": "Image"},
                ]
            )

    class Printer:
        def __init__(self):
            self.flushes = 0

        def set(self, **kwargs):
            pass

        def text(self, value):
            pass

        def flush_pending(self):
            self.flushes += 1

        def close(self):
            pass

    def gather(block, client, media_options):
        media_blocks.append(block["id"])
        return [], []

    monkeypatch.setattr(cli_module, "ArenaClient", Client)
    monkeypatch.setattr(cli_module, "ChannelIterator", Iterator)
    printer = Printer()
    monkeypatch.setattr(cli_module, "connect_printer", lambda: printer)
    monkeypatch.setattr(cli_module, "gather_images_for_block", gather)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "are.na",
            "channel",
            "--max-blocks",
            "2",
            "--no-cut",
            "https://www.are.na/example/private-channel",
        ],
    )

    assert result.exit_code == 0
    assert media_blocks == [1, 2]
    assert printer.flushes == 2


def test_resolve_arena_access_token_prefers_contextualize_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTUALIZE_ARENA_CACHE", str(tmp_path))
    monkeypatch.setenv("ARENA_ACCESS_TOKEN", "contextualize-token")
    monkeypatch.setenv("ARENA_TOKEN", "legacy-token")

    assert resolve_arena_access_token() == "contextualize-token"


def test_resolve_arena_access_token_supports_legacy_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXTUALIZE_ARENA_CACHE", str(tmp_path))
    monkeypatch.delenv("ARENA_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("ARENA_TOKEN", "legacy-token")

    assert resolve_arena_access_token() == "legacy-token"


def test_resolve_arena_access_token_reads_contextualize_cache(monkeypatch, tmp_path):
    clear_arena_auth_env(monkeypatch)
    monkeypatch.setenv("CONTEXTUALIZE_ARENA_CACHE", str(tmp_path))
    store_contextualize_token(
        tmp_path,
        "stored-token",
        datetime.now(timezone.utc) + timedelta(hours=1),
    )

    assert resolve_arena_access_token() == "stored-token"


def test_resolve_arena_access_token_ignores_expiring_cache(monkeypatch, tmp_path):
    clear_arena_auth_env(monkeypatch)
    monkeypatch.setenv("CONTEXTUALIZE_ARENA_CACHE", str(tmp_path))
    store_contextualize_token(
        tmp_path,
        "expiring-token",
        datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    assert resolve_arena_access_token() is None


def test_arena_client_uses_resolved_token(monkeypatch, tmp_path):
    clear_arena_auth_env(monkeypatch)
    monkeypatch.setenv("CONTEXTUALIZE_ARENA_CACHE", str(tmp_path))
    store_contextualize_token(
        tmp_path,
        "stored-token",
        datetime.now(timezone.utc) + timedelta(hours=1),
    )

    client = ArenaClient(cache_enabled=False)

    assert client.authenticated is True
    assert client.session.headers["Authorization"] == "Bearer stored-token"
    client.close()


def test_v3_block_fields_are_normalized():
    block = {
        "type": "Image",
        "description": {
            "html": "<p>Structured description</p>",
            "markdown": "Structured **description**",
            "plain": "Structured description",
        },
        "image": {
            "large": {"src": "https://example.test/large.jpg"},
            "small": {"src": "https://example.test/small.jpg"},
        },
        "connection": {"connected_at": "2026-07-24T10:00:00Z"},
    }

    assert block_class(block) == "image"
    assert block_description(block) == "Structured **description**"
    assert block_preview_urls(block) == [
        "https://example.test/large.jpg",
        "https://example.test/small.jpg",
    ]
    assert block_connected_at(block) == datetime(
        2026, 7, 24, 10, 0, tzinfo=timezone.utc
    )


def test_v3_structured_text_content_is_normalized():
    block = {
        "content": {
            "html": "<p>Receipt body</p>",
            "markdown": "Receipt **body**",
            "plain": "Receipt body",
        }
    }

    assert block_text_content(block) == "Receipt **body**"


def test_channel_iterator_reads_v3_data_pages():
    class Client:
        def fetch_channel_meta_by_slug(self, slug, page, per):
            return {"id": 7, "slug": slug}

        def fetch_channel_contents_by_id(self, channel_id, page, per):
            pages = {
                1: {
                    "data": [{"id": 1}],
                    "meta": {"next_page": 2, "has_more_pages": True},
                },
                2: {
                    "data": [{"id": 2}],
                    "meta": {"next_page": None, "has_more_pages": False},
                },
            }
            return pages[page]

    iterator = ChannelIterator(Client(), ChannelRef(slug="private-channel"), per=1)

    assert list(iterator) == [{"id": 1}, {"id": 2}]
    assert iterator.meta == {"id": 7, "slug": "private-channel"}
