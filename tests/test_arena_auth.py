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
from receipt_print.arena_evaluate import fetch_channel_snapshot
from receipt_print.arena_document import expected_qr_payloads, normalize_channel
from receipt_print.receipt_core import SubmissionResult


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
    assert "--layout [paired|column|minimal|escpos]" in result.output
    assert "--channel-qr / --no-channel-qr" in result.output
    assert "--core-url TEXT" in result.output
    assert "--core-bin FILE" in result.output
    assert "--random-seed INTEGER" in result.output


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
            "--layout",
            "escpos",
            "--max-blocks",
            "2",
            "--no-cut",
            "https://www.are.na/example/private-channel",
        ],
    )

    assert result.exit_code == 0
    assert media_blocks == [1, 2]
    assert printer.flushes == 2


def test_arena_channel_defaults_to_core_column_full_channel_and_full_cut(
    monkeypatch,
):
    calls = {}
    value = normalize_channel(
        {
            "id": 7,
            "slug": "core-channel",
            "title": "Core Channel",
            "owner": {"full_name": "Owner"},
        },
        [{"id": 1, "type": "Text", "content": "one"}],
        "https://www.are.na/owner/core-channel",
        fetched_at="2026-08-08T12:00:00+00:00",
    )

    class Client:
        def __init__(self, cache_enabled=True):
            calls["cache_enabled"] = cache_enabled

        def close(self):
            calls["arena_closed"] = True

    class Core:
        def __init__(self, url, *, executable=None):
            calls["core_url"] = url
            calls["core_bin"] = executable

        def submit(self, document):
            calls["document"] = document
            return SubmissionResult(b"escpos", None, {})

        def close(self):
            calls["core_closed"] = True

    def fetch(client, channel, *, selection, limit, seed):
        calls["fetch"] = (channel, selection, limit, seed)
        return value, {}

    def media(client, channel):
        calls["media"] = channel.slug
        return {}

    monkeypatch.setattr(cli_module, "ArenaClient", Client)
    monkeypatch.setattr(cli_module, "ReceiptCoreClient", Core)
    monkeypatch.setattr(cli_module, "fetch_channel_snapshot", fetch)
    monkeypatch.setattr(cli_module, "collect_media", media)
    monkeypatch.setattr(
        cli_module,
        "print_raw_bytes",
        lambda data, *, cut: calls.setdefault("prints", []).append((data, cut)),
    )

    result = CliRunner().invoke(
        cli_module.cli,
        ["are.na", "channel", "https://www.are.na/owner/core-channel"],
    )

    assert result.exit_code == 0
    assert calls["fetch"] == (
        "https://www.are.na/owner/core-channel",
        "full",
        None,
        None,
    )
    assert calls["media"] == "core-channel"
    assert calls["core_url"] is None
    assert calls["core_bin"] is None
    assert calls["prints"] == [(b"escpos", False)]
    assert calls["document"]["realization"]["params"]["layout"] == "column"
    assert calls["document"]["realization"]["params"]["channelQr"] is True
    assert calls["document"]["blocks"][-1] == {"type": "cut", "kind": "full"}
    assert "as edition" not in result.output


def test_arena_channel_core_exposes_limit_media_and_channel_qr_controls(
    monkeypatch,
):
    calls = {}
    value = normalize_channel(
        {
            "id": 8,
            "slug": "limited",
            "title": "Limited",
            "owner": {"full_name": "Owner"},
        },
        [
            {"id": 1, "type": "Text", "content": "one"},
            {"id": 2, "type": "Text", "content": "two"},
        ],
        "https://www.are.na/owner/limited",
        fetched_at="2026-08-08T12:00:00+00:00",
    )

    class Client:
        def __init__(self, cache_enabled=True):
            pass

        def close(self):
            pass

    class Core:
        def __init__(self, url, *, executable=None):
            pass

        def submit(self, document):
            calls["document"] = document
            return SubmissionResult(None, None, {})

        def close(self):
            pass

    def fetch(client, channel, *, selection, limit, seed):
        calls["fetch"] = (selection, limit, seed)
        return value, {}

    def fail_media(client, channel):
        raise AssertionError("media collection should be disabled")

    monkeypatch.setattr(cli_module, "ArenaClient", Client)
    monkeypatch.setattr(cli_module, "ReceiptCoreClient", Core)
    monkeypatch.setattr(cli_module, "fetch_channel_snapshot", fetch)
    monkeypatch.setattr(cli_module, "collect_media", fail_media)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "are.na",
            "channel",
            "--layout",
            "paired",
            "--max-blocks",
            "2",
            "--no-channel-qr",
            "--no-media",
            "--no-cut",
            "https://www.are.na/owner/limited",
        ],
    )

    assert result.exit_code == 0
    assert calls["fetch"] == ("top", 2, None)
    document = calls["document"]
    assert document["realization"]["params"]["layout"] == "paired"
    assert document["realization"]["params"]["channelQr"] is False
    assert document["blocks"][-1]["type"] != "cut"
    assert "https://www.are.na/owner/limited" not in expected_qr_payloads(document)


def test_arena_channel_core_exposes_seeded_random_selection(monkeypatch):
    calls = {}
    channel = normalize_channel(
        {"id": 9, "slug": "sampled", "title": "Sampled"},
        [{"id": 2, "type": "Text", "content": "two"}],
        "https://www.are.na/owner/sampled",
        fetched_at="2026-08-08T12:00:00+00:00",
    )

    class Client:
        def __init__(self, cache_enabled=True):
            pass

        def close(self):
            pass

    class Core:
        def __init__(self, url, *, executable=None):
            pass

        def submit(self, document):
            calls["document"] = document
            calls["selection"] = document["realization"]["params"]["selection"]
            return SubmissionResult(None, None, {})

        def close(self):
            pass

    def fetch(client, value, *, selection, limit, seed):
        calls["fetch"] = (selection, limit, seed)
        return channel, {}

    monkeypatch.setattr(cli_module, "ArenaClient", Client)
    monkeypatch.setattr(cli_module, "ReceiptCoreClient", Core)
    monkeypatch.setattr(cli_module, "fetch_channel_snapshot", fetch)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "are.na",
            "channel",
            "--sort",
            "random",
            "--max-blocks",
            "20",
            "--random-seed",
            "17",
            "--no-media",
            "https://www.are.na/owner/sampled",
        ],
    )

    assert result.exit_code == 0
    assert calls["fetch"] == ("random", 20, 17)
    assert calls["selection"] == "random"
    assert calls["document"]["realization"]["params"]["order"] == (
        "pinned_first_position_desc"
    )


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

        def fetch_channel_contents_by_id(self, channel_id, page, per, sort):
            assert sort == "position_desc"
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


def test_top_selection_stops_on_first_page_and_preserves_api_order():
    class Client:
        def __init__(self):
            self.pages = []

        def fetch_channel_meta_by_slug(self, slug, page, per):
            return {
                "id": 7,
                "slug": slug,
                "title": "Ordered",
                "owner": {"slug": "owner", "full_name": "Owner"},
                "counts": {"contents": 8},
            }

        def fetch_channel_contents_by_id(self, channel_id, page, per, sort):
            self.pages.append((page, sort))
            pages = {
                1: {
                    "data": [
                        {
                            "id": item,
                            "type": "Text",
                            "content": str(item),
                            "connection": {
                                "position": 10 - item,
                                "pinned": item == 4,
                                "connected_at": f"2020-01-0{item + 1}T00:00:00Z",
                            },
                            "created_at": f"2026-01-0{item + 1}T00:00:00Z",
                        }
                        for item in range(5)
                    ],
                    "meta": {"next_page": 2, "has_more_pages": True},
                },
                2: {
                    "data": [{"id": 99, "type": "Text", "content": "later"}],
                    "meta": {"next_page": None, "has_more_pages": False},
                },
            }
            return pages[page]

    client = Client()

    channel, source = fetch_channel_snapshot(
        client,
        "https://www.are.na/owner/ordered",
        selection="top",
        limit=None,
    )

    assert client.pages == [(1, "position_desc")]
    assert [item.id for item in channel.blocks] == ["0", "1", "2", "3", "4"]
    assert channel.blocks[4].placement.pinned is True
    assert source["limit"] == 5


def test_random_selection_samples_full_channel_and_preserves_api_order():
    class Client:
        def __init__(self):
            self.pages = []

        def fetch_channel_meta_by_slug(self, slug, page, per):
            return {
                "id": 7,
                "slug": slug,
                "title": "Ordered",
                "owner": {"slug": "owner", "full_name": "Owner"},
                "counts": {"contents": 8},
            }

        def fetch_channel_contents_by_id(self, channel_id, page, per, sort):
            self.pages.append((page, sort))
            pages = {
                1: {
                    "data": [
                        {"id": item, "type": "Text", "content": str(item)}
                        for item in range(5)
                    ],
                    "meta": {"next_page": 2, "has_more_pages": True},
                },
                2: {
                    "data": [
                        {
                            "id": item,
                            "type": "Text",
                            "content": str(item),
                            "connection": {"pinned": item == 6},
                        }
                        for item in range(5, 8)
                    ],
                    "meta": {"next_page": None, "has_more_pages": False},
                },
            }
            return pages[page]

    client = Client()

    channel, source = fetch_channel_snapshot(
        client,
        "https://www.are.na/owner/ordered",
        selection="random",
        limit=3,
        seed=17,
    )

    assert client.pages == [(1, "position_desc"), (2, "position_desc")]
    assert [item.id for item in channel.blocks] == ["6", "2", "7"]
    assert channel.blocks[0].placement.pinned is True
    assert source["randomSeed"] == 17
    assert source["populationCount"] == 8
    assert source["pinOrder"] == "pinned-first"


def test_random_selection_keeps_small_channel_complete_and_ordered():
    class Client:
        def fetch_channel_meta_by_slug(self, slug, page, per):
            return {
                "id": 7,
                "slug": slug,
                "title": "Small",
                "owner": {"slug": "owner", "full_name": "Owner"},
                "counts": {"contents": 3},
            }

        def fetch_channel_contents_by_id(self, channel_id, page, per, sort):
            return {
                "data": [
                    {"id": item, "type": "Text", "content": str(item)}
                    for item in range(3)
                ],
                "meta": {"next_page": None, "has_more_pages": False},
            }

    channel, source = fetch_channel_snapshot(
        Client(),
        "https://www.are.na/owner/small",
        selection="random",
        limit=20,
        seed=17,
    )

    assert [item.id for item in channel.blocks] == ["0", "1", "2"]
    assert source["populationCount"] == 3
