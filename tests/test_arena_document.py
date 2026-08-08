from receipt_print.arena_document import (
    ArenaBlock,
    ArenaChannel,
    block_destinations,
    compose_channel_document,
    expected_qr_payloads,
    normalize_channel,
)


def block(
    block_id,
    *,
    kind="Text",
    title=None,
    source=None,
    attachment=None,
    position=None,
    pinned=False,
    connected_at=None,
    created_at=None,
    updated_at=None,
    description=None,
    state="available",
):
    value = {
        "id": block_id,
        "type": kind,
        "title": title or f"Block {block_id}",
        "content": {"markdown": f"Content {block_id}"},
        "state": state,
        "created_at": created_at,
        "updated_at": updated_at,
        "user": {"full_name": "Creator"},
        "connection": {
            "position": position,
            "pinned": pinned,
            "connected_at": connected_at,
            "connected_by": {"full_name": "Connector"},
        },
    }
    if source:
        value["source"] = {"url": source}
    if attachment:
        value["attachment"] = {"url": attachment}
    if description:
        value["description"] = description
    return value


def channel(raw_blocks):
    return normalize_channel(
        {
            "id": 7,
            "slug": "manual-order",
            "title": "Manual Order",
            "owner": {"full_name": "Owner"},
            "counts": {"contents": 12},
        },
        raw_blocks,
        "https://www.are.na/owner/manual-order",
        fetched_at="2026-08-08T12:00:00+00:00",
    )


def test_normalization_preserves_returned_manual_order_and_pin_state():
    value = channel(
        [
            block(3, position=3, connected_at="2025-01-01T00:00:00Z"),
            block(
                1,
                position=1,
                pinned=True,
                connected_at="2026-01-01T00:00:00Z",
            ),
            block(2, position=2, connected_at="2024-01-01T00:00:00Z"),
        ]
    )

    assert [item.id for item in value.blocks] == ["3", "1", "2"]
    assert value.blocks[1].placement.pinned is True
    assert value.blocks[1].placement.position == 1


def test_external_source_is_primary_and_arena_is_context():
    value = ArenaBlock.from_api(
        block(11, kind="Link", source="https://example.org/article")
    )

    destinations = block_destinations(value)

    assert [item["role"] for item in destinations] == ["primary", "context"]
    assert destinations[0]["url"] == "https://example.org/article"
    assert destinations[1]["url"] == "https://www.are.na/block/11"


def test_attachment_is_used_when_no_external_source_exists():
    value = ArenaBlock.from_api(
        block(12, kind="Attachment", attachment="https://attachments.are.na/file.pdf")
    )

    destinations = block_destinations(value)

    assert destinations[0]["kind"] == "attachment"
    assert destinations[1]["kind"] == "arena"


def test_linked_channel_keeps_the_arena_block_as_context():
    raw = block(121, kind="Channel")
    raw.update(
        {
            "slug": "linked-channel",
            "owner": {"slug": "linked-owner"},
        }
    )
    value = ArenaBlock.from_api(raw)

    destinations = block_destinations(value)

    assert [item["url"] for item in destinations] == [
        "https://www.are.na/linked-owner/linked-channel",
        "https://www.are.na/block/121",
    ]


def test_failed_source_promotes_arena_context():
    value = ArenaBlock.from_api(
        block(
            13,
            kind="Link",
            source="https://example.org/missing",
            state="failure",
        )
    )

    assert block_destinations(value) == [
        {
            "role": "primary",
            "kind": "arena",
            "url": "https://www.are.na/block/13",
            "title": "block",
            "detail": "13",
        }
    ]


def test_column_document_exposes_pin_and_standard_destinations():
    value = channel(
        [
            block(
                21,
                kind="Link",
                source="https://example.org/source",
                pinned=True,
            )
        ]
    )

    document = compose_channel_document(value, "column")
    flow = next(
        item
        for item in document["blocks"]
        if item["type"] == "qr-flow" and item["side"] == "right"
    )

    assert flow["lead"][0] == {
        "type": "rule",
        "weight": "light",
        "ref": flow["ref"],
    }
    assert flow["lead"][1]["runs"][0] == {
        "text": "Block 21",
        "bold": True,
    }
    assert flow["lead"][1]["align"] == "center"
    assert flow["lead"][1]["marker"] == "pin"
    assert [item["role"] for item in flow["destinations"]] == [
        "primary",
        "context",
    ]
    assert [
        "".join(run["text"] for run in item["caption"]) for item in flow["destinations"]
    ] == [
        "source",
        "block 21",
    ]
    channel_flow = document["blocks"][0]
    assert channel_flow["type"] == "qr-flow"
    assert channel_flow["side"] == "left"
    assert "caption" not in channel_flow["destinations"][0]
    assert channel_flow["destinations"][0]["payload"] == (
        "https://www.are.na/owner/manual-order"
    )


def test_minimal_is_qr_free_by_default_and_paired_has_block_and_channel_qrs():
    value = channel([block(31)])

    minimal = compose_channel_document(value, "minimal")
    with_qrs = compose_channel_document(value, "paired")

    assert all(item["type"] not in {"qr-rail", "qr-flow"} for item in minimal["blocks"])
    assert expected_qr_payloads(minimal) == []
    assert expected_qr_payloads(with_qrs) == [
        "https://www.are.na/owner/manual-order",
        "https://www.are.na/block/31",
    ]


def test_channel_qr_can_be_enabled_or_disabled_in_every_layout():
    value = channel([block(311)])

    for variant in ("column", "paired", "minimal"):
        with_qr = compose_channel_document(value, variant, channel_qr=True)
        without_qr = compose_channel_document(value, variant, channel_qr=False)

        assert expected_qr_payloads(with_qr)[0] == (
            "https://www.are.na/owner/manual-order"
        )
        assert "https://www.are.na/owner/manual-order" not in expected_qr_payloads(
            without_qr
        )


def test_column_keeps_arena_only_destination_at_standard_size():
    value = channel([block(32)])

    document = compose_channel_document(value, "column")
    flow = next(
        item
        for item in document["blocks"]
        if item["type"] == "qr-flow" and item["side"] == "right"
    )

    assert len(flow["destinations"]) == 1
    assert "size" not in flow["destinations"][0]
    assert flow["destinations"][0]["caption"] == [
        {"text": "block ", "font": "b", "bold": True},
        {"text": "32", "font": "b"},
    ]


def test_snapshot_round_trip_preserves_nested_dataclasses():
    original = channel([block(41, pinned=True)])

    restored = ArenaChannel.from_dict(original.to_dict())

    assert restored == original


def test_normalization_folds_styled_unicode_without_splitting_emoji_sequences():
    value = channel(
        [
            {
                **block(42),
                "description": "𝘛𝘩𝘦 𝘕𝘦𝘸 𝘖𝘳𝘢𝘭𝘪𝘵𝘺 👮🏽‍♂️",
            }
        ]
    )

    assert value.blocks[0].description == "*The New Orality* 👮🏽‍♂️"

    snapshot = value.to_dict()
    snapshot["blocks"][0]["description"] = "𝘛𝘩𝘦 𝘕𝘦𝘸 𝘖𝘳𝘢𝘭𝘪𝘵𝘺 👮🏽‍♂️"
    document = compose_channel_document(ArenaChannel.from_dict(snapshot), "column")
    flow = next(
        item
        for item in document["blocks"]
        if item["type"] == "qr-flow" and item["side"] == "right"
    )
    body_text = "".join(run["text"] for run in flow["body"]["runs"])
    assert body_text.startswith("The New Orality 👮🏽‍♂️")
    italic_run = next(
        run for run in flow["body"]["runs"] if run["text"] == "The New Orality"
    )
    assert italic_run["italic"] is True


def test_paired_preserves_all_items_in_returned_order():
    value = channel([block(item) for item in range(8)])

    document = compose_channel_document(value, "paired")
    block_qrs = [
        item
        for item in document["blocks"]
        if item["type"] == "qr" and item["ref"].get("note") == "arena"
    ]

    assert [item["ref"]["id"] for item in block_qrs] == [str(item) for item in range(8)]
    assert document["realization"]["params"]["order"] == "position_desc"


def test_header_and_metadata_follow_arena_anatomy():
    value = channel(
        [
            block(
                51,
                connected_at="2026-07-06T20:08:43Z",
                updated_at="2026-07-07T20:08:43Z",
            )
        ]
    )

    document = compose_channel_document(value, "paired")
    text = [
        "".join(run["text"] for run in item["runs"])
        for item in document["blocks"]
        if item["type"] == "text"
    ]

    channel_flow = document["blocks"][0]
    assert channel_flow["body"]["runs"] == [
        {
            "text": "Owner / Manual Order",
            "bold": True,
            "size": {"w": 1.12, "h": 1.12},
        }
    ]
    assert "header" not in document
    assert not any("channel order" in line for line in text)
    assert "Added6 Jul 2026" in text
    assert "Modified7 Jul 2026" in text
    assert "ByCreator" in text
    assert "Added byConnector" in text
    assert "SourceAre.na" in text

    metadata = {
        item["runs"][0]["text"]: item["runs"][1]
        for item in document["blocks"]
        if item["type"] == "text"
        and len(item["runs"]) == 2
        and item["runs"][0].get("font") == "b"
    }
    assert metadata["Added"] == {
        "text": "6 Jul 2026",
        "font": "b",
        "dock": "right",
    }
    assert metadata["Modified"] == {
        "text": "7 Jul 2026",
        "font": "b",
        "dock": "right",
    }
    assert metadata["By"]["bold"] is True
    assert metadata["Added by"]["bold"] is True
    assert metadata["Source"]["bold"] is True


def test_channel_description_starts_full_width_below_the_qr_title_lockup():
    value = normalize_channel(
        {
            "id": 7,
            "slug": "manual-order",
            "title": "Manual Order",
            "description": "Introductory resources for Comms Braintrust",
            "owner": {"full_name": "Owner"},
        },
        [block(52)],
        "https://www.are.na/owner/manual-order",
        fetched_at="2026-08-08T12:00:00+00:00",
    )

    document = compose_channel_document(value, "column")

    assert document["blocks"][0]["type"] == "qr-flow"
    assert document["blocks"][1] == {
        "type": "text",
        "runs": [{"text": "Introductory resources for Comms Braintrust"}],
        "wrap": "word",
        "ref": {
            "source": "arena",
            "id": "7",
            "note": "channel description",
        },
    }


def test_markdown_semantics_become_styled_runs_without_source_punctuation():
    value = channel(
        [
            block(
                53,
                kind="Link",
                description=(
                    "Read **closely** and visit [IG](https://instagram.com/example).\n\n"
                    "> A quoted *thought*.\n"
                    "> \u2060\n"
                    "> Date: Sat 16 May\u2060\n"
                    "> Time: 13.30–18.00\u2060\n"
                    "> @htmx_org - [https://x.com/htmx_org/status/1](https://x.com/htmx_org/status/1)\n"
                    "> \\* Footnote by [profile](https://example.com/profile)\n\n"
                    "- first item\n- second `item`"
                ),
            )
        ]
    )

    document = compose_channel_document(value, "column")
    flow = next(
        item
        for item in document["blocks"]
        if item["type"] == "qr-flow" and item["side"] == "right"
    )
    runs = flow["body"]["runs"]
    text = "".join(run["text"] for run in runs)

    assert "**" not in text
    assert "[IG]" not in text
    assert "https://instagram.com/example" not in text
    assert "https://example.com/profile" not in text
    assert "\\*" not in text
    assert "> A quoted thought" not in text
    assert "Read closely and visit IG." in text
    assert (
        "A quoted thought.\n\n"
        "Date: Sat 16 May\n"
        "Time: 13.30–18.00\n"
        "@htmx_org - https://x.com/htmx_org/status/1\n"
        "* Footnote"
    ) in text
    assert "• first item" in text
    assert "• second item" in text
    assert next(run for run in runs if run["text"] == "closely")["bold"] is True
    assert next(run for run in runs if run["text"] == "IG")["bold"] is True
    assert next(run for run in runs if run["text"] == "profile")["bold"] is True
    assert next(run for run in runs if run["text"] == "thought")["italic"] is True
    assert all(
        run["quote"] is True
        for run in runs
        if run["text"]
        in {
            "A quoted ",
            "thought",
            (".\n\nDate: Sat 16 May\nTime: 13.30–18.00\n@htmx_org - "),
            "https://x.com/htmx_org/status/1",
            "\n* Footnote by ",
            "profile",
        }
    )
    assert next(run for run in runs if run["text"] == "item")["font"] == "b"


def test_arena_markdown_preserves_authored_lines_and_decodes_entities():
    raw = block(
        531,
        description=(
            "Description first line\n"
            "Description second line &amp; continuation.<br>We've barely begun."
        ),
    )
    raw["content"] = {
        "markdown": (
            "Core Canon:\nOrality and Literacy - Ong\nUnderstanding Media - McLuhan"
        )
    }
    value = channel([raw])

    document = compose_channel_document(value, "column")
    flow = next(
        item
        for item in document["blocks"]
        if item["type"] == "qr-flow" and item["side"] == "right"
    )
    content = "".join(run["text"] for run in flow["lead"][2]["runs"])
    text = "".join(run["text"] for run in flow["body"]["runs"])

    assert content == (
        "Core Canon:\nOrality and Literacy - Ong\nUnderstanding Media - McLuhan"
    )
    assert text.startswith(
        "Description first line\n"
        "Description second line & continuation.\nWe've barely begun."
    )


def test_print_document_records_selection_channel_qr_and_cut():
    value = channel([block(54)])

    document = compose_channel_document(
        value,
        "minimal",
        channel_qr=True,
        selection="top",
        cut="partial",
    )

    assert document["realization"]["params"] == {
        "layout": "minimal",
        "order": "position_desc",
        "selection": "top",
        "count": 1,
        "channelQr": True,
    }
    assert document["blocks"][-1] == {"type": "cut", "kind": "partial"}


def test_column_keeps_title_and_text_content_in_the_full_width_lead():
    value = channel([block(61, kind="Text", description="A secondary note")])

    document = compose_channel_document(value, "column")
    flow = next(
        item
        for item in document["blocks"]
        if item["type"] == "qr-flow" and item["side"] == "right"
    )

    assert [item["runs"][0]["text"] for item in flow["lead"][1:3]] == [
        "Block 61",
        "Content 61",
    ]
    assert flow["lead"][3] == {
        "type": "feed",
        "dots": 8,
        "ref": {"source": "arena", "id": "61"},
    }
    body_text = "".join(run["text"] for run in flow["body"]["runs"])
    assert body_text.startswith("A secondary note\n\n")
    assert body_text.endswith("SourceAre.na")
    source_run = next(
        run for run in flow["body"]["runs"] if run["text"] == "Are.na"
    )
    assert source_run == {
        "text": "Are.na",
        "font": "b",
        "dock": "right",
        "bold": True,
    }


def test_column_keeps_image_in_the_full_width_lead_before_description_flow():
    value = channel(
        [
            block(
                62,
                kind="Image",
                source="https://example.org/image",
                description="A long image description",
            )
        ]
    )

    document = compose_channel_document(
        value,
        "column",
        media={"62": "data:image/png;base64,AA=="},
    )
    flow = next(
        item
        for item in document["blocks"]
        if item["type"] == "qr-flow" and item["side"] == "right"
    )

    assert [item["type"] for item in flow["lead"]] == ["rule", "text", "image"]
    assert flow["lead"][2]["width"] == "full"
    assert flow["lead"][2]["spacing"] == {"beforeDots": 8, "afterDots": 16}
    assert flow["body"]["runs"][0]["text"] == "A long image description"
