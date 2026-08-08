import base64
import io
import json

from PIL import Image

from receipt_print.arena_document import normalize_channel
from receipt_print.arena_evaluate import evaluate_channel
from receipt_print.receipt_core import PreviewResult


class Core:
    def __init__(self):
        image = Image.new("1", (480, 200), 1)
        output = io.BytesIO()
        image.save(output, format="PNG")
        self.png = base64.b64encode(output.getvalue()).decode("ascii")
        self.studies = []

    def preview(self, document, *, study=None):
        self.studies.append(study)
        payload = "https://www.are.na/block/1"
        return PreviewResult(
            png=self.png,
            pngs=(self.png,),
            text="preview",
            economy={"dotLines": 200, "lengthMm": 25, "estSeconds": 1},
            regions=(
                {
                    "path": "/blocks/1",
                    "page": 0,
                    "bounds": {"x": 0, "y": 0, "width": 480, "height": 100},
                    "ref": {
                        "source": "arena",
                        "id": "1",
                        "note": "pinned connection",
                    },
                },
            ),
            qr_results=(
                {"payload": payload, "decoded": True, "path": "/blocks/1"},
            ),
        )


def test_evaluator_writes_replayable_artifacts_and_study_metadata(tmp_path):
    channel = normalize_channel(
        {
            "id": 7,
            "slug": "evaluation",
            "title": "Evaluation",
            "owner": {"full_name": "Owner"},
            "counts": {"contents": 1},
        },
        [
            {
                "id": 1,
                "type": "Text",
                "content": {"markdown": "Pinned body"},
                "connection": {"position": 1, "pinned": True},
            }
        ],
        "https://www.are.na/owner/evaluation",
        fetched_at="2026-08-08T12:00:00+00:00",
    )
    core = Core()

    artifacts = evaluate_channel(
        channel,
        {"contents": []},
        core,
        tmp_path,
        variants=("paired", "column", "minimal"),
    )

    channel_dir = tmp_path / "evaluation"
    manifest = json.loads((channel_dir / "manifest.json").read_text())
    assert len(artifacts) == 3
    assert (channel_dir / "snapshot.json").exists()
    assert (channel_dir / "source-raw.json").exists()
    assert (channel_dir / "contact-sheet.png").exists()
    assert (channel_dir / "pin-treatment-sheet.png").exists()
    assert core.studies == [None, None, None]
    assert manifest["order"] == "position_desc"
    assert manifest["pinTreatmentSheet"] == "pin-treatment-sheet.png"
    qr_report = json.loads(
        (channel_dir / "paired" / "qr-report.json").read_text()
    )
    assert any(code["decoded"] is True for code in qr_report["codes"])
    assert any(code["status"] == "decoded" for code in qr_report["codes"])
