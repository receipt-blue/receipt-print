from receipt_print.journal import PrintJournal


def test_printed_job_replays_without_reclaiming(tmp_path):
    journal = PrintJournal(str(tmp_path / "jobs.sqlite3"))
    first = journal.claim("job-1", b"receipt")
    assert first.state == "printing"
    assert first.replayed is False
    journal.finish("job-1", "printed")

    replay = journal.claim("job-1", b"receipt")
    assert replay.state == "printed"
    assert replay.replayed is True


def test_job_identity_rejects_different_bytes(tmp_path):
    journal = PrintJournal(str(tmp_path / "jobs.sqlite3"))
    journal.claim("job-1", b"first")
    journal.finish("job-1", "printed")
    conflict = journal.claim("job-1", b"second")
    assert conflict.state == "conflict"
    assert conflict.replayed is True


def test_restart_marks_inflight_job_ambiguous(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    journal = PrintJournal(str(path))
    journal.claim("job-1", b"receipt")

    recovered = PrintJournal(str(path)).claim("job-1", b"receipt")
    assert recovered.state == "ambiguous"
    assert recovered.replayed is True
