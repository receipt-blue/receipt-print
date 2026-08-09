import io

from click.testing import CliRunner

from receipt_print import cli as cli_module
from receipt_print import printer as printer_module


class FakeLock:
    instances = []

    def __init__(self):
        self.acquired = 0
        self.released = 0
        self.instances.append(self)

    def acquire(self):
        self.acquired += 1

    def release(self):
        self.released += 1


class FakePrinter:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class RecordingPrinter:
    def __init__(self):
        self.texts = []
        self.cuts = 0
        self.cut_modes = []
        self.closed = 0

    def set(self, **_kwargs):
        return None

    def text(self, value):
        self.texts.append(value)

    def cut(self, mode=None):
        self.cuts += 1
        self.cut_modes.append(mode)

    def close(self):
        self.closed += 1


def test_line_limit_confirmation_covers_the_rest_of_the_job(monkeypatch, capsys):
    prompts = 0

    def open_tty(*_args, **_kwargs):
        nonlocal prompts
        prompts += 1
        return io.StringIO("yes\n")

    monkeypatch.setattr("builtins.open", open_tty)
    printer_module.reset_line_limit_confirmation()

    printer_module.enforce_line_limit(printer_module.MAX_LINES + 1)
    printer_module.enforce_line_limit(
        printer_module.MAX_LINES + 3000,
        unit="image-lines",
    )

    assert prompts == 1
    assert capsys.readouterr().out.count("without further size warnings") == 1


def test_service_mode_uses_service_when_ready(monkeypatch):
    monkeypatch.setenv("RP_PRINT_MODE", "service")
    monkeypatch.setattr("receipt_print.routing.service_ready", lambda url: True)
    monkeypatch.setattr("receipt_print.routing.submit_raw", lambda *args, **kwargs: None)
    monkeypatch.setattr(printer_module, "_resolve_speed", lambda: None)
    remote = printer_module.connect_printer()
    assert remote.__class__.__name__ == "ServicePrinter"
    remote.close()


def test_default_mode_uses_direct_printer_without_probing_service(monkeypatch):
    FakeLock.instances.clear()
    physical = FakePrinter()
    monkeypatch.delenv("RP_PRINT_MODE", raising=False)
    monkeypatch.setattr(
        "receipt_print.routing.service_ready",
        lambda url: (_ for _ in ()).throw(AssertionError("service path probed")),
    )
    monkeypatch.setattr("receipt_print.routing.DeviceLock", FakeLock)
    monkeypatch.setattr(printer_module, "connect_direct_printer", lambda: physical)

    leased = printer_module.connect_printer()
    assert FakeLock.instances[0].acquired == 1
    leased.close()
    assert physical.closed == 1
    assert FakeLock.instances[0].released == 1


def test_direct_mode_uses_locked_device(monkeypatch):
    FakeLock.instances.clear()
    physical = FakePrinter()
    monkeypatch.setenv("RP_PRINT_MODE", "direct")
    monkeypatch.setattr("receipt_print.routing.DeviceLock", FakeLock)
    monkeypatch.setattr(printer_module, "connect_direct_printer", lambda: physical)

    leased = printer_module.connect_printer()
    assert FakeLock.instances[0].acquired == 1
    leased.close()
    assert physical.closed == 1
    assert FakeLock.instances[0].released == 1


def test_service_mode_never_falls_back_to_device(monkeypatch):
    monkeypatch.setenv("RP_PRINT_MODE", "service")
    monkeypatch.setattr("receipt_print.routing.service_ready", lambda url: False)
    monkeypatch.setattr(
        printer_module,
        "connect_direct_printer",
        lambda: (_ for _ in ()).throw(AssertionError("direct path used")),
    )
    try:
        printer_module.connect_printer()
        raised = False
    except RuntimeError as error:
        raised = True
        assert "not ready" in str(error)
    assert raised


def test_open_usb_filters_device_and_bounds_writes(monkeypatch):
    calls = []

    class FakeUsb:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def open(self):
            return None

        def charcode(self, _value):
            return None

    monkeypatch.setattr(printer_module, "Usb", FakeUsb)
    monkeypatch.setattr(printer_module, "_apply_speed", lambda _printer, _speed: None)

    backend = object()
    printer_module._open_usb(0x04B8, 0x0E2B, None, backend=backend)

    assert calls == [
        {
            "idVendor": 0x04B8,
            "idProduct": 0x0E2B,
            "profile": printer_module.PRINTER_PROFILE,
            "timeout": 30000,
            "usb_args": {"backend": backend},
        }
    ]


def test_direct_raw_output_is_chunked(monkeypatch):
    writes = []

    class RawPrinter:
        def _raw(self, data):
            writes.append(data)

        def close(self):
            return None

    monkeypatch.setattr(
        printer_module,
        "_connect_locked_direct_printer",
        lambda: RawPrinter(),
    )
    payload = b"x" * (printer_module.RAW_CHUNK_BYTES * 2 + 7)

    printer_module.print_raw_bytes_direct(payload)

    assert [len(chunk) for chunk in writes] == [
        printer_module.RAW_CHUNK_BYTES,
        printer_module.RAW_CHUNK_BYTES,
        7,
    ]


def test_direct_raw_output_retries_connection_before_writing(monkeypatch):
    attempts = 0
    writes = []

    class RawPrinter:
        def _raw(self, data):
            writes.append(data)

        def close(self):
            return None

    def connect():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise printer_module.PrinterUnavailableError("not ready")
        return RawPrinter()

    monkeypatch.setattr(printer_module, "_connect_locked_direct_printer", connect)
    monkeypatch.setattr(printer_module.time, "sleep", lambda _seconds: None)

    printer_module.print_raw_bytes_direct(b"receipt")

    assert attempts == 3
    assert writes == [b"receipt"]


def test_no_cut_text_terminates_the_final_line(monkeypatch):
    physical = RecordingPrinter()
    monkeypatch.setattr(printer_module, "connect_printer", lambda: physical)

    printer_module.print_text("hello", no_cut=True)

    assert physical.texts == ["hello", "\n"]
    assert physical.cuts == 0
    assert physical.closed == 1


def test_no_cut_text_does_not_add_a_second_line_feed(monkeypatch):
    physical = RecordingPrinter()
    monkeypatch.setattr(printer_module, "connect_printer", lambda: physical)

    printer_module.print_text("hello\n", no_cut=True)

    assert physical.texts == ["hello\n"]
    assert physical.cuts == 0
    assert physical.closed == 1


def test_partial_cut_text_uses_partial_escpos_mode(monkeypatch):
    physical = RecordingPrinter()
    monkeypatch.setattr(printer_module, "connect_printer", lambda: physical)

    printer_module.print_text("hello", partial_cut=True)

    assert physical.cut_modes == ["PART"]
    assert physical.closed == 1


def test_cli_partial_cut_uses_partial_escpos_mode(monkeypatch):
    physical = RecordingPrinter()
    monkeypatch.setattr(printer_module, "connect_printer", lambda: physical)

    result = CliRunner().invoke(
        cli_module.cli, ["text", "--partial-cut", "hello"]
    )

    assert result.exit_code == 0
    assert physical.cut_modes == ["PART"]


def test_cli_partial_cut_is_mutually_exclusive_with_no_cut(monkeypatch):
    physical = RecordingPrinter()
    monkeypatch.setattr(cli_module, "connect_printer", lambda: physical)

    result = CliRunner().invoke(
        cli_module.cli, ["text", "--no-cut", "--partial-cut", "hello"]
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
    assert physical.closed == 0
