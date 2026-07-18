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
        self.closed = 0

    def set(self, **_kwargs):
        return None

    def text(self, value):
        self.texts.append(value)

    def cut(self):
        self.cuts += 1

    def close(self):
        self.closed += 1


def test_auto_mode_uses_service_when_ready(monkeypatch):
    monkeypatch.setenv("RP_PRINT_MODE", "auto")
    monkeypatch.setattr("receipt_print.routing.service_ready", lambda url: True)
    monkeypatch.setattr("receipt_print.routing.submit_raw", lambda *args, **kwargs: None)
    monkeypatch.setattr(printer_module, "_resolve_speed", lambda: None)
    remote = printer_module.connect_printer()
    assert remote.__class__.__name__ == "ServicePrinter"
    remote.close()


def test_auto_mode_rechecks_service_while_holding_device_lease(monkeypatch):
    FakeLock.instances.clear()
    readiness = iter([False, True])
    monkeypatch.setenv("RP_PRINT_MODE", "auto")
    monkeypatch.setattr(
        "receipt_print.routing.service_ready", lambda url: next(readiness)
    )
    monkeypatch.setattr("receipt_print.routing.DeviceLock", FakeLock)
    monkeypatch.setattr("receipt_print.routing.submit_raw", lambda *args, **kwargs: None)
    monkeypatch.setattr(printer_module, "_resolve_speed", lambda: None)
    remote = printer_module.connect_printer()
    assert remote.__class__.__name__ == "ServicePrinter"
    assert FakeLock.instances[0].acquired == 1
    assert FakeLock.instances[0].released == 1
    remote.close()


def test_auto_mode_preserves_locked_direct_fallback(monkeypatch):
    FakeLock.instances.clear()
    physical = FakePrinter()
    monkeypatch.setenv("RP_PRINT_MODE", "auto")
    monkeypatch.setattr("receipt_print.routing.service_ready", lambda url: False)
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
