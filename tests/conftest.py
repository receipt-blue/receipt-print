import threading

import pytest


class FakePrinter:
    def __init__(self):
        self.writes = []
        self.closed = False
        self.cut_called = False

    def _raw(self, data):
        self.writes.append(data)

    def cut(self):
        self.cut_called = True

    def close(self):
        self.closed = True


@pytest.fixture
def fake_printer():
    return FakePrinter()


@pytest.fixture(autouse=True)
def _clean_speed_env(monkeypatch):
    monkeypatch.delenv("RP_SPEED_OVERRIDE", raising=False)
    monkeypatch.delenv("RP_SPEED", raising=False)
    monkeypatch.delenv("RP_HOST", raising=False)


@pytest.fixture
def gate():
    return threading.Event()
