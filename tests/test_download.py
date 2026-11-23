"""Tests for the interactive confirmation (no network involved)."""
from __future__ import annotations

import pytest

from fp8_quantize import download


class _FakeStdin:
    def __init__(self, isatty: bool) -> None:
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


def test_confirm_assume_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise AssertionError("input() must not be called with assume_yes")

    monkeypatch.setattr("builtins.input", boom)
    assert download.confirm("Do it?", assume_yes=True) is True


def test_confirm_yes_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_: "yes")
    monkeypatch.setattr(download.sys, "stdin", _FakeStdin(True))
    assert download.confirm("Do it?", assume_yes=False) is True


def test_confirm_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_: "maybe")
    monkeypatch.setattr(download.sys, "stdin", _FakeStdin(True))
    assert download.confirm("Do it?", assume_yes=False) is False


def test_confirm_requires_tty_without_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(download.sys, "stdin", _FakeStdin(False))
    with pytest.raises(SystemExit):
        download.confirm("Do it?", assume_yes=False)
    assert "needs an interactive prompt" in capsys.readouterr().err
