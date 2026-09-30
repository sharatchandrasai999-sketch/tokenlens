"""Tests for CLI input handling."""
import argparse

import pytest

from tokenlens import __version__
from tokenlens.cli import _text_from, main


def _args(text="", file=None):
    return argparse.Namespace(text=text, file=file)


def test_text_from_warns_on_empty_input(capsys):
    assert _text_from(_args(text="")) == ""
    assert "warning: empty input" in capsys.readouterr().err


def test_text_from_warns_on_whitespace_only(capsys, tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n")
    assert _text_from(_args(file=str(f))) == "   \n"
    assert "warning: empty input" in capsys.readouterr().err


def test_text_from_quiet_on_real_text(capsys):
    assert _text_from(_args(text="hello")) == "hello"
    assert capsys.readouterr().err == ""


def test_text_from_reads_file(capsys, tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("Summarize this.")
    assert _text_from(_args(file=str(f))) == "Summarize this."
    assert capsys.readouterr().err == ""


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "tokenlens" in out
    assert __version__ in out
