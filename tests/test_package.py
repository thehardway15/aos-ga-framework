"""Smoke tests for the aos_ga package and its command-line entry point."""

import pytest

import aos_ga
from aos_ga.cli import main


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(aos_ga.__version__, str)
    assert aos_ga.__version__


def test_cli_without_arguments_prints_help() -> None:
    assert main([]) == 0


def test_cli_version_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == aos_ga.__version__


def test_cli_rejects_unimplemented_command() -> None:
    with pytest.raises(SystemExit):
        main(["run"])
