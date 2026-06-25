"""Smoke test that the experiment study package imports."""

import experiments


def test_study_package_imports() -> None:
    assert experiments.__doc__ is not None
