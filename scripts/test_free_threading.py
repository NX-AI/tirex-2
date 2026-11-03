"""Run core tests and verify that the GIL remains disabled."""

import sys

import pytest

if __name__ == "__main__":
    assert not sys._is_gil_enabled(), "GIL is enabled"
    result = pytest.main(["test"])
    assert not sys._is_gil_enabled(), "A test dependency enabled the GIL"
    raise SystemExit(result)
