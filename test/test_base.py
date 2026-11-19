"""Checkpoint resolution and early device validation, without network or GPU access."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from tirex2 import base


@pytest.fixture
def download(monkeypatch, tmp_path):
    snapshot = Mock(return_value=str(tmp_path / "snapshot"))
    monkeypatch.setattr(base, "snapshot_download", snapshot)
    return snapshot


@pytest.mark.parametrize("as_path", [False, True], ids=["string", "path"])
@pytest.mark.parametrize("is_directory", [False, True], ids=["file", "directory"])
def test_existing_local_path_takes_precedence_over_repo_id(tmp_path, monkeypatch, download, as_path, is_directory):
    monkeypatch.chdir(tmp_path)
    path = Path("org/model")
    path.parent.mkdir()
    if is_directory:
        path.mkdir()
    else:
        path.touch()

    resolved = base._resolve_ckpt_dir(path if as_path else str(path))

    assert resolved == path
    download.assert_not_called()


@pytest.mark.parametrize("path", ["missing", "./missing/model", "../missing", "org/model/nested", "~/missing-model"])
def test_missing_local_path_is_not_downloaded(tmp_path, monkeypatch, download, path):
    monkeypatch.chdir(tmp_path)

    assert base._resolve_ckpt_dir(path) == Path(path).expanduser()
    download.assert_not_called()


def test_missing_absolute_path_is_not_downloaded(tmp_path, download):
    path = tmp_path / "missing-model"

    assert base._resolve_ckpt_dir(path) == path
    download.assert_not_called()


@pytest.mark.parametrize("path", ["org/model", "hf://org/model"])
@pytest.mark.parametrize("patterns", [None, ["config.json"], []], ids=["default", "custom", "empty"])
def test_hub_resolution_forwards_options(tmp_path, download, path, patterns):
    options = {"revision": "test-revision", "cache_dir": str(tmp_path / "cache"), "local_files_only": True}

    resolved = base._resolve_ckpt_dir(path, hf_kwargs=options, allow_patterns=patterns)

    assert resolved == Path(download.return_value)
    download.assert_called_once_with(
        repo_id="org/model",
        allow_patterns=["config.json", "model.safetensors"] if patterns is None else patterns,
        **options,
    )
    assert options == {
        "revision": "test-revision",
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
    }


@pytest.mark.parametrize("device, backend", [("cuda", "CUDA"), ("cuda:0", "CUDA"), ("mps", "MPS")])
def test_unavailable_device_fails_before_downloading(monkeypatch, download, device, backend):
    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(base.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=f"Execution on {backend} was requested but is not available"):
        base.load_model("hf://org/model", device=device)

    download.assert_not_called()
