"""Safetensors checkpoint loading and legacy-format rejection."""

import json
import warnings
from pathlib import Path

import pytest
import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import load_file as load_safetensors, save_model

from tirex2.base import (
    _LEGACY_CONVERTER_URL,
    CKPT_FILENAME,
    CONFIG_FILENAME,
    LEGACY_CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    _resolve_ckpt_dir,
    load_model,
)
from tirex2.model.tirex2 import TiRex2

# Both block types: historical checkpoints stored aliases for the reverse cell
# and the sLSTM recurrent kernel.
RECIPE = ["small_mlstm", "small_slstm"]


@pytest.fixture
def aliased_model(build_small_model) -> TiRex2:
    torch.manual_seed(0)
    return build_small_model("cpu", recipe=RECIPE)


def _legacy_state_dict(model: TiRex2) -> dict[str, torch.Tensor]:
    """``model``'s state dict with the alias names older checkpoints also store."""
    state_dict = model.state_dict()
    aliases = {name.replace(".fwd_cell.", ".rev_cell."): t for name, t in state_dict.items() if ".fwd_cell." in name}
    aliases |= {name.replace("._recurrent_kernel_", ".recurrent_kernel"): t for name, t in state_dict.items()}
    return state_dict | aliases


def _write_legacy_checkpoint(directory: Path, model: TiRex2, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML; these files are only used to test legacy rejection.
    (directory / LEGACY_CONFIG_FILENAME).write_text(json.dumps(config))
    torch.save(_legacy_state_dict(model), directory / CKPT_FILENAME)
    return directory


def _write_safetensors_checkpoint(directory: Path, model: TiRex2, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_FILENAME).write_text(json.dumps(config))
    save_model(model, str(directory / WEIGHTS_FILENAME))


def _assert_same_parameters(actual: TiRex2, expected: TiRex2) -> None:
    actual_state, expected_state = actual.state_dict(), expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, tensor in expected_state.items():
        torch.testing.assert_close(actual_state[name], tensor, rtol=0, atol=0, msg=f"{name} differs")


def test_load_model_loads_safetensors(tmp_path, aliased_model, small_model_kwargs):
    _write_safetensors_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))

    loaded = load_model(str(tmp_path), device="cpu")

    _assert_same_parameters(loaded.model, aliased_model)


def test_load_model_prefers_safetensors_over_the_legacy_checkpoint(
    tmp_path, aliased_model, build_small_model, small_model_kwargs
):
    # The two files hold different weights, so the loaded model reveals which one was read.
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    torch.manual_seed(7)
    other_model = build_small_model("cpu", recipe=RECIPE)
    _write_safetensors_checkpoint(tmp_path, other_model, small_model_kwargs("cpu", RECIPE))

    loaded = load_model(str(tmp_path), device="cpu")

    _assert_same_parameters(loaded.model, other_model)


def test_load_model_does_not_warn_for_safetensors(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    _write_safetensors_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        load_model(str(tmp_path), device="cpu")


def test_load_model_rejects_legacy_checkpoints_with_the_last_release_that_loads_them(
    tmp_path, aliased_model, small_model_kwargs
):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))

    with pytest.raises(FileNotFoundError, match=r"tirex-2<=0\.4\.0") as error:
        load_model(str(tmp_path), device="cpu")
    assert _LEGACY_CONVERTER_URL in str(error.value)


def test_load_model_reports_the_missing_weights(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    _write_safetensors_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    (tmp_path / WEIGHTS_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match=WEIGHTS_FILENAME):
        load_model(str(tmp_path), device="cpu")


def test_load_model_reports_the_missing_config_without_a_legacy_hint(tmp_path, aliased_model):
    save_model(aliased_model, str(tmp_path / WEIGHTS_FILENAME))

    with pytest.raises(FileNotFoundError, match=CONFIG_FILENAME) as error:
        load_model(str(tmp_path), device="cpu")
    assert "legacy" not in str(error.value)


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Stand in for ``snapshot_download``, serving ``repo`` and recording each request.

    The real function copies exactly the files matching ``allow_patterns`` into the
    cache, so the stub does the same: what a caller asks for is what it gets.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    requests: list[list[str]] = []

    def fake_snapshot_download(repo_id, allow_patterns, **kwargs):
        requests.append(list(allow_patterns))
        for name in allow_patterns:
            source = repo / name
            if source.is_file():
                (cache / name).write_bytes(source.read_bytes())
        return str(cache)

    monkeypatch.setattr("tirex2.base.snapshot_download", fake_snapshot_download)
    return repo, requests


def test_load_model_downloads_only_the_current_format_when_the_repo_has_both(
    fake_hub, tmp_path, aliased_model, small_model_kwargs
):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))
    _write_safetensors_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))

    loaded = load_model("NX-AI/TiRex-2", device="cpu")

    # The legacy files stay on the hub for older clients, but fetching them as well
    # would transfer the weights twice and read one copy.
    assert requests == [[CONFIG_FILENAME, WEIGHTS_FILENAME]]
    _assert_same_parameters(loaded.model, aliased_model)


def test_load_model_does_not_download_the_legacy_checkpoint(fake_hub, tmp_path, aliased_model, small_model_kwargs):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))

    with pytest.raises(FileNotFoundError):
        load_model("hf://NX-AI/TiRex-2", device="cpu")

    assert requests == [[CONFIG_FILENAME, WEIGHTS_FILENAME]]


@pytest.fixture
def offline_hub_cache(tmp_path, aliased_model, small_model_kwargs):
    # Older supported Hub clients do not have the completeness check that caused
    # this regression. Exercise the real client when that check is available.
    tree_cache = pytest.importorskip("huggingface_hub._tree_cache")
    repo = tmp_path / "models--review--model"
    commit = "a" * 40
    snapshot = repo / "snapshots" / commit
    _write_legacy_checkpoint(snapshot, aliased_model, small_model_kwargs("cpu", RECIPE))
    _write_safetensors_checkpoint(snapshot, aliased_model, small_model_kwargs("cpu", RECIPE))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)
    tree_cache.write_tree_cache(
        str(repo),
        commit,
        {
            name: tree_cache.TreeCacheEntry(size=(snapshot / name).stat().st_size, blob_id="b" * 40)
            for name in [CONFIG_FILENAME, WEIGHTS_FILENAME, LEGACY_CONFIG_FILENAME, CKPT_FILENAME]
        },
    )
    return snapshot, {"cache_dir": str(tmp_path), "local_files_only": True}


@pytest.mark.parametrize(
    "uncached",
    [[CONFIG_FILENAME, WEIGHTS_FILENAME], [CONFIG_FILENAME], [WEIGHTS_FILENAME]],
    ids=["legacy-only", "legacy-config", "legacy-weights"],
)
def test_load_model_rejects_legacy_files_in_incomplete_offline_snapshot(offline_hub_cache, aliased_model, uncached):
    snapshot, hf_kwargs = offline_hub_cache
    for name in uncached:
        (snapshot / name).unlink()

    with pytest.raises((LocalEntryNotFoundError, FileNotFoundError)):
        load_model("review/model", device="cpu", hf_kwargs=hf_kwargs)


def test_offline_snapshot_without_either_weights_format_still_fails(offline_hub_cache):
    snapshot, hf_kwargs = offline_hub_cache
    (snapshot / WEIGHTS_FILENAME).unlink()
    (snapshot / CKPT_FILENAME).unlink()

    with pytest.raises(LocalEntryNotFoundError):
        _resolve_ckpt_dir("review/model", hf_kwargs=hf_kwargs)


def test_hub_cache_miss_without_snapshot_path_is_preserved(monkeypatch):
    error = LocalEntryNotFoundError("No cached snapshot")

    def fail_download(**kwargs):
        raise error

    monkeypatch.setattr("tirex2.base.snapshot_download", fail_download)
    with pytest.raises(LocalEntryNotFoundError) as raised:
        _resolve_ckpt_dir("review/model", hf_kwargs={"local_files_only": True})
    assert raised.value is error


def test_legacy_state_dict_with_alias_names_loads(aliased_model, build_small_model):
    legacy = _legacy_state_dict(aliased_model)
    assert any(".rev_cell." in name for name in legacy) and any(name.endswith(".recurrent_kernel") for name in legacy)

    model = build_small_model("cpu", recipe=RECIPE)
    model.load_state_dict(legacy, strict=True)
    _assert_same_parameters(model, aliased_model)


def test_safetensors_stores_each_parameter_once(tmp_path, aliased_model):
    weights_file = tmp_path / WEIGHTS_FILENAME
    state_dict = aliased_model.state_dict()
    save_model(aliased_model, str(weights_file))
    written = load_safetensors(weights_file)
    assert written.keys() == state_dict.keys()
    for name, tensor in written.items():
        torch.testing.assert_close(tensor, state_dict[name], rtol=0, atol=0)
