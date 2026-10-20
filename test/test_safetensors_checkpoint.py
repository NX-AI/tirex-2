"""Safetensors checkpoint conversion and loading."""

import datetime
import json
import sys
import warnings
from pathlib import Path

import pytest
import torch
import yaml
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

from tirex2.base import (
    CKPT_FILENAME,
    CONFIG_FILENAME,
    LEGACY_CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    _resolve_ckpt_dir,
    load_model,
)
from tirex2.model.tirex2 import TiRex2
from tirex2.model.types import TimeseriesType

# ``scripts`` is repo tooling rather than a package, so make the converter importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from convert_checkpoint import MANIFEST_FILENAME, convert, write_config  # noqa: E402

# Both block types: legacy checkpoints alias the shared bidirectional cell under ``rev_cell``
# and the sLSTM recurrent kernel under ``recurrent_kernel``.
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
    with (directory / LEGACY_CONFIG_FILENAME).open("w") as f:
        yaml.safe_dump(config, f)
    torch.save(_legacy_state_dict(model), directory / CKPT_FILENAME)
    return directory


def _assert_same_parameters(actual: TiRex2, expected: TiRex2) -> None:
    actual_state, expected_state = actual.state_dict(), expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, tensor in expected_state.items():
        torch.testing.assert_close(actual_state[name], tensor, rtol=0, atol=0, msg=f"{name} differs")


def test_convert_writes_the_model_state_dict_without_aliases(tmp_path, aliased_model, small_model_kwargs):
    config = small_model_kwargs("cpu", RECIPE)
    _write_legacy_checkpoint(tmp_path, aliased_model, config)
    state_dict = aliased_model.state_dict()

    weights_file = convert(str(tmp_path), None, should_verify=True)

    assert weights_file == tmp_path / WEIGHTS_FILENAME
    written = load_safetensors(weights_file, device="cpu")
    assert written.keys() == state_dict.keys()
    for name, tensor in written.items():
        assert torch.equal(tensor, state_dict[name])

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest["num_tensors_in_checkpoint"] == len(_legacy_state_dict(aliased_model))
    assert manifest["num_tensors_written"] == len(written)

    assert json.loads((tmp_path / CONFIG_FILENAME).read_text()) == config


def test_convert_config_only_needs_no_legacy_weights(tmp_path, aliased_model, small_model_kwargs):
    config = small_model_kwargs("cpu", RECIPE)
    _write_legacy_checkpoint(tmp_path, aliased_model, config)
    (tmp_path / CKPT_FILENAME).unlink()

    config_file = convert(str(tmp_path), None, should_verify=True, config_only=True)

    assert config_file == tmp_path / CONFIG_FILENAME
    assert json.loads(config_file.read_text()) == config
    assert not (tmp_path / WEIGHTS_FILENAME).exists()


@pytest.mark.parametrize(
    "config, match",
    [
        ({"released": datetime.date(2026, 1, 1)}, "cannot be represented as JSON"),
        ({"eps": float("nan")}, "cannot be represented as JSON"),
        ({"templates": {1: "mlstm"}}, "changes when written as JSON"),
    ],
    ids=["date", "nan", "int-key"],
)
def test_write_config_rejects_what_json_cannot_store_faithfully(tmp_path, config, match):
    config_file = tmp_path / CONFIG_FILENAME

    with pytest.raises(ValueError, match=match):
        write_config(config, config_file)

    assert not config_file.exists()


def test_load_model_loads_converted_safetensors(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(tmp_path), None, should_verify=False)
    (tmp_path / CKPT_FILENAME).unlink()
    (tmp_path / LEGACY_CONFIG_FILENAME).unlink()

    loaded = load_model(str(tmp_path), device="cpu")

    _assert_same_parameters(loaded.model, aliased_model)


def test_load_model_prefers_safetensors_over_the_legacy_checkpoint(
    tmp_path, aliased_model, build_small_model, small_model_kwargs
):
    # The two files hold different weights, so the loaded model reveals which one was read.
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    torch.manual_seed(7)
    other_model = build_small_model("cpu", recipe=RECIPE)
    save_safetensors(other_model.state_dict(), str(tmp_path / WEIGHTS_FILENAME))
    write_config(small_model_kwargs("cpu", RECIPE), tmp_path / CONFIG_FILENAME)

    loaded = load_model(str(tmp_path), device="cpu")

    _assert_same_parameters(loaded.model, other_model)


def test_load_model_does_not_warn_for_safetensors(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(tmp_path), None, should_verify=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        load_model(str(tmp_path), device="cpu")


def test_load_model_rejects_legacy_checkpoints_with_the_last_release_that_loads_them(
    tmp_path, aliased_model, small_model_kwargs
):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))

    with pytest.raises(FileNotFoundError, match=r"tirex-2<=0\.3\.1.*convert_checkpoint\.py"):
        load_model(str(tmp_path), device="cpu")


def test_load_model_reports_the_missing_weights(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(tmp_path), None, should_verify=False)
    (tmp_path / WEIGHTS_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match=WEIGHTS_FILENAME):
        load_model(str(tmp_path), device="cpu")


def test_load_model_reports_the_missing_config_without_a_legacy_hint(tmp_path, aliased_model):
    save_safetensors(aliased_model.state_dict(), str(tmp_path / WEIGHTS_FILENAME))

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
    convert(str(repo), None, should_verify=False)

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


def test_convert_downloads_the_legacy_checkpoint_it_reads(fake_hub, tmp_path, aliased_model, small_model_kwargs):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))

    convert("NX-AI/TiRex-2", tmp_path / "out", should_verify=True)

    assert requests == [[LEGACY_CONFIG_FILENAME, CKPT_FILENAME]]
    assert (tmp_path / "out" / WEIGHTS_FILENAME).is_file()
    assert (tmp_path / "out" / CONFIG_FILENAME).is_file()


@pytest.fixture
def offline_hub_cache(tmp_path, aliased_model, small_model_kwargs):
    # Older supported Hub clients do not have the completeness check that caused
    # this regression. Exercise the real client when that check is available.
    tree_cache = pytest.importorskip("huggingface_hub._tree_cache")
    repo = tmp_path / "models--review--model"
    commit = "a" * 40
    snapshot = repo / "snapshots" / commit
    _write_legacy_checkpoint(snapshot, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(snapshot), None, should_verify=False)
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
