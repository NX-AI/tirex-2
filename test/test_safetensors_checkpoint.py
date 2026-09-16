"""Safetensors checkpoint conversion and loading."""

import json
import sys
import warnings
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

from tirex2._state_dict import drop_shared_duplicates, expand_shared_duplicates, shared_tensor_groups
from tirex2.base import CKPT_FILENAME, CONFIG_FILENAME, WEIGHTS_FILENAME, load_model
from tirex2.model.tirex2 import TiRex2
from tirex2.model.types import TimeseriesType

# ``scripts`` is repo tooling rather than a package, so make the converter importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from convert_checkpoint import MANIFEST_FILENAME, convert  # noqa: E402

# Both block types are needed: ``bi-mlstm`` contributes the ``rev_cell``/``fwd_cell``
# module aliases and ``bi-slstm`` additionally the ``ParameterProxy`` parameter aliases.
RECIPE = ["small_mlstm", "small_slstm"]


@pytest.fixture
def aliased_model(build_small_model) -> TiRex2:
    torch.manual_seed(0)
    return build_small_model("cpu", recipe=RECIPE)


def _write_legacy_checkpoint(directory: Path, model: TiRex2, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CONFIG_FILENAME).open("w") as f:
        yaml.safe_dump(config, f)
    torch.save(model.state_dict(), directory / CKPT_FILENAME)
    return directory


def _assert_same_parameters(actual: TiRex2, expected: TiRex2) -> None:
    actual_state, expected_state = actual.state_dict(), expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, tensor in expected_state.items():
        torch.testing.assert_close(actual_state[name], tensor, rtol=0, atol=0, msg=f"{name} differs")


def test_shared_tensor_groups_are_derived_from_tensor_identity(aliased_model):
    groups = shared_tensor_groups(aliased_model)
    state_dict = aliased_model.state_dict()

    assert groups, "the test model is expected to share tensors across state_dict names"
    for group in groups:
        assert len(group) > 1
        assert all(name in state_dict for name in group)
        # Every member is literally the same tensor object, which is why safetensors
        # refuses to store them separately.
        tensors = [state_dict[name] for name in group]
        assert all(tensor.data_ptr() == tensors[0].data_ptr() for tensor in tensors)

    # A name may only belong to one group.
    names = [name for group in groups for name in group]
    assert len(names) == len(set(names))


def test_drop_then_expand_round_trips_the_state_dict(aliased_model):
    state_dict = aliased_model.state_dict()

    kept, dropped = drop_shared_duplicates(aliased_model, state_dict)

    assert dropped, "the test model is expected to have redundant alias names"
    assert set(kept) | set(dropped) == set(state_dict)
    assert not set(kept) & set(dropped)

    restored = expand_shared_duplicates(aliased_model, kept)

    assert restored.keys() == state_dict.keys()
    for name, tensor in state_dict.items():
        assert torch.equal(restored[name], tensor)


def test_drop_shared_duplicates_rejects_alias_names_that_disagree(aliased_model):
    state_dict = dict(aliased_model.state_dict())
    canonical, duplicate = shared_tensor_groups(aliased_model)[0][:2]
    state_dict[duplicate] = state_dict[canonical] + 1.0

    with pytest.raises(ValueError, match="deduplication would lose information"):
        drop_shared_duplicates(aliased_model, state_dict)


def test_expand_shared_duplicates_rejects_alias_names_that_disagree(aliased_model):
    state_dict = dict(aliased_model.state_dict())
    canonical, duplicate = shared_tensor_groups(aliased_model)[0][:2]
    state_dict[duplicate] = state_dict[canonical] + 1.0

    with pytest.raises(ValueError, match="alias the same tensor"):
        expand_shared_duplicates(aliased_model, state_dict)


def test_expand_shared_duplicates_keeps_wholly_absent_groups_missing(aliased_model):
    group = shared_tensor_groups(aliased_model)[0]
    state_dict = {name: tensor for name, tensor in aliased_model.state_dict().items() if name not in group}

    expanded = expand_shared_duplicates(aliased_model, state_dict)

    # Nothing is invented, so a strict load still reports the genuine gap.
    assert not set(group) & set(expanded)
    with pytest.raises(RuntimeError, match="Missing key"):
        aliased_model.load_state_dict(expanded, strict=True)


def test_convert_writes_one_tensor_per_alias_group(tmp_path, aliased_model, small_model_kwargs):
    config = small_model_kwargs("cpu", RECIPE)
    _write_legacy_checkpoint(tmp_path, aliased_model, config)
    state_dict = aliased_model.state_dict()
    num_aliases = sum(len(group) - 1 for group in shared_tensor_groups(aliased_model))

    weights_file = convert(str(tmp_path), None, should_verify=True)

    assert weights_file == tmp_path / WEIGHTS_FILENAME
    written = load_safetensors(weights_file, device="cpu")
    assert len(written) == len(state_dict) - num_aliases
    assert set(written) < set(state_dict)
    for name, tensor in written.items():
        assert torch.equal(tensor, state_dict[name])

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest["num_tensors_in_checkpoint"] == len(state_dict)
    assert manifest["num_tensors_written"] == len(written)
    assert len(manifest["dropped_keys"]) == num_aliases


def test_load_model_loads_converted_safetensors(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(tmp_path), None, should_verify=False)
    (tmp_path / CKPT_FILENAME).unlink()

    loaded = load_model(str(tmp_path), device="cpu")

    _assert_same_parameters(loaded.model, aliased_model)


def test_safetensors_and_legacy_checkpoints_produce_identical_forecasts(tmp_path, aliased_model, small_model_kwargs):
    config = small_model_kwargs("cpu", RECIPE)
    legacy_dir = _write_legacy_checkpoint(tmp_path / "legacy", aliased_model, config)
    safetensors_dir = tmp_path / "safetensors"
    convert(str(legacy_dir), safetensors_dir, should_verify=False)
    (safetensors_dir / CONFIG_FILENAME).write_text(yaml.safe_dump(config))

    torch.manual_seed(1)
    series = [TimeseriesType(target=torch.randn(1, 16), past_covariates=None, future_covariates=None)]

    with pytest.warns(FutureWarning):
        legacy_forecast = load_model(str(legacy_dir), device="cpu").model.predict(series, prediction_length=4)[0]
    safetensors_forecast = load_model(str(safetensors_dir), device="cpu").model.predict(series, prediction_length=4)[0]

    torch.testing.assert_close(safetensors_forecast, legacy_forecast, rtol=0, atol=0)


def test_load_model_prefers_safetensors_over_the_legacy_checkpoint(
    tmp_path, aliased_model, build_small_model, small_model_kwargs
):
    # The two files hold different weights, so the loaded model reveals which one was read.
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    torch.manual_seed(7)
    other_model = build_small_model("cpu", recipe=RECIPE)
    kept, _ = drop_shared_duplicates(other_model, other_model.state_dict())
    save_safetensors(kept, str(tmp_path / WEIGHTS_FILENAME))

    loaded = load_model(str(tmp_path), device="cpu")

    _assert_same_parameters(loaded.model, other_model)


def test_load_model_does_not_warn_for_safetensors(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(tmp_path), None, should_verify=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        load_model(str(tmp_path), device="cpu")


def test_load_model_warns_when_falling_back_to_the_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))

    with pytest.warns(FutureWarning, match="deprecated"):
        load_model(str(tmp_path), device="cpu")


def test_load_model_reports_both_filenames_when_no_weights_are_present(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    (tmp_path / CKPT_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match=rf"{WEIGHTS_FILENAME}.*{CKPT_FILENAME}"):
        load_model(str(tmp_path), device="cpu")


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


def test_load_model_downloads_only_safetensors_when_the_repo_has_both(
    fake_hub, tmp_path, aliased_model, small_model_kwargs
):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(repo), None, should_verify=False)

    loaded = load_model("NX-AI/TiRex-2", device="cpu")

    # The legacy pickle stays on the hub for older clients, but fetching it as well
    # would transfer the weights twice and read one copy.
    assert requests == [[CONFIG_FILENAME, WEIGHTS_FILENAME]]
    _assert_same_parameters(loaded.model, aliased_model)


def test_load_model_falls_back_to_downloading_the_legacy_checkpoint(
    fake_hub, tmp_path, aliased_model, small_model_kwargs
):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))

    with pytest.warns(FutureWarning):
        loaded = load_model("hf://NX-AI/TiRex-2", device="cpu")

    assert requests == [[CONFIG_FILENAME, WEIGHTS_FILENAME], [CONFIG_FILENAME, CKPT_FILENAME]]
    _assert_same_parameters(loaded.model, aliased_model)


def test_convert_downloads_the_legacy_checkpoint_it_reads(fake_hub, tmp_path, aliased_model, small_model_kwargs):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))

    convert("NX-AI/TiRex-2", tmp_path / "out", should_verify=True)

    assert requests == [[CONFIG_FILENAME, CKPT_FILENAME]]
    assert (tmp_path / "out" / WEIGHTS_FILENAME).is_file()
