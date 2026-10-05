"""Safetensors checkpoint conversion and loading."""

import datetime
import json
import sys
import warnings
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

from tirex2._state_dict import expand_shared_duplicates, shared_tensor_groups
from tirex2.base import CKPT_FILENAME, CONFIG_FILENAME, LEGACY_CONFIG_FILENAME, WEIGHTS_FILENAME, load_model
from tirex2.model.tirex2 import TiRex2
from tirex2.model.types import TimeseriesType

# ``scripts`` is repo tooling rather than a package, so make the converter importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from convert_checkpoint import MANIFEST_FILENAME, convert, drop_shared_duplicates, write_config  # noqa: E402

# Both block types are needed: ``bi-mlstm`` contributes the ``rev_cell``/``fwd_cell``
# module aliases and ``bi-slstm`` additionally the ``ParameterProxy`` parameter aliases.
RECIPE = ["small_mlstm", "small_slstm"]


@pytest.fixture
def aliased_model(build_small_model) -> TiRex2:
    torch.manual_seed(0)
    return build_small_model("cpu", recipe=RECIPE)


def _write_legacy_checkpoint(directory: Path, model: TiRex2, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / LEGACY_CONFIG_FILENAME).open("w") as f:
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


def test_safetensors_and_legacy_checkpoints_produce_identical_forecasts(tmp_path, aliased_model, small_model_kwargs):
    config = small_model_kwargs("cpu", RECIPE)
    legacy_dir = _write_legacy_checkpoint(tmp_path / "legacy", aliased_model, config)
    safetensors_dir = tmp_path / "safetensors"
    convert(str(legacy_dir), safetensors_dir, should_verify=False)

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
    write_config(small_model_kwargs("cpu", RECIPE), tmp_path / CONFIG_FILENAME)

    loaded = load_model(str(tmp_path), device="cpu")

    _assert_same_parameters(loaded.model, other_model)


def test_load_model_prefers_json_over_the_legacy_config(tmp_path, aliased_model, small_model_kwargs):
    # The two configs disagree on a setting, so the loaded model reveals which one was read.
    config = small_model_kwargs("cpu", RECIPE)
    _write_legacy_checkpoint(tmp_path, aliased_model, {**config, "tta_diff": False})
    write_config({**config, "tta_diff": True}, tmp_path / CONFIG_FILENAME)

    with pytest.warns(FutureWarning) as record:
        loaded = load_model(str(tmp_path), device="cpu")

    assert loaded.model.tta_diff is True
    # Only the pickle is legacy here; the YAML config is ignored without a warning.
    assert [CKPT_FILENAME in str(warning.message) for warning in record] == [True]


def test_load_model_does_not_warn_for_safetensors(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(tmp_path), None, should_verify=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        load_model(str(tmp_path), device="cpu")


def test_load_model_warns_once_per_legacy_file(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))

    with pytest.warns(FutureWarning) as record:
        load_model(str(tmp_path), device="cpu")

    messages = [str(warning.message) for warning in record]
    assert len(messages) == 2
    assert LEGACY_CONFIG_FILENAME in messages[0] and CKPT_FILENAME in messages[1]
    # Users are pointed at the Hub, not at repo tooling a pip install does not ship.
    assert all("deprecated" in message and "Hugging Face" in message for message in messages)
    assert not any("convert_checkpoint" in message for message in messages)
    # Attributed to the caller of load_model, not to tirex2 internals.
    assert {warning.filename for warning in record} == {__file__}


def test_load_model_reports_both_filenames_when_no_weights_are_present(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    (tmp_path / CKPT_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match=rf"{WEIGHTS_FILENAME}.*{CKPT_FILENAME}"):
        load_model(str(tmp_path), device="cpu")


def test_load_model_reports_both_filenames_when_no_config_is_present(tmp_path, aliased_model, small_model_kwargs):
    _write_legacy_checkpoint(tmp_path, aliased_model, small_model_kwargs("cpu", RECIPE))
    (tmp_path / LEGACY_CONFIG_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match=rf"{CONFIG_FILENAME}.*{LEGACY_CONFIG_FILENAME}"):
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


def test_load_model_falls_back_to_downloading_the_legacy_checkpoint(
    fake_hub, tmp_path, aliased_model, small_model_kwargs
):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))

    with pytest.warns(FutureWarning):
        loaded = load_model("hf://NX-AI/TiRex-2", device="cpu")

    assert requests == [[CONFIG_FILENAME, WEIGHTS_FILENAME], [LEGACY_CONFIG_FILENAME, CKPT_FILENAME]]
    _assert_same_parameters(loaded.model, aliased_model)


def test_load_model_downloads_only_the_legacy_files_that_have_no_replacement(
    fake_hub, tmp_path, aliased_model, small_model_kwargs
):
    # A repo that gained safetensors before it gained a JSON config.
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))
    convert(str(repo), None, should_verify=False)
    (repo / CONFIG_FILENAME).unlink()

    with pytest.warns(FutureWarning, match=LEGACY_CONFIG_FILENAME):
        loaded = load_model("NX-AI/TiRex-2", device="cpu")

    assert requests == [[CONFIG_FILENAME, WEIGHTS_FILENAME], [LEGACY_CONFIG_FILENAME]]
    _assert_same_parameters(loaded.model, aliased_model)


def test_convert_downloads_the_legacy_checkpoint_it_reads(fake_hub, tmp_path, aliased_model, small_model_kwargs):
    repo, requests = fake_hub
    _write_legacy_checkpoint(repo, aliased_model, small_model_kwargs("cpu", RECIPE))

    convert("NX-AI/TiRex-2", tmp_path / "out", should_verify=True)

    assert requests == [[LEGACY_CONFIG_FILENAME, CKPT_FILENAME]]
    assert (tmp_path / "out" / WEIGHTS_FILENAME).is_file()
    assert (tmp_path / "out" / CONFIG_FILENAME).is_file()
