import json
from pathlib import Path

import pytest
import torch

from tirex2 import TimeseriesType, load_model
from tirex2.model.component.flashrnn_slstm import _FlashRNNLayer
from tirex2.model.component.mlstm_block import mLSTMLayer

REFERENCES = Path(__file__).parent / "references"
CHECKPOINT = "NX-AI/TiRex-2"
RTOL, ATOL = 1e-5, 1e-5

REFERENCE_INPUT = json.loads((REFERENCES / "reference_input.json").read_text())
REFERENCE_OUTPUT = json.loads((REFERENCES / "reference_output_cpu.json").read_text())

H = REFERENCE_INPUT["H"]
TARGET = torch.tensor(REFERENCE_INPUT["target"], dtype=torch.float32)
FUTURE_COVARIATES = torch.tensor(REFERENCE_INPUT["future_covariates"], dtype=torch.float32)

SCENARIOS = {
    "univariate": TimeseriesType(target=TARGET[None], past_covariates=None, future_covariates=None),
    "multivariate": TimeseriesType(target=TARGET[None], past_covariates=None, future_covariates=FUTURE_COVARIATES),
}


@pytest.fixture(scope="module")
def compiled_model():
    try:
        return load_model(CHECKPOINT, device="cpu", compile=True)
    except Exception as exc:
        pytest.skip(f"reference checkpoint {CHECKPOINT} is unavailable: {exc}")


@pytest.mark.parametrize("layer_cls", [mLSTMLayer, _FlashRNNLayer])
def test_recurrent_layers_are_compiled(compiled_model, layer_cls):
    layers = [m for m in compiled_model.model.modules() if isinstance(m, layer_cls)]
    assert layers, f"model has no {layer_cls.__name__} to compile"
    uncompiled = [m for m in layers if m._compiled_call_impl is None]
    assert not uncompiled, f"{len(uncompiled)}/{len(layers)} {layer_cls.__name__} left uncompiled"


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_compiled_forecast_matches_cpu_reference(compiled_model, scenario):
    forecast = compiled_model.forecast([SCENARIOS[scenario]], prediction_length=H, output_type="torch")[0]
    expected = torch.tensor(REFERENCE_OUTPUT[scenario], dtype=torch.float32)

    assert forecast.shape == expected.shape
    torch.testing.assert_close(forecast, expected, rtol=RTOL, atol=ATOL)
