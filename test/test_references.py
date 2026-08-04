import json
import sys
from pathlib import Path

import pytest
import torch

from tirex2 import TimeseriesType, load_model
from tirex2.model.component.attention_block import is_flex_attention_available

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="reference outputs were recorded on Linux")

REFERENCES = Path(__file__).parent / "references"
CHECKPOINT = "NX-AI/TiRex-2"
RTOL, ATOL = 1e-5, 1e-5


def _reference(name: str) -> dict:
    return json.loads((REFERENCES / f"{name}.json").read_text())


REFERENCE_INPUT = _reference("reference_input")
REFERENCE_OUTPUT = _reference("reference_output_cpu")

H = REFERENCE_INPUT["H"]
TARGET = torch.tensor(REFERENCE_INPUT["target"], dtype=torch.float32)
OTHER_TARGET = torch.tensor(REFERENCE_INPUT["other_target"], dtype=torch.float32)
FUTURE_COVARIATES = torch.tensor(REFERENCE_INPUT["future_covariates"], dtype=torch.float32)

SCENARIOS = {
    "univariate": TimeseriesType(target=TARGET[None], past_covariates=None, future_covariates=None),
    "multivariate": TimeseriesType(target=TARGET[None], past_covariates=None, future_covariates=FUTURE_COVARIATES),
    "multitarget": TimeseriesType(
        target=torch.stack([TARGET, OTHER_TARGET]), past_covariates=None, future_covariates=None
    ),
}


def _load_reference_model(**kwargs):
    try:
        return load_model(CHECKPOINT, device="cpu", **kwargs)
    except Exception as exc:
        pytest.skip(f"reference checkpoint {CHECKPOINT} is unavailable: {exc}")


@pytest.fixture(scope="module")
def model():
    return _load_reference_model()


@pytest.fixture(scope="module")
def flex_model():
    if not is_flex_attention_available():
        pytest.skip("FlexAttention is not available in this PyTorch installation")

    model = _load_reference_model(use_flex_attention=True)
    assert all(block.variate_mixer.attn.use_flex_attention for block in model.model.stack)
    return model


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_forecast_matches_cpu_reference(model, scenario):
    forecast = model.forecast([SCENARIOS[scenario]], prediction_length=H, output_type="torch")[0]
    expected = torch.tensor(REFERENCE_OUTPUT[scenario], dtype=torch.float32)

    assert forecast.shape == expected.shape
    torch.testing.assert_close(forecast, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_flex_attention_forecast_matches_cpu_reference(flex_model, scenario):
    forecast = flex_model.forecast([SCENARIOS[scenario]], prediction_length=H, output_type="torch")[0]
    expected = torch.tensor(REFERENCE_OUTPUT[scenario], dtype=torch.float32)

    assert forecast.shape == expected.shape
    torch.testing.assert_close(forecast, expected, rtol=FLEX_RTOL, atol=FLEX_ATOL)
