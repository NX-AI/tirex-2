import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors

from tirex2 import TimeseriesType, load_model
from tirex2.base import CONFIG_FILENAME, WEIGHTS_FILENAME

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from convert_checkpoint import write_config  # noqa: E402

RECIPE = ["small_mlstm", "small_slstm"]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"))]


@pytest.mark.parametrize("device", DEVICES)
def test_compiled_forecast_matches_eager(tmp_path, build_small_model, small_model_kwargs, device):
    torch.manual_seed(0)
    model = build_small_model("cpu", recipe=RECIPE)
    save_safetensors(model.state_dict(), str(tmp_path / WEIGHTS_FILENAME))
    write_config(small_model_kwargs("cpu", RECIPE), tmp_path / CONFIG_FILENAME)

    eager = load_model(tmp_path, device=device, compile=False)
    compiled = load_model(tmp_path, device=device, compile=True)
    assert not hasattr(eager.model._stack, "_torchdynamo_orig_callable")
    assert hasattr(compiled.model._stack, "_torchdynamo_orig_callable")

    torch.manual_seed(1)
    ts = [TimeseriesType(target=torch.randn(2, 64).cumsum(-1), past_covariates=None, future_covariates=None)]
    expected = eager.forecast(ts, prediction_length=8, output_type="torch")[0]
    actual = compiled.forecast(ts, prediction_length=8, output_type="torch")[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
