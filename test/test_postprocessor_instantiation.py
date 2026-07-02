"""Instantiation tests for :class:`PostProcessor` and :class:`PostProcessorConfig`."""

import pytest
import torch

from tirex2.model.component.postprocessor import PostProcessor, PostProcessorConfig

# A set of field values that satisfies every validator in ``PostProcessorConfig``.
VALID_CONFIG = {
    "r2": 0.9,
    "trend_min_r2": 0.9,
    "trend_threshold": 3.0,
    "sigma": 3.0,
    "trend_window": 1.0,
    "diff_band_exponent": 0.7,
    "diff_band_scale": 1.0,
    "raw_band_exponent": 0.3,
    "raw_band_scale": 0.5,
}


def test_instantiates_with_default_config():
    pp = PostProcessor()

    assert isinstance(pp.cfg, PostProcessorConfig)
    assert pp.cfg.trend_min_r2 == 0.87
    assert pp.cfg.diff_band_scale == 1.68


def test_instantiates_from_dict():
    pp = PostProcessor(dict(VALID_CONFIG))

    assert isinstance(pp.cfg, PostProcessorConfig)
    assert pp.cfg.trend_window == VALID_CONFIG["trend_window"]


def test_tta_diff_false_disables_differencing_masks():
    pp = PostProcessor({**VALID_CONFIG, "trend_threshold": 0.1, "trend_min_r2": 0.1})
    target = [torch.arange(16, dtype=torch.float32).unsqueeze(0)]
    covariates = [None]

    _, _, meta = pp.transform_input(
        target,
        prediction_length=4,
        past_covariates=covariates,
        past_future_covariates=covariates,
        tta_diff=False,
    )

    assert meta["diff_masks"] == [[False]]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trend_window", 0.0),  # must be in (0, 1]
        ("trend_window", 1.5),  # must be in (0, 1]
        ("diff_band_exponent", -0.1),  # must be >= 0
        ("diff_band_scale", 0.0),  # must be > 0
        ("raw_band_exponent", -1.0),  # must be >= 0
        ("raw_band_scale", 0.0),  # must be > 0
        ("trend_min_r2", 0.0),  # must be in (0, 1]
        ("trend_min_r2", 1.5),  # must be in (0, 1]
        ("trend_threshold", 0.0),  # must be > 0
    ],
)
def test_invalid_values_raise(field, value):
    config = dict(VALID_CONFIG)
    config[field] = value

    with pytest.raises(ValueError, match=field):
        PostProcessor(config)
