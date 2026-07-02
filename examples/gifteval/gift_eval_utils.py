"""Helpers for running the GiftEval benchmark with a TiRex2 :class:`ForecastModel`.

This module wires the GiftEval datasets (loaded through
:mod:`data_fixed_new_numpy`) into GluonTS' :func:`evaluate_model`, using a thin
wrapper that exposes a TiRex2 ``ForecastModel`` through the GluonTS predictor
interface expected by the benchmark.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_fixed_new_numpy import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality


# avoid excessive logging
class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))

SHORT_DATA = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
MED_LONG_DATA = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
ALL_DATASETS = list(set(SHORT_DATA.split() + MED_LONG_DATA.split()))

METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]

try:
    current_dir = Path(__file__).parent
    with open(current_dir / "dataset_properties.json") as f:
        dataset_properties_map = json.load(f)
except FileNotFoundError:
    raise ValueError("Can not find needed dataset_properties.json file!")


def gift_eval_dataset_iter():
    for ds_num, ds_name in enumerate(ALL_DATASETS):
        ds_key = ds_name.split("/")[0]
        terms = ["short", "medium", "long"]
        for term in terms:
            if (term == "medium" or term == "long") and ds_name not in MED_LONG_DATA.split():
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            yield {
                "ds_name": ds_name,
                "ds_key": ds_key,
                "ds_freq": ds_freq,
                "term": term,
            }


# Setup GiftEval evaluation
def evaluate_dataset(predictor, ds_name, ds_key, ds_freq, term, eval_multivariate=True):
    print(f"Processing dataset: {ds_name}")
    ds_config = f"{ds_key}/{ds_freq}/{term}"
    # Initialize the dataset. In multivariate mode the native ``[V, T]`` target is
    # kept and every variate is fed to the model jointly (the variate-mixing path is
    # exercised). Otherwise multivariate datasets are split into independent univariate
    # channels via ``MultivariateToUnivariate`` -- the standard GIFT-Eval protocol.
    base_dataset = Dataset(name=ds_name, term=term, to_univariate=False)
    to_univariate = base_dataset.target_dim != 1 and not eval_multivariate
    dataset = Dataset(name=ds_name, term=term, to_univariate=True) if to_univariate else base_dataset
    predictor.set_prediction_len(dataset.prediction_length)
    predictor.set_ds_freq(ds_freq)
    season_length = get_seasonality(dataset.freq)

    # Measure the time taken for evaluation
    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=METRICS,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    result = {
        "dataset": ds_config,
        "model": predictor.model_id,
        "eval_metrics/MSE[mean]": res["MSE[mean]"].iloc[0],
        "eval_metrics/MSE[0.5]": res["MSE[0.5]"].iloc[0],
        "eval_metrics/MAE[0.5]": res["MAE[0.5]"].iloc[0],
        "eval_metrics/MASE[0.5]": res["MASE[0.5]"].iloc[0],
        "eval_metrics/MAPE[0.5]": res["MAPE[0.5]"].iloc[0],
        "eval_metrics/sMAPE[0.5]": res["sMAPE[0.5]"].iloc[0],
        "eval_metrics/MSIS": res["MSIS"].iloc[0],
        "eval_metrics/RMSE[mean]": res["RMSE[mean]"].iloc[0],
        "eval_metrics/NRMSE[mean]": res["NRMSE[mean]"].iloc[0],
        "eval_metrics/ND[0.5]": res["ND[0.5]"].iloc[0],
        "eval_metrics/mean_weighted_sum_quantile_loss": res["mean_weighted_sum_quantile_loss"].iloc[0],
        "domain": dataset_properties_map[ds_key]["domain"],
        "num_variates": dataset_properties_map[ds_key]["num_variates"],
    }
    return result


@dataclass
class TiRexGiftEvalWrapper:
    """GluonTS predictor adapter around a TiRex2 :class:`ForecastModel`.

    Parameters
    ----------
    model : ForecastModel
        A loaded TiRex2 forecast model exposing ``forecast_gluon``.
    freq : str or None, optional
        Frequency string of the dataset currently being evaluated. Set via
        :meth:`set_ds_freq` before each dataset.
    pred_len : int, optional
        Forecast horizon for the current dataset. Set via
        :meth:`set_prediction_len` before each dataset.
    """

    model: Any
    freq: str | None = None
    pred_len: int = 32
    eval_multivariate: bool = True

    def set_ds_freq(self, freq):
        self.freq = freq

    def set_prediction_len(self, pred_len):
        self.pred_len = pred_len

    def predict(self, test_data_input):
        """Forecast each GIFT-Eval entry through :meth:`ForecastModel.forecast_gluon`.

        With ``eval_multivariate=True`` the native ``[V, T]`` target is kept intact and all
        variates are forecast jointly, exercising the backbone's variate-mixing path; the
        resulting forecasts retain the variate axis (``[Q, H, V]``) so GluonTS scores them
        against the multivariate ``[V, H]`` label. With ``eval_multivariate=False`` each variate
        is rendered as an independent univariate ``QuantileForecast`` -- the standard GIFT-Eval
        protocol. In both cases the model's configured ``tta_sign_flip`` / ``tta_diff`` defaults
        (from ``model-config.yaml``) apply.
        """
        return self.model.forecast_gluon(
            test_data_input,
            prediction_length=self.pred_len,
            output_type="gluonts",
            multivariate=self.eval_multivariate,
        )

    @property
    def model_id(self):
        return "TiRex"
