# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import numpy as np
import torch

from app.config import Settings
from tirex2 import ForecastModel, TimeseriesType, load_model


class Tirex2Model:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        self.model: ForecastModel = load_model(
            settings.model_path,
            device=settings.model_device,
        )

    def predict(self, contexts: list[TimeseriesType], prediction_length: int):
        forecasts = self.model.forecast(
            contexts, prediction_length=prediction_length, output_type="numpy", batch_size=min(len(contexts), 512)
        )

        return forecasts

    def warmup(self) -> None:
        print("Compiling the model...")
        context = TimeseriesType(
            target=torch.arange(2048, dtype=torch.float32).reshape(1, -1),
            past_covariates=None,
            future_covariates=None,
        )
        self.predict([context], prediction_length=32)
        print("Compilation done.")
