from dataclasses import dataclass

import torch


@dataclass
class TimeseriesType:
    target: torch.Tensor  # [V_t, T]
    past_covariates: torch.Tensor | None  # [V_p, T]
    future_covariates: torch.Tensor | None  # [V_f, >=T+H]; extra future steps are ignored

    @property
    def n_past_covariates(self) -> int:
        return 0 if self.past_covariates is None else len(self.past_covariates)

    @property
    def n_future_covariates(self) -> int:
        return 0 if self.future_covariates is None else len(self.future_covariates)

    @property
    def past_length(self) -> int:
        return self.target.shape[-1]

    @property
    def future_length(self) -> int:
        return 0 if self.future_covariates is None else self.future_covariates.shape[-1] - self.past_length
