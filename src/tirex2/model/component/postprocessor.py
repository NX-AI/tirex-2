"""Single-pass, native-multivariate postprocessor with per-target differencing."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import pad

logger = logging.getLogger(__name__)


@dataclass
class PostProcessorConfig:
    """Postprocessor defaults for trend detection and band calibration.

    These constants are the differencing/calibration values that used to be
    serialized in ``model/model-config.yaml``. Checkpoint configs now carry only
    ``tta_diff`` to enable or disable the differencing path at inference time.
    """

    r2: float = 0.87
    trend_min_r2: float = 0.87
    trend_threshold: float = 1.5
    sigma: float = 1.5
    trend_window: float = 1.0
    diff_band_exponent: float = 0.51
    diff_band_scale: float = 1.68
    raw_band_exponent: float = 0.52
    raw_band_scale: float = 0.90

    def __post_init__(self):
        if not 0.0 < self.trend_window <= 1.0:
            raise ValueError(f"trend_window must be a fraction in (0, 1], got {self.trend_window!r}.")
        if not self.diff_band_exponent >= 0.0:
            raise ValueError(f"diff_band_exponent must be >= 0, got {self.diff_band_exponent!r}.")
        if not self.diff_band_scale > 0.0:
            raise ValueError(f"diff_band_scale must be strictly positive, got {self.diff_band_scale!r}.")
        if not self.raw_band_exponent >= 0.0:
            raise ValueError(f"raw_band_exponent must be >= 0, got {self.raw_band_exponent!r}.")
        if not self.raw_band_scale > 0.0:
            raise ValueError(f"raw_band_scale must be strictly positive, got {self.raw_band_scale!r}.")
        if not (self.trend_min_r2 > 0.0 and self.trend_min_r2 <= 1.0):
            raise ValueError(f"trend_min_r2 ({self.trend_min_r2}) to be between 0 and 1.")
        if not self.trend_threshold > 0.0:
            raise ValueError(f"trend_threshold has to be larger than 0. Got {self.trend_threshold}.")


class PostProcessor:
    """Single-pass joint multivariate forecasting; trending targets are differenced."""

    def __init__(self, config: PostProcessorConfig | dict | None = None, *args, **kwargs) -> None:
        if config is None:
            self.cfg = PostProcessorConfig()
        elif isinstance(config, dict):
            self.cfg = PostProcessorConfig(**config)
        else:
            self.cfg = config

    def transform_input(
        self,
        target: list[torch.Tensor],
        prediction_length: int,
        past_covariates: list[torch.Tensor],
        past_future_covariates: list[torch.Tensor],
        *args,
        tta_diff: bool = True,
        **kwargs,
    ):
        """Pack each sample into one joint group and optionally difference trending targets."""
        # ``past_covariates`` / ``past_future_covariates`` are fed to the model
        # but always on their raw scale - they are never differenced.
        prepared_targets = []
        transform_params = []
        diff_masks = []

        for t in target:
            if t.ndim == 1:
                t = t.unsqueeze(0)

            params = self._diff_params(t)
            should_diff = self._detect_trends(t) if tta_diff else [False] * t.shape[0]
            diff_t = self._diff_forward(t)[0] if any(should_diff) else t

            prepared_targets.append({"raw": t, "diff": diff_t})
            transform_params.append(params)
            diff_masks.append(should_diff)

        max_target_len = max(item["raw"].shape[-1] for item in prepared_targets)

        x = []
        group_vector = []
        target_mask = []
        group_sample_map = {}
        group_target_indices = {}
        group_row_is_diff = {}

        for sample_idx, (target_item, diff_mask, pc, pfc) in enumerate(
            zip(prepared_targets, diff_masks, past_covariates, past_future_covariates)
        ):
            t_raw = target_item["raw"]
            t_diff = target_item["diff"]

            # One joint group per sample: every target variate (differenced rows
            # picked from ``t_diff``, the rest from ``t_raw``) plus its raw covariates.
            diff_sel = torch.tensor(diff_mask, device=t_raw.device, dtype=torch.bool)
            target_rows = torch.where(diff_sel.unsqueeze(-1), t_diff, t_raw)
            V_t = target_rows.shape[0]

            pc_raw = pc if pc is not None else torch.empty(0, t_raw.shape[-1], device=t_raw.device, dtype=t_raw.dtype)
            pfc_raw = (
                pfc
                if pfc is not None
                else torch.empty(0, t_raw.shape[-1] + prediction_length, device=t_raw.device, dtype=t_raw.dtype)
            )
            if pc_raw.shape[-1] != t_raw.shape[-1]:
                raise ValueError(
                    "Past covariates and targets have to have the same length "
                    f"(expected {t_raw.shape[-1]}, got {pc_raw.shape[-1]})."
                )
            expected_future_covariate_length = t_raw.shape[-1] + prediction_length
            if pfc_raw.shape[-1] < expected_future_covariate_length:
                raise ValueError(
                    "Future known covariates must be at least as long as the target + the prediction length "
                    f"(expected at least {expected_future_covariate_length}, got {pfc_raw.shape[-1]})."
                )
            if pfc_raw.shape[-1] > expected_future_covariate_length:
                # Extra future-known covariate steps are harmless (e.g. when an
                # overlong requested horizon is capped); keep only what the model
                # consumes.
                pfc_raw = pfc_raw[..., :expected_future_covariate_length]
            V = V_t + pc_raw.shape[0] + pfc_raw.shape[0]

            next_group = (max(group_vector) if group_vector else 0) + 1
            group_sample_map[next_group] = sample_idx
            group_target_indices[next_group] = list(range(V_t))
            group_row_is_diff[next_group] = list(diff_mask)

            left_pad = max_target_len - t_raw.shape[-1]
            right_pad = prediction_length
            item = torch.concat(
                [
                    pad(target_rows, (left_pad, right_pad), value=torch.nan),
                    pad(pc_raw, (left_pad, right_pad), value=torch.nan),
                    pad(pfc_raw, (left_pad, 0), value=torch.nan),
                ],
                dim=0,
            )
            x.append(item)

            target_mask.extend([True] * V_t + [False] * (V - V_t))
            group_vector.extend([next_group] * V)

        device = prepared_targets[0]["raw"].device
        x = torch.concat(x, dim=0)
        group_vector = torch.tensor(group_vector, device=device, dtype=torch.float32)
        target_mask = torch.tensor(target_mask, device=device, dtype=torch.bool)

        return (
            {"x": x, "group_vector": group_vector, "target_mask": target_mask},
            [],
            {
                "group_sample_map": group_sample_map,
                "group_vector": group_vector,
                "target_mask": target_mask,
                "prediction_window_is_padded": True,
                "single_pass": True,
                "transform_params": transform_params,
                "diff_masks": diff_masks,
                "group_target_indices": group_target_indices,
                "group_row_is_diff": group_row_is_diff,
            },
        )

    def transform_output(
        self,
        output: torch.Tensor,
        *args,
        group_sample_map: dict[int, int],
        group_vector: torch.Tensor,
        target_mask: torch.Tensor,
        transform_params: list[dict],
        diff_masks: list[list[bool]],
        group_target_indices: dict[int, list[int]],
        group_row_is_diff: dict[int, list[bool]],
        **kwargs,
    ):
        """Calibrate the band per row type and integrate differenced rows back."""
        output = output.detach().cpu()
        group_vector = group_vector.detach().cpu()
        target_mask = target_mask.detach().cpu()

        sample_rows = [[None] * len(mask) for mask in diff_masks]
        group_id = int(group_vector[0].item())

        while group_id <= group_vector.max().item():
            sample_idx = group_sample_map[group_id]
            target_indices = group_target_indices[group_id]
            mask = (group_vector == group_id) & target_mask
            group_sample = output[mask]

            # ``row_is_diff`` is per target row: differenced and raw rows can mix.
            row_is_diff = torch.tensor(group_row_is_diff[group_id], dtype=torch.bool)

            # Raw-row band calibration: sqrt(raw_band_scale) * t^(raw_band_exponent-0.5)
            # multiplies the model's per-step offset around its median.
            not_diff = ~row_is_diff
            if not_diff.any():
                q_med = group_sample.shape[1] // 2
                raw = group_sample[not_diff]
                median_step = raw[:, q_med : q_med + 1, :]
                offset = raw - median_step
                t_idx = torch.arange(1, raw.shape[-1] + 1, device=raw.device, dtype=raw.dtype)
                horizon_factor = t_idx ** (self.cfg.raw_band_exponent - 0.5)
                rescaled = median_step + (self.cfg.raw_band_scale**0.5) * horizon_factor * offset
                group_sample = group_sample.clone()
                group_sample[not_diff] = rescaled

            if row_is_diff.any():
                last_values = transform_params[sample_idx]["last_values"].detach().cpu()[target_indices]
                group_sample = self._diff_inverse(group_sample, row_is_diff, {"last_values": last_values})

            for row, target_idx in zip(group_sample, target_indices):
                sample_rows[sample_idx][target_idx] = row

            group_id += 1

        return [torch.stack(rows, dim=0) for rows in sample_rows]

    def _detect_trends(self, x: torch.Tensor) -> list[bool]:
        """Per-row trend flags from an ``ols_sigma`` fit on the last ``trend_window`` fraction."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        results = []
        for row in x:
            window = max(1, round(self.cfg.trend_window * row.shape[-1]))
            row = row[-window:]
            mask = ~torch.isnan(row)
            indices = torch.where(mask)[0].detach().cpu().numpy().astype(np.float64)
            vals = row[mask].detach().cpu().numpy().astype(np.float64)
            if len(vals) < 4:
                results.append(False)
                continue
            score, r2 = self._ols_sigma_score(indices, vals, len(row))
            results.append(score >= self.cfg.trend_threshold and r2 >= self.cfg.trend_min_r2)
        return results

    @staticmethod
    def _ols_sigma_score(indices: np.ndarray, vals: np.ndarray, length: int) -> tuple[float, float]:
        """Return ``(|slope_z| * length, r2)`` for an OLS fit on z-scored values."""
        std = float(np.std(vals, ddof=1))
        if std < 1e-12:
            return 0.0, 0.0
        z = (vals - np.mean(vals)) / std
        coeffs = np.polyfit(indices, z, 1)
        slope_z = float(coeffs[0])
        predicted = coeffs[0] * indices + coeffs[1]
        ss_res = float(np.sum((z - predicted) ** 2))
        ss_tot = float(np.sum((z - np.mean(z)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return abs(slope_z) * length, r2

    @classmethod
    def _diff_forward(cls, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """First-order row-wise differencing, retaining last observed levels for inversion."""
        diff = x[..., 1:] - x[..., :-1]
        diff = torch.cat([torch.zeros_like(x[..., :1]), diff], dim=-1)
        return diff, cls._diff_params(x)

    @staticmethod
    def _diff_params(x: torch.Tensor) -> dict:
        """Retain last observed levels (NaN-safe, falling back to 0.0) for inversion."""
        not_nan = ~torch.isnan(x)
        last_idx = not_nan.long().cumsum(dim=-1).argmax(dim=-1)
        last_values = x.gather(-1, last_idx.unsqueeze(-1)).squeeze(-1)
        return {"last_values": torch.nan_to_num(last_values, nan=0.0)}

    def _diff_inverse(self, sample: torch.Tensor, mask: torch.Tensor, params: dict) -> torch.Tensor:
        """Invert first-order differencing on ``mask`` rows: median path + rss band term."""
        out = sample.clone()
        if not mask.any():
            return out
        constants = params["last_values"]
        broadcast_shape = (-1,) + (1,) * (sample.ndim - 1)
        diff = sample[mask]

        q_med = diff.shape[1] // 2
        median_step = diff[:, q_med : q_med + 1, :]
        median_path = torch.cumsum(median_step, dim=-1)
        offset = diff - median_step

        sign = offset.sum(dim=-1, keepdim=True).sign()
        rss = torch.sqrt(torch.cumsum(offset * offset, dim=-1))
        t_idx = torch.arange(1, diff.shape[-1] + 1, device=diff.device, dtype=diff.dtype)
        horizon_factor = t_idx ** (self.cfg.diff_band_exponent - 0.5)
        band = sign * rss * horizon_factor

        recon = median_path + (self.cfg.diff_band_scale**0.5) * band
        out[mask] = recon + constants[mask].reshape(broadcast_shape)
        return out
