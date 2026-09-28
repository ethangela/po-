import itertools
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

MINUTE = pd.Timedelta("00:01:00")

DEFAULT_OPTIMIZER_CONFIG = {
    "all_assets": ["A"],
    "components_to_remove": [],
    "checkpoint_minute_since_market_start": [0, 1],
    "alpha_horizons": [1],
    "numerical_config": {
        "scale": 1,
        "gamma": 1,
        "limit": 1000000,
        "liquidate_at_eod": False,
        "enable_gmv_constraint": False,
        "include_transact_cost": False,
        "estimated_spread_cost_ratio": 1,
        "optimality_criteria_multiplier": 1,
        "gradient_analysis": False,
    },
}

SOYBEAN_CRUSH_DEFAULT_OPTIMIZER_CONFIG = {
    "all_assets": ["ZL", "ZM", "ZS", "CS"],
    "components_to_remove": [[0.11, 0.022, -1, -1]],
    "checkpoint_minute_since_market_start": [
        0,
        60,
        120,
        150,
        270,
        390,
    ],  # 9:30, 10:30, 11:30, 12:00, 14:00, 16:00
    "alpha_horizons": [1, 1, 1, 2, 5, 5, 15],
    "numerical_config": {
        "scale": 1000000,
        "gamma": 1 / 2,
        "limit": 1000000,
        "liquidate_at_eod": True,
        "enable_gmv_constraint": True,
        "include_transact_cost": True,
        "estimated_spread_cost_ratio": 1,
        "optimality_criteria_multiplier": 1,
        "gradient_analysis": True,
    },
}

CME_DEFAULT_OPTIMIZER_CONFIG = {
    "all_assets": ["ES", "NQ", "YM", "RTY", "EMD", "ZN", "ZB"],
    "sector_neutral": False,
    "components_to_remove": [],
    "checkpoint_minute_since_market_start": [
        0,
        60,
        120,
        150,
        270,
        390,
    ],  # 9:30, 10:30, 11:30, 12:00, 14:00, 16:00
    "alpha_horizons": [1, 1, 1, 2, 5, 5, 15],
    "numerical_config": {
        "scale": 1e6,
        "gamma": 2e-3,
        "gmv_limit": 1e7,
        "liquidate_at_eod": True,
        "enable_gmv_constraint": True,
        "include_transact_cost": True,
        "include_quadratic_transact_cost": False,
        "estimated_spread_cost_ratio": 1,
        "estimated_quadratic_transact_cost_ratio": 1,
        "optimality_criteria_multiplier": 1,
        "gradient_analysis": True,
    },
}


def get_asset_ref_map_da(asset_info):
    return xr.DataArray(
        asset_info["ref_product"].values,
        dims=["asset2"],
        coords={"asset2": asset_info.index.values},
    )


def get_beta_from_cov(covariance_data, ref_asset_per_asset2):
    asset_size = min(
        size
        for name, size in covariance_data.sizes.items()
        if name in ["asset1", "asset2"]
    )
    var = xr.apply_ufunc(
        np.diag,
        covariance_data,
        input_core_dims=[["asset1", "asset2"]],
        output_core_dims=[["asset"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"asset": asset_size}},
    )
    cov_with_ref = covariance_data.sel(
        asset1=ref_asset_per_asset2, drop=True
    ).rename({"asset2": "asset"})
    beta = cov_with_ref / var
    return beta


def get_sector_from_index(sector_index):
    unique_indices = np.unique(sector_index)
    return [[1 if index == i else 0 for index in sector_index] for i in unique_indices]


def da_to_joined_dict(da, dims=None, sep="_"):
    if dims is None:
        dims = da.dims
    stk = da.transpose(*dims).stack(idx=dims)
    key_tuples = stk.indexes["idx"].tolist()
    values = stk.values.tolist()
    return {sep.join(map(str, p)): v for p, v in zip(key_tuples, values)}


to_td = pd.to_timedelta


def normalize(x):
    return str(to_td(x)) if isinstance(x, np.timedelta64) else str(x)


def da_to_joined_dict_faster(da, dims=None, sep="_"):
    if dims is None:
        dims = da.dims
    da2 = da.transpose(*dims)
    values = np.ravel(da2.values)
    coords = [da2.coords[d].values for d in dims]
    return {
        sep.join(normalize(k) for k in key): value
        for key, value in zip(itertools.product(*coords), values)
    }


def compute_max_hurdle_and_decay_rate(alpha_hurdle, risk_hurdle, remaining_time):
    risk_penalty_coef = 0.2
    risk_hurdle = risk_hurdle * risk_penalty_coef
    total_hurdle = alpha_hurdle + risk_hurdle
    total_hurdle_mean = total_hurdle.mean()
    total_hurdle_std = total_hurdle.std()
    if total_hurdle_std > 0:
        zscore_alpha_hurdle = (alpha_hurdle - total_hurdle_mean) / total_hurdle_std
        zscore_risk_hurdle = (risk_hurdle - total_hurdle_mean) / total_hurdle_std
    else:
        zscore_alpha_hurdle = np.zeros_like(alpha_hurdle)
        zscore_risk_hurdle = np.zeros_like(risk_hurdle)
    cdf_alpha_hurdle = norm.cdf(zscore_alpha_hurdle)
    cdf_risk_hurdle = norm.cdf(zscore_risk_hurdle)
    max_hurdle = 2 - 2 * cdf_alpha_hurdle
    decay_rate = 1 * (cdf_alpha_hurdle + cdf_risk_hurdle)

    end_time_adjustment = 1 - np.minimum(remaining_time, 10) / 10
    max_hurdle += end_time_adjustment
    decay_rate += end_time_adjustment

    return max_hurdle, decay_rate


def compute_max_hurdle_and_decay_rate2(
    alpha, risk, remaining_time, max_hurdle, target_average_time
):
    risk_coef = 0.2
    risk = risk * risk_coef
    total = alpha + risk
    total_mean = total.mean()
    total_std = total.std()
    if total_std > 0:
        zscore = (total - total_mean) / total_std
    else:
        zscore = np.zeros_like(total)
    cdf = norm.cdf(zscore)

    x_intercept = target_average_time * 2 * (1 - cdf)
    x_intercept *= np.clip(remaining_time / 10, 0.01, 1)
    decay_rate = max_hurdle / x_intercept
    return np.ones_like(alpha) * max_hurdle, decay_rate
