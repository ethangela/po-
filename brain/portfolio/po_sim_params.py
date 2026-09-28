import bisect
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

from brain.cfg.config_manager import BrainConfigManager
from brain.portfolio.data_interface import DataInterface, GCSDataInterface
from brain.portfolio.utils import CME_DEFAULT_OPTIMIZER_CONFIG, MINUTE


@dataclass
class POSimParams:
    config_manager: BrainConfigManager
    external_fill_only: bool
    per_product_external_fill_dim: str
    fill_model: str
    ref_price_name: str
    optimizer_config: Dict[str, Any]
    execution_optimizer_config: Dict[str, Any]
    verbose_level: int
    produce_ep: bool
    produce_constraint_info: bool
    max_hurdle: float
    target_average_time: float
    build_fundamental_alpha_curve: bool
    intraday_portfolio_ratio: float
    universe: List[str]
    tz: str
    market_start_time: pd.Timedelta
    market_end_time: pd.Timedelta
    start_time: pd.Timedelta
    end_time: pd.Timedelta
    time_gap: pd.Timedelta
    leverage_ratio: float
    realized_spread_cost_ratio: float
    transact_cost_feedback: bool
    cost_ema_alpha: float
    batch_days: int
    date: str = ""
    data_interface: DataInterface = DataInterface("")


def compute_components_to_remove(
    universe: List[str], relations: List[str]
) -> List[List[float]]:
    components_to_remove = []
    universe_map = {asset: index for index, asset in enumerate(universe)}
    for relation in relations:
        component_vector = np.zeros(len(universe))
        components = relation.split("=")[0].split("+")
        for component in components:
            coeff, name = component.split("*")
            coeff = float(coeff[1:-1])
            component_vector[universe_map[name]] = coeff
        components_to_remove.append(component_vector)
    return components_to_remove


def build_po_sim_params(cm: BrainConfigManager) -> POSimParams:
    section = "PO_SIM"
    universe = cm.get_universe()

    external_fill_only = cm.get_bool(section, "external_fill_only")
    per_product_external_fill_dim = cm.get_with_default(
        section, "per_product_external_fill_dim", "", type_converter=str
    )
    fill_model = cm.get_str(section, "fill_model")
    ref_price_name = cm.get_str(section, "ref_price_name")

    optimizer_config = CME_DEFAULT_OPTIMIZER_CONFIG
    optimizer_config["all_assets"] = universe
    optimizer_config["sector_neutral"] = cm.get_bool(section, "sector_neutral")

    numerical_config_params = [
        ("liquidate_at_eod", "bool"),
        ("additional_horizon_from_checkpoint", "bool"),
        ("target_horizon", "float"),
        ("alpha_scalings", "list"),
        ("gamma", "float"),
        ("gmv_limit", "float"),
        ("trade_limit_1m", "float"),
        ("nmv_limit_ratio", "float"),
        ("turnover_limit", "float"),
        ("estimated_spread_cost_ratio", "float"),
        ("estimated_cost_volume_ratio", "float"),
        ("historical_spread_cost_weight", "float"),
        ("include_quadratic_transact_cost", "bool"),
        ("estimated_quadratic_transact_cost_ratio", "float"),
        ("enable_vlm_limit", "bool"),
        ("trade_limit_multiplier_list", "list"),
        ("vol_limit", "float"),
        ("num_of_horizons_to_ignore_factor_neutral", "int"),
        ("alpha_realization_coeff", "float"),
        ("produce_constraint_info", "bool"),
        ("liquidation_mode", False),
        ("liquidation_remaining_time", np.inf),
        ("liquidation_entrance_threshold", -np.inf),
        ("liquidation_exit_threshold", np.inf),
    ]
    for param, dtype_or_value in numerical_config_params:
        match dtype_or_value:
            case "float":
                optimizer_config["numerical_config"][param] = cm.get_float(
                    section, param
                )
            case "bool":
                optimizer_config["numerical_config"][param] = cm.get_bool(
                    section, param
                )
            case "list":
                optimizer_config["numerical_config"][param] = cm.get_list(
                    section, param
                )
            case "int":
                optimizer_config["numerical_config"][param] = cm.get_int(section, param)
            case _:
                optimizer_config["numerical_config"][param] = dtype_or_value

    optimizer_config["numerical_config"]["execution_target_horizon"] = cm.get_with_default(
        section,
        "execution_target_horizon",
        optimizer_config["numerical_config"]["target_horizon"],
        type_converter=float,
    )

    optimizer_config["numerical_config"]["dynamic_gamma_vol_adapt"] = cm.get_with_default(
        section,
        "dynamic_gamma_vol_adapt",
        False,
        type_converter=lambda v: str(v).strip().lower() in ("1", "true", "yes", "on"),
    )
    optimizer_config["numerical_config"]["mosek_num_threads"] = cm.get_int(
        section, "mosek_num_threads"
    )

    if cm.has_option(section, "relations"):
        try:
            relations = cm.get_list(section, "relations")
            optimizer_config["components_to_remove"] = compute_components_to_remove(
                universe, relations
            )
            print(f"PO loaded {len(relations)} relations: {relations}")
        except:
            print(f"Relations are not consistent with universe, using empty relations")
            optimizer_config["components_to_remove"] = []
    else:
        print(f"PO no relations found")
        optimizer_config["components_to_remove"] = []

    horizons = cm.get_list(section, "horizons")
    optimizer_config["alpha_horizons"] = [
        int(h / 60) for h in np.diff(horizons, prepend=0)
    ]

    execution_optimizer_config = {}
    if fill_model == "secondly_aggressive":
        execution_optimizer_config["alpha_horizons"] = cm.get_list(
            section, "execution_alpha_horizons"
        )
        execution_optimizer_config["gamma"] = cm.get_float(
            section, "execution_optimizer_gamma"
        )
        execution_optimizer_config["eta_multiplier"] = cm.get_float(
            section, "execution_optimizer_eta_multiplier"
        )
    elif fill_model == "algo_container_spirit":
        execution_optimizer_config["mtm_mark_price"] = cm.get_with_default(
            section, "spirit_mtm_mark_price", None, type_converter=str
        )

    tz = cm.get_tz()
    market_start_time = pd.Timedelta(cm.get_value(section, "market_start_time"))
    market_end_time = pd.Timedelta(cm.get_value(section, "market_end_time"))
    start_time = pd.Timedelta(cm.get_value(section, "start_time"))
    end_time = pd.Timedelta(cm.get_value(section, "end_time"))
    time_gap = pd.Timedelta(cm.get_value(section, "time_gap"))
    if start_time < market_start_time:
        raise ValueError(
            f"start_time {start_time} is before market_start_time {market_start_time}"
        )
    if end_time > market_end_time:
        raise ValueError(
            f"end_time {end_time} is after market_end_time {market_end_time}"
        )
    start_time_minute_since_market_start = int(
        (start_time - market_start_time) / MINUTE
    )
    end_time_minute_since_market_start = int((end_time - market_start_time) / MINUTE)
    checkpoints_list = cm.get_list(section, "checkpoint_minute_since_market_start")
    if start_time_minute_since_market_start != checkpoints_list[0]:
        print(
            f"start_time {start_time_minute_since_market_start} is not consistent with checkpoints[0] {checkpoints_list[0]}"
        )
    if end_time_minute_since_market_start != checkpoints_list[-1]:
        print(
            f"end_time {end_time_minute_since_market_start} is not consistent with checkpoints[-1] {checkpoints_list[-1]}"
        )
    start_index = bisect.bisect_right(
        checkpoints_list, start_time_minute_since_market_start
    )
    end_index = bisect.bisect_left(checkpoints_list, end_time_minute_since_market_start)
    optimizer_config["checkpoint_minute_since_market_start"] = (
        [start_time_minute_since_market_start]
        + checkpoints_list[start_index:end_index]
        + [end_time_minute_since_market_start]
    )
    print(
        f"Adjusted optimizer_config['checkpoint_minute_since_market_start']: {optimizer_config['checkpoint_minute_since_market_start']}"
    )

    verbose_level = cm.get_int(section, "verbose_level")
    produce_ep = cm.get_bool(section, "produce_ep")
    produce_constraint_info = cm.get_bool(section, "produce_constraint_info")
    max_hurdle = cm.get_float(section, "max_hurdle")
    target_average_time = cm.get_float(section, "target_average_time")
    build_fundamental_alpha_curve = cm.get_bool(
        section, "build_fundamental_alpha_curve"
    )
    intraday_portfolio_ratio = cm.get_float(section, "intraday_portfolio_ratio")
    leverage_ratio = cm.get_float(section, "leverage_ratio")
    realized_spread_cost_ratio = cm.get_float(section, "realized_spread_cost_ratio")
    transact_cost_feedback = cm.get_bool(section, "transact_cost_feedback")
    cost_ema_alpha = cm.get_float(section, "cost_ema_alpha")
    data_paths = cm.get_po_data_paths()
    batch_days = cm.get_int(section, "batch_days")

    return POSimParams(
        config_manager=cm,
        external_fill_only=external_fill_only,
        per_product_external_fill_dim=per_product_external_fill_dim,
        fill_model=fill_model,
        ref_price_name=ref_price_name,
        optimizer_config=optimizer_config,
        execution_optimizer_config=execution_optimizer_config,
        verbose_level=verbose_level,
        produce_ep=produce_ep,
        produce_constraint_info=produce_constraint_info,
        max_hurdle=max_hurdle,
        target_average_time=target_average_time,
        build_fundamental_alpha_curve=build_fundamental_alpha_curve,
        intraday_portfolio_ratio=intraday_portfolio_ratio,
        universe=universe,
        tz=tz,
        market_start_time=market_start_time,
        market_end_time=market_end_time,
        start_time=start_time,
        end_time=end_time,
        time_gap=time_gap,
        leverage_ratio=leverage_ratio,
        realized_spread_cost_ratio=realized_spread_cost_ratio,
        transact_cost_feedback=transact_cost_feedback,
        cost_ema_alpha=cost_ema_alpha,
        batch_days=batch_days,
        data_interface=GCSDataInterface(data_paths, universe, read_resolver=cm.resolve_read_file),
    )
