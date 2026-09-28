import numpy as np
import pandas as pd
import xarray as xr

from brain.portfolio.config_tuner import ConfigTuner
from brain.portfolio.optimizer import Optimizer
from brain.portfolio.utils import (
    MINUTE,
    get_asset_ref_map_da,
    get_beta_from_cov,
    get_sector_from_index,
    da_to_joined_dict_faster,
    compute_max_hurdle_and_decay_rate2,
)


class POExecutor:
    def __init__(self, po_sim_params):
        print(f"po_sim_params: {po_sim_params}")
        self.date = po_sim_params.date
        self.date_time = pd.Timestamp(self.date, tz=po_sim_params.tz)
        self.optimizer_config = po_sim_params.optimizer_config
        self.numerical_config = self.optimizer_config["numerical_config"]
        self.config_tuner = ConfigTuner(po_sim_params)
        self.verbose_level = po_sim_params.verbose_level

        self.universe = po_sim_params.universe
        self.market_start_time = po_sim_params.market_start_time
        self.market_end_time = po_sim_params.market_end_time
        self.market_duration_minutes = int(
            (self.market_end_time - self.market_start_time) / MINUTE
        )
        self.market_duration_in_half_hours = self.market_duration_minutes / 30

        self.leverage_ratio = po_sim_params.leverage_ratio
        self.spread_cost_ratio = self.numerical_config["estimated_spread_cost_ratio"]
        self.cost_volume_ratio = self.numerical_config["estimated_cost_volume_ratio"]
        self.quadratic_transact_cost_ratio = self.numerical_config[
            "estimated_quadratic_transact_cost_ratio"
        ]
        self.historical_spread_cost_weight = self.numerical_config[
            "historical_spread_cost_weight"
        ]
        self.transact_cost_feedback = po_sim_params.transact_cost_feedback

        self.build_fundamental_alpha_curve = po_sim_params.build_fundamental_alpha_curve
        self.start_of_day_initialized = False
        self.fundamental_alpha_curve_active = False
        self.intraday_portfolio_ratio = po_sim_params.intraday_portfolio_ratio
        self.total_gmv_limit = self.numerical_config["gmv_limit"]
        self.intraday_gmv_limit = self.total_gmv_limit * self.intraday_portfolio_ratio
        self.current_gmv_limit = 0
        self.daily_gmv_window_length = 10
        self.nmv_limit_ratio = self.numerical_config["nmv_limit_ratio"]
        self.trade_limit_multiplier_list = self.numerical_config[
            "trade_limit_multiplier_list"
        ] + [np.inf]
        self.turnover_limit = self.numerical_config["turnover_limit"]

        self.vol_limit = self.numerical_config["vol_limit"]

        self.max_hurdle = po_sim_params.max_hurdle
        self.target_average_time = po_sim_params.target_average_time

        self.alpha_horizons = np.cumsum(self.optimizer_config["alpha_horizons"])

        self.optimizer = Optimizer(self.optimizer_config)
        self.optimizer_start_flag = False

        self.data_interface = po_sim_params.data_interface
        self.pp_dict = None
        self.time_list = []
        self.q_list = []
        self.constraint_info_dict = {}

        self.load_data()

    def load_data(self):
        self.asset_info_data = self.data_interface.get_asof_data(
            "asset_info", self.date
        ).loc[self.universe]
        self.trading_end_time_minute_since_market_start = self.asset_info_data[
            "trading_end_time_minute_since_sod"
        ].values
        self.part_rate = self.asset_info_data["part_rate"].values
        self.spread_cost_ratio = (
            self.spread_cost_ratio
            * self.asset_info_data["spread_cost_ratio_multiplier"].values
        )
        # (TODO) do we want a seperate quadratic_transact_cost_ratio_multiplier?
        self.quadratic_transact_cost_ratio = (
            self.quadratic_transact_cost_ratio
            * self.asset_info_data["spread_cost_ratio_multiplier"].values
        )
        self.gamma_multiplier = xr.DataArray(
            np.sqrt(self.asset_info_data["gamma_multiplier"].values), dims=["asset1"]
        ).assign_coords(asset1=self.universe) * xr.DataArray(
            np.sqrt(self.asset_info_data["gamma_multiplier"].values), dims=["asset2"]
        ).assign_coords(
            asset2=self.universe
        )

        self.price_name = "s1_mid_price_close"
        self.spread_fraction_name = "s2_spread_pct_mean"
        self.bid_size_name = "s2_bid_size_mean"
        self.ask_size_name = "s2_ask_size_mean"
        self.share_volume_name = "s3_share_volume_total"
        self.minutebar_names = [
            self.price_name,
            self.spread_fraction_name,
            self.bid_size_name,
            self.ask_size_name,
            self.share_volume_name,
        ]
        self.alpha_names = [f"y_{i * 60}" for i in self.alpha_horizons]

        start = int(self.market_start_time / MINUTE)
        end = start + self.optimizer_config["checkpoint_minute_since_market_start"][-1]
        start_time = self.date_time + pd.Timedelta(minutes=start + 1)
        end_time = self.date_time + pd.Timedelta(minutes=end)
        vlm_name = "timebin_mean_s3_share_volume_total"
        spread_name = "timebin_mean_s2_spread_pct_mean"
        mktstats_names = [vlm_name, spread_name]
        mktstats_data = self.data_interface.get_asof_data(
            "mktstats_timebin", self.date, mktstats_names
        )
        self.vlm_data = (
            mktstats_data.sel(columns=vlm_name, drop=True)
            .sel(timestamp=slice(start_time, end_time))
            .transpose("timestamp", "asset")
            .values
        ) * self.part_rate
        self.historical_spread_fraction = mktstats_data.sel(
            columns=spread_name, drop=True
        )
        try:
            vlm_coeff_names = ["vlm_coeff"]
            # per Shubham, this vlm_coeff is more like forward looking on events
            vlm_coeff = (
                self.data_interface.get_asof_data(
                    "econevents", self.date, vlm_coeff_names
                )
                .sel(columns=vlm_coeff_names[0], drop=True)
                .sel(timestamp=slice(start_time - MINUTE, end_time))
                .transpose("timestamp", "asset")
                .values
            )
            self.vlm_data = self.vlm_data * vlm_coeff
            print("POExecutor updated vlm_data by vlm_coeff")
        except Exception as e:
            print(f"POExecutor error in getting vlm_coeff: {e}")

        self.covariance_data = self.data_interface.get_asof_data(
            "timebin_cov", self.date
        ).sel(asset1=self.universe, asset2=self.universe)
        self.gamma_adjusted_covariance_data = (
            self.covariance_data * self.gamma_multiplier
        ).sel(timestamp=slice(start_time, end_time))

    def slice_with_range(self, df, start_timestamp, end_timestamp):
        df = df.between_time(
            start_timestamp.time(), end_timestamp.time(), inclusive="both"
        )
        if start_timestamp not in df.index:
            insert = pd.DataFrame(
                [df.iloc[0].values], columns=df.columns, index=[start_timestamp]
            )
            df = pd.concat([insert, df])
        if end_timestamp not in df.index:
            insert = pd.DataFrame(
                [df.iloc[-1].values], columns=df.columns, index=[end_timestamp]
            )
            df = pd.concat([df, insert])
        return df

    def build_curve_from_fundamental_alpha(self):
        self.fundamental_alpha_data = self.data_interface.get_asof_data(
            "fundamental_alpha_panel", self.date
        )[self.universe]
        start_timestamp = self._minute_since_market_start_to_time(
            self.optimizer_config["checkpoint_minute_since_market_start"][0]
        )
        end_timestamp = self._minute_since_market_start_to_time(
            self.optimizer_config["checkpoint_minute_since_market_start"][-1]
        )
        self.fundamental_alpha_data = self.slice_with_range(
            self.fundamental_alpha_data, start_timestamp, end_timestamp
        )

        times = self.fundamental_alpha_data.index
        self.fundamental_alpha_checkpoints_times_since_market_start = [
            self._time_to_minute_since_market_start(t) for t in times
        ]
        alpha_horizons = [
            int(h)
            for h in np.diff(
                self.fundamental_alpha_checkpoints_times_since_market_start
            )
        ]
        self.setup_start_of_day_as_of(
            self.fundamental_alpha_checkpoints_times_since_market_start,
            alpha_horizons,
        )
        self.run_start_of_day_snapshot()

    def start_day(self):
        if self.build_fundamental_alpha_curve:
            self.build_curve_from_fundamental_alpha()
            self.fundamental_alpha_curve_active = True
        self.setup_start_of_day_as_of(
            self.optimizer_config["checkpoint_minute_since_market_start"],
            self.optimizer_config["alpha_horizons"],
        )
        self.start_of_day_initialized = True

    def _time_to_minute_since_market_start(self, time):
        return int((time - (self.date_time + self.market_start_time)) / MINUTE)

    def _minute_since_market_start_to_time(self, minute_since_market_start):
        return (
            self.date_time + self.market_start_time + MINUTE * minute_since_market_start
        )

    def get_orthogonal_factor_exposures(self, cov_data, sector_neutral=True):
        n = len(self.universe)
        if sector_neutral:
            asset_ref_map_da = get_asset_ref_map_da(self.asset_info_data)
            sector_index = self.asset_info_data["sector_index"].values
            sectors = get_sector_from_index(sector_index)
            beta = get_beta_from_cov(cov_data, asset_ref_map_da).values
            orthogonal_factor_exposures = beta * np.array(sectors)
            return orthogonal_factor_exposures
        else:
            return np.zeros((0, n))

    def setup_start_of_day_as_of(
        self, checkpoint_minute_since_market_start, alpha_horizons
    ):
        orthogonal_factor_exposures = self.get_orthogonal_factor_exposures(
            self.gamma_adjusted_covariance_data.mean(dim="timestamp"),
            self.optimizer_config["sector_neutral"],
        )
        return_cov = self.gamma_adjusted_covariance_data.values
        if np.isnan(return_cov).any():
            print(f"There is na in the return covariance data after gamma multiplier")

        self.sod_data = {
            "return_cov": return_cov,
            "orthogonal_factor_exposures": orthogonal_factor_exposures,
        }

        self.optimizer.checkpoint_minute_since_market_start = (
            checkpoint_minute_since_market_start
        )
        self.optimizer.alpha_horizons = alpha_horizons
        self.optimizer.setup_start_of_day_as_of(
            self.sod_data,
            self.trading_end_time_minute_since_market_start,
            self.vlm_data,
        )

    def update_gmv_limit(self, time, target_gmv_limit):
        current_gmv = self.pp_dict["GMV"]
        self.prev_gmv_limit = max(current_gmv, self.current_gmv_limit)
        self.prev_gmv_time = time
        self.target_gmv_limit = target_gmv_limit
        self.window_length = np.ceil(
            self.daily_gmv_window_length
            * self.safe_div(
                np.abs(self.target_gmv_limit - self.prev_gmv_limit),
                self.intraday_gmv_limit,
            )
        )
        self.target_gmv_time = time + self.window_length * MINUTE
        if self.verbose_level >= 1:
            print(
                f"POExecutor updating gmv limit from {self.prev_gmv_limit} to {self.target_gmv_limit} at {time}"
                f" with window length {self.window_length}"
                f" current_gmv: {current_gmv}"
            )

    def run_start_of_day_snapshot(self):
        # items that need to be better picked:
        # level_size, spread_fraction
        # anything like daily stats here?
        # do we do anything for intraday run?
        # level size even updated to some volume based number?
        minute_since_market_start = (
            self.fundamental_alpha_checkpoints_times_since_market_start[0]
        )
        time = self._minute_since_market_start_to_time(minute_since_market_start)
        gmv_limit = self.total_gmv_limit * (1 - self.intraday_portfolio_ratio)
        portfolio_volatility_limit = (
            (self.vol_limit * (gmv_limit / self.leverage_ratio)) ** 2
            / (252 * self.market_duration_minutes)
            * (self.market_duration_minutes - minute_since_market_start)
        )

        minutebar_data = self.data_interface.get_intraday_data(
            "minutebar", self.date, time, self.minutebar_names
        )
        price = minutebar_data.sel(columns=self.price_name, drop=True).values
        contract_unit = self.asset_info_data["handle_value"].values
        dollar_value_per_share = price * contract_unit
        level_size = (
            minutebar_data.sel(columns=[self.bid_size_name, self.ask_size_name])
            .mean(dim="columns")
            .values
        )
        share_position = np.zeros_like(price)

        current_spread_fraction = (
            minutebar_data.sel(columns=self.spread_fraction_name, drop=True).values / 6
        )
        historical_spread_fraction = (
            self.historical_spread_fraction.sel(timestamp=time, drop=True).values / 6
        )
        spread_fraction = (
            current_spread_fraction * (1 - self.historical_spread_cost_weight)
            + historical_spread_fraction * self.historical_spread_cost_weight
        )
        if (spread_fraction < 0).any():
            print(f"There is negative spread fraction at {time}")
            spread_fraction = np.abs(spread_fraction)
        transact_cost_fraction = spread_fraction * self.spread_cost_ratio
        np.maximum(
            transact_cost_fraction,
            self.cost_volume_ratio,
            out=transact_cost_fraction,
        )

        tick_size = self.asset_info_data["tick_size"].values
        quadratic_tc_coef = np.sqrt(
            2
            / level_size
            * tick_size
            / price
            / dollar_value_per_share
            * self.quadratic_transact_cost_ratio
        )

        turnover_limit = np.inf

        alpha_data = self.fundamental_alpha_data
        asset_index = alpha_data.columns
        if "MES" in asset_index and "ES" in asset_index:
            alpha_data.loc[:, "MES"] = alpha_data.loc[:, "ES"]
        if "MNQ" in asset_index and "NQ" in asset_index:
            alpha_data.loc[:, "MNQ"] = alpha_data.loc[:, "NQ"]
        raw_alpha = alpha_data.values
        alpha = np.exp(raw_alpha) - 1
        optimizer_alpha = np.diff(alpha, axis=0)

        intraday_data = {
            "minute_since_market_start": minute_since_market_start,
            "portfolio_volatility_limit": portfolio_volatility_limit,
            "share_position": share_position,
            "dollar_value_per_share": dollar_value_per_share,
            "transact_cost_fraction": transact_cost_fraction,
            "quadratic_tc_coef": quadratic_tc_coef,
            "alpha": optimizer_alpha,
            "turnover_limit": turnover_limit,
            "position_limit": [
                gmv_limit * self.asset_info_data["position_limit_ratio"].values
            ],
            "gmv_limit": [gmv_limit],
            "nmv_limit": [gmv_limit * self.nmv_limit_ratio],
        }

        run_po = True
        if run_po:
            for trade_limit_multiplier in self.trade_limit_multiplier_list:
                try:
                    po_result = self.optimizer.optimize(
                        intraday_data,
                        trade_limit_multiplier=trade_limit_multiplier,
                    )
                    if trade_limit_multiplier > 1 and self.verbose_level >= 1:
                        print(
                            f"With trade limit multiplier {trade_limit_multiplier}, success in optimizer at {time}"
                        )
                    break
                except Exception as e:
                    if self.verbose_level >= 1:
                        print(
                            f"With trade limit multiplier {trade_limit_multiplier}, error in optimizer at {time}: {e}"
                        )
                finally:
                    self.optimizer.post_order_process(gc_collect=False)
        else:
            print(f"No curve built from fundamental alpha")
            run_po = False
        if run_po:
            print(
                f"fundamental_alpha_checkpoints_times_since_market_start: {self.fundamental_alpha_checkpoints_times_since_market_start}"
            )
            share_trade_desires = po_result["share_trade_desires"]
            target_share_position = share_position + share_trade_desires.cumsum(axis=0)
            self.target_share_position_on_checkpoints = np.concatenate(
                [share_position.reshape(1, -1), target_share_position], axis=0
            )
            average_target_share_position = (
                target_share_position + self.target_share_position_on_checkpoints[:-1]
            ) / 2
            ideal_pnl = (
                average_target_share_position * dollar_value_per_share * optimizer_alpha
            )
            print(ideal_pnl.sum())

    def get_target_share_position_from_fundamental_alpha(
        self, time_since_market_start
    ):
        for i in range(
            len(self.fundamental_alpha_checkpoints_times_since_market_start)
        ):
            if (
                time_since_market_start
                <= self.fundamental_alpha_checkpoints_times_since_market_start[i]
            ):
                break
        else:
            return self.target_share_position_on_checkpoints[-1]
        if i == 0:
            return self.target_share_position_on_checkpoints[0]
        weight = (
            self.fundamental_alpha_checkpoints_times_since_market_start[i]
            - time_since_market_start
        ) / (
            self.fundamental_alpha_checkpoints_times_since_market_start[i]
            - self.fundamental_alpha_checkpoints_times_since_market_start[i - 1]
        )
        return (
            self.target_share_position_on_checkpoints[i] * (1 - weight)
            + self.target_share_position_on_checkpoints[i - 1] * weight
        )

    def get_adjusted_share_trade_desire_from_fundamental_alpha(
        self, time_since_market_start
    ):
        adjusted_share_trade_desire = np.zeros(
            (len(self.alpha_horizons), len(self.universe))
        )
        previous_time_since_market_start = time_since_market_start
        for i in range(len(self.alpha_horizons)):
            current_time_since_market_start = (
                time_since_market_start + self.alpha_horizons[i]
            )
            adjusted_share_trade_desire[i] = (
                self.get_target_share_position_from_fundamental_alpha(
                    current_time_since_market_start
                )
                - self.get_target_share_position_from_fundamental_alpha(
                    previous_time_since_market_start
                )
            )
            previous_time_since_market_start = current_time_since_market_start
        return adjusted_share_trade_desire

    def run_snapshot(self, time, in_range=True):
        # 1. gather all the raw data
        self.P = self.data_interface.get_intraday_data("P", self.date, time)

        minute_since_market_start = self._time_to_minute_since_market_start(time)
        if in_range and not self.start_of_day_initialized:
            self.start_day()
        if minute_since_market_start % 30 == 0 and self.verbose_level >= 1:
            print(f"POExecutor at {time}")

        minutebar_data = self.data_interface.get_intraday_data(
            "minutebar", self.date, time, self.minutebar_names
        )

        alpha_data = (
            self.data_interface.get_intraday_data(
                "alpha_po", self.date, time, self.alpha_names
            )
            * self.numerical_config["alpha_scalings"][0]
        )
        for index, alpha_scaling in enumerate(
            self.numerical_config["alpha_scalings"][1:]
        ):
            alpha_data += (
                self.data_interface.get_intraday_data(
                    f"alpha_po_{index + 1}", self.date, time, self.alpha_names
                )
                * alpha_scaling
            )

        intraday_vlm_limit = (
            minutebar_data.sel(columns=self.share_volume_name, drop=True).values
            * self.part_rate
        )

        raw_share_position = self.P.sel(columns="Position", drop=True).values
        share_position = raw_share_position.copy()
        if self.fundamental_alpha_curve_active:
            share_position -= self.get_target_share_position_from_fundamental_alpha(
                minute_since_market_start
            )
        self.update_PP_dict_pre_po(time)

        turnover_limit = self.turnover_limit - self.turnover
        if turnover_limit < self.gmv_ratio:
            turnover_limit = self.gmv_ratio
        if turnover_limit < np.inf and self.verbose_level >= 2:
            print(
                f"time: {time}, turnover_limit: {turnover_limit}, self.turnover: {self.turnover}, self.gmv_ratio: {self.gmv_ratio}"
            )

        price = minutebar_data.sel(columns=self.price_name, drop=True).values
        contract_unit = self.asset_info_data["handle_value"].values
        dollar_value_per_share = price * contract_unit

        # 2. dynamically update the po config
        self.config_tuner.update(
            time, in_range, self.pp_dict, share_position, dollar_value_per_share
        )
        assert self.config_tuner.numerical_config is self.optimizer.numerical_config
        assert (
            self.config_tuner.numerical_config
            is self.optimizer.numerical_solver.numerical_config
        )

        # 3. compute the data for po
        level_size = (
            minutebar_data.sel(columns=[self.bid_size_name, self.ask_size_name])
            .mean(dim="columns")
            .values
        )

        current_spread_fraction = (
            minutebar_data.sel(columns=self.spread_fraction_name, drop=True).values / 6
        )
        historical_spread_fraction = (
            self.historical_spread_fraction.sel(timestamp=time, drop=True).values / 6
        )
        spread_fraction = (
            current_spread_fraction * (1 - self.historical_spread_cost_weight)
            + historical_spread_fraction * self.historical_spread_cost_weight
        )
        if (spread_fraction < 0).any():
            print(f"There is negative spread fraction at {time}")
            spread_fraction = np.abs(spread_fraction)
        transact_cost_fraction = spread_fraction * self.spread_cost_ratio
        np.maximum(
            transact_cost_fraction,
            self.cost_volume_ratio,
            out=transact_cost_fraction,
        )
        if self.transact_cost_feedback:
            realized_transact_cost_fraction = (
                self.P.sel(columns="CostPerTradeDesire", drop=True).values
                / dollar_value_per_share
            )
            np.maximum(
                transact_cost_fraction,
                realized_transact_cost_fraction,
                out=transact_cost_fraction,
            )

        tick_size = self.asset_info_data["tick_size"].values
        quadratic_tc_coef = np.sqrt(
            2
            / level_size
            * tick_size
            / price
            / dollar_value_per_share
            * self.quadratic_transact_cost_ratio
        )

        asset_index = alpha_data.get_index("asset")
        if "MES" in asset_index and "ES" in asset_index:
            alpha_data.loc[dict(asset="MES")] = alpha_data.sel(asset="ES")
        if "MNQ" in asset_index and "NQ" in asset_index:
            alpha_data.loc[dict(asset="MNQ")] = alpha_data.sel(asset="NQ")

        raw_alpha = alpha_data.transpose("columns", "asset").values
        alpha = np.exp(raw_alpha) - 1
        optimizer_alpha = np.diff(alpha, axis=0, prepend=0)

        # in some sense, it's more correct we search a beta,
        # such that the solution in PP matches the target portfolio volatility
        portfolio_volatility_limit = (
            (self.vol_limit * (self.intraday_gmv_limit / self.leverage_ratio)) ** 2
            / (252 * self.market_duration_minutes)
            * (self.market_duration_minutes - minute_since_market_start)
        )

        intraday_data = {
            "minute_since_market_start": minute_since_market_start,
            "portfolio_volatility_limit": portfolio_volatility_limit,
            "share_position": share_position,
            "dollar_value_per_share": dollar_value_per_share,
            "transact_cost_fraction": transact_cost_fraction,
            "quadratic_tc_coef": quadratic_tc_coef,
            "alpha": optimizer_alpha,
            "intraday_vlm_limit": intraday_vlm_limit,
            "turnover_limit": turnover_limit,
        }

        if in_range:
            if not self.optimizer_start_flag:
                self.optimizer_start_flag = True
                self.update_gmv_limit(time, self.intraday_gmv_limit)
            position_limit_ratio = self.asset_info_data["position_limit_ratio"].values
            remaining_time = int((self.target_gmv_time - time) / MINUTE)
            if remaining_time >= 1:
                ratio = (remaining_time - 1) / self.window_length
                self.current_gmv_limit = (
                    self.prev_gmv_limit * ratio
                    + self.target_gmv_limit * (1 - ratio)
                )
            gmv_limits = []
            for h in self.alpha_horizons:
                if h < remaining_time:
                    ratio = (remaining_time - h) / self.window_length
                    gmv_limits.append(
                        self.prev_gmv_limit * ratio
                        + self.target_gmv_limit * (1 - ratio)
                    )
                else:
                    gmv_limits.append(self.target_gmv_limit)
                    break
            intraday_data.update(
                {
                    "position_limit": [
                        np.array(gmv_limit * position_limit_ratio)
                        for gmv_limit in gmv_limits
                    ],
                    "gmv_limit": gmv_limits,
                    "nmv_limit": [
                        gmv_limit * self.nmv_limit_ratio for gmv_limit in gmv_limits
                    ],
                }
            )

        # 4. po runs
        run_po = in_range
        if np.isnan(price).any():
            print(f"There is na in the price data at {time}")
            run_po = False
        if np.isnan(spread_fraction).any():
            print(f"There is na in the spread fraction data at {time}")
            run_po = False
        if np.isnan(alpha).any():
            print(f"There is na in the alpha data at {time}")
            run_po = False
        if self.intraday_gmv_limit <= 0:
            run_po = False
        if run_po:
            for trade_limit_multiplier in self.trade_limit_multiplier_list:
                try:
                    if trade_limit_multiplier > 1:
                        self.optimizer.post_order_process(gc_collect=False)
                    po_result = self.optimizer.optimize(
                        intraday_data,
                        trade_limit_multiplier=trade_limit_multiplier,
                    )
                    if trade_limit_multiplier > 1 and self.verbose_level >= 1:
                        print(
                            f"With trade limit multiplier {trade_limit_multiplier}, success in optimizer at {time}\n"
                        )
                    break
                except Exception as e:
                    if self.verbose_level >= 1:
                        print(
                            f"With trade limit multiplier {trade_limit_multiplier}, error in optimizer at {time}: {e}"
                        )
            else:
                print(f"No success in optimizer at {time}")
                run_po = False

        # 5. Q and PP after po
        if run_po:
            share_trade_desires = po_result["share_trade_desires"]
            alpha_gradient = po_result["alpha_gradient"]
            cost_gradient = po_result["cost_gradient"]
            risk_gradient = po_result["risk_gradient"]
            alpha_hurdle_raw = po_result["alpha_hurdle_raw"]
            risk_hurdle_raw = po_result["risk_hurdle_raw"]
        else:
            po_result = dict(
                alpha_utility=0,
                risk_utility=0,
                cost_utility=0,
            )
            share_trade_desires = np.zeros_like(alpha)
            alpha_gradient = np.zeros_like(alpha)
            cost_gradient = np.zeros_like(alpha)
            risk_gradient = np.zeros_like(alpha)
            alpha_hurdle_raw = np.zeros_like(share_position)
            risk_hurdle_raw = np.zeros_like(share_position)
            if minute_since_market_start >= self.market_duration_minutes - 10:
                share_trade_desires[0] = -share_position
        self.update_PP_dict_post_po(po_result)
        if "constraint_info" in po_result:
            self.constraint_info_dict[time] = po_result["constraint_info"]

        if self.fundamental_alpha_curve_active:
            adjusted_share_trade_desires = (
                self.get_adjusted_share_trade_desire_from_fundamental_alpha(
                    minute_since_market_start
                )
            )
            adjusted_share_trade_desires.resize(
                len(share_trade_desires), len(self.universe)
            )
            share_trade_desires += adjusted_share_trade_desires

        remaining_time = (
            self.trading_end_time_minute_since_market_start - minute_since_market_start
        )
        share_position = raw_share_position
        target_share_position = share_position + share_trade_desires.cumsum(axis=0)
        share_trade_desire_sign = np.sign(share_trade_desires[0])

        alpha_hurdle = alpha_hurdle_raw * share_trade_desire_sign
        risk_hurdle = risk_hurdle_raw * share_trade_desire_sign
        hurdle = alpha_hurdle + risk_hurdle

        max_hurdle, decay_rate = compute_max_hurdle_and_decay_rate2(
            alpha_hurdle,
            risk_hurdle,
            remaining_time,
            self.max_hurdle,
            self.target_average_time,
        )

        q_data_named = []
        columns = [
            "current_position",
            "arrival_mid",
            "alpha",
            "transact_cost_ratio",
            "target_position",
            "expected_alpha_pnl",
            "trade_desire",
            "alpha_gradient",
            "cost_gradient",
            "risk_gradient",
            "alpha_hurdle",
            "risk_hurdle",
            "max_hurdle",
            "decay_rate",
            "hurdle",
        ]
        zeroes = np.zeros_like(share_position)
        ones = np.ones_like(share_position)
        last_position = share_position
        for i in range(len(self.alpha_horizons)):
            row = {
                "current_position": share_position,
                "arrival_mid": price,
                "alpha": raw_alpha[i],
                "transact_cost_ratio": transact_cost_fraction,
            }
            if i < len(target_share_position):
                expected_alpha_pnl = (
                    (target_share_position[i] + last_position)
                    / 2
                    * dollar_value_per_share
                    * raw_alpha[i]
                )
                last_position = target_share_position[i]
                row.update(
                    {
                        "target_position": last_position,
                        "expected_alpha_pnl": expected_alpha_pnl,
                        "trade_desire": share_trade_desires[i],
                        "alpha_gradient": alpha_gradient[i],
                        "cost_gradient": cost_gradient[i],
                        "risk_gradient": risk_gradient[i],
                        "alpha_hurdle": alpha_hurdle,
                        "risk_hurdle": risk_hurdle,
                        "max_hurdle": max_hurdle,
                        "decay_rate": decay_rate,
                        "hurdle": hurdle,
                    }
                )
            else:
                row.update(
                    {
                        "target_position": last_position,
                        "expected_alpha_pnl": zeroes,
                        "trade_desire": zeroes,
                        "alpha_gradient": zeroes,
                        "cost_gradient": zeroes,
                        "risk_gradient": zeroes,
                        "alpha_hurdle": zeroes,
                        "risk_hurdle": zeroes,
                        "max_hurdle": ones,
                        "decay_rate": ones,
                        "hurdle": zeroes,
                    }
                )
            q_data_named.append(row)

        # append a 1min part if not present
        if self.alpha_horizons[0] == 1:
            modified_alpha_horizons = self.alpha_horizons
        else:
            modified_alpha_horizons = np.append(self.alpha_horizons, 1)
            np.maximum(remaining_time, 1, out=remaining_time)
            np.minimum(remaining_time, self.alpha_horizons[0], out=remaining_time)
            one_minute_share_trade_desire = share_trade_desires[0] / remaining_time
            q_data_named.append(
                {
                    "current_position": share_position,
                    "arrival_mid": price,
                    "alpha": raw_alpha[0] / self.alpha_horizons[0],
                    "transact_cost_ratio": transact_cost_fraction,
                    "target_position": share_position + one_minute_share_trade_desire,
                    "expected_alpha_pnl": (
                        share_position + one_minute_share_trade_desire / 2
                    )
                    * dollar_value_per_share
                    * raw_alpha[0]
                    / self.alpha_horizons[0],
                    "trade_desire": one_minute_share_trade_desire,
                    "alpha_gradient": alpha_gradient[0],
                    "cost_gradient": cost_gradient[0],
                    "risk_gradient": risk_gradient[0],
                    "alpha_hurdle": alpha_hurdle / self.alpha_horizons[0],
                    "risk_hurdle": risk_hurdle / self.alpha_horizons[0],
                    "max_hurdle": max_hurdle,
                    "decay_rate": decay_rate,
                    "hurdle": hurdle / self.alpha_horizons[0],
                }
            )
            q_data_named[0]["trade_desire"] -= one_minute_share_trade_desire
            q_data_named[0]["expected_alpha_pnl"] -= q_data_named[-1][
                "expected_alpha_pnl"
            ]

        q_data = [[row[column] for column in columns] for row in q_data_named]
        q_da = xr.DataArray(
            data=q_data,
            dims=["horizon", "column", "asset"],
            coords={
                "horizon": [MINUTE * i for i in modified_alpha_horizons],
                "column": columns,
                "asset": self.universe,
            },
        ).astype(np.float64)
        self.time_list.append(time)
        self.q_list.append(q_da)

        if self.verbose_level >= 3:
            horizons = [MINUTE * i for i in modified_alpha_horizons]
            logging_horizons = horizons if self.verbose_level >= 4 else horizons[:1]
            print(f"Optimizer output at {time}:")
            print(
                f"current_position_gmv: {np.dot(np.abs(q_da.sel(horizon=horizons[0], column='current_position').values), dollar_value_per_share)}, current_position_nmv: {np.dot(q_da.sel(horizon=horizons[0], column='current_position').values, dollar_value_per_share)}"
            )
            for horizon in logging_horizons:
                print(
                    f"horizon: {horizon}, target_position_gmv: {np.dot(np.abs(q_da.sel(horizon=horizon, column='target_position').values), dollar_value_per_share)}, target_position_nmv: {np.dot(q_da.sel(horizon=horizon, column='target_position').values, dollar_value_per_share)}"
                )
            print()

        return {
            "Q": self.get_last_q(time),
            "PP": self.get_last_pp(time),
        }

    def post_order_process(self, gc_collect=True):
        self.optimizer.post_order_process(gc_collect=gc_collect)

    @staticmethod
    def safe_div(a, b):
        return a / b if b != 0 else 0

    def update_PP_dict_pre_po(self, time):
        p_column_names = [
            "PositionVal",
            "RealizedPnl",
            "UnRealizedPnl",
            "MarkToMarketVal",
            "Pnl",
            "ValOnStreet",
            "NotionalVal",
            "ValBot",
            "ValSold",
            "GMV",
            "SpreadCost",
            "FarTouchCost",
            "ExecCost",
            "OppCost",
        ]
        p_sum = self.P.sel(columns=p_column_names).sum(dim="asset")

        last_pnl = self.pp_dict["Pnl"] if self.pp_dict is not None else 0
        self.pp_dict = {}
        self.pp_dict["timestamp"] = time
        for name in p_column_names:
            self.pp_dict[name] = p_sum.sel(columns=name, drop=True).item()
        self.pp_dict["PnlBps"] = self.safe_div(
            self.pp_dict["Pnl"] * 10000, self.pp_dict["NotionalVal"]
        )
        # TODO: this is duplicated
        self.pp_dict["NMV"] = self.pp_dict["MarkToMarketVal"]

        self.pp_dict["CurrentGMVLimit"] = self.current_gmv_limit
        self.pp_dict["DeltaPnl"] = self.pp_dict["Pnl"] - last_pnl

        position = self.P.sel(columns="MarkToMarketVal", drop=True).values
        cov = self.covariance_data.sel(
            timestamp=time, method="nearest", drop=True
        ).values
        portfolio_std = np.sqrt(position @ cov @ position)
        self.pp_dict["PortfolioVolatility"] = (
            portfolio_std
            / self.total_gmv_limit
            * np.sqrt(252 * self.market_duration_minutes)
            * self.leverage_ratio
        )

        self.turnover = (
            self.pp_dict["ValBot"] - self.pp_dict["ValSold"]
        ) / self.total_gmv_limit
        self.gmv_ratio = self.pp_dict["GMV"] / self.total_gmv_limit
        self.pp_dict["Turnover"] = self.turnover
        self.pp_dict["GMVRatio"] = self.gmv_ratio

    def update_PP_dict_post_po(self, po_result):
        self.pp_dict["AlphaUtility"] = po_result["alpha_utility"]
        self.pp_dict["RiskUtility"] = po_result["risk_utility"]
        self.pp_dict["CostUtility"] = po_result["cost_utility"]
        self.pp_dict["TotalUtility"] = (
            self.pp_dict["AlphaUtility"]
            + self.pp_dict["RiskUtility"]
            + self.pp_dict["CostUtility"]
        )

    def get_last_pp(self, time):
        return self.pp_dict

    def get_last_q_per_asset(self, time, asset):
        da = self.q_list[-1].sel(asset=asset, drop=True)
        d = da_to_joined_dict_faster(da, dims=["horizon", "column"])
        d["timestamp"] = time
        return d

    def get_last_q(self, time):
        q = dict()
        for asset in self.universe:
            q[asset] = self.get_last_q_per_asset(time, asset)
        return q

    def get_q_table(self):
        return xr.concat(self.q_list, dim="timestamp").assign_coords(
            timestamp=self.time_list
        )

    def get_constraint_info(self):
        return self.constraint_info_dict
