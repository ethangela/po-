import numpy as np
import pandas as pd
from brain.portfolio.utils import MINUTE


class TradeDesireInterpolator:
    def __init__(self, po_sim_params):
        self.numerical_config = po_sim_params.optimizer_config["numerical_config"]

        self.data_interface = po_sim_params.data_interface
        self.date = po_sim_params.date
        self.date_time = pd.Timestamp(self.date, tz=po_sim_params.tz)
        self.market_start_time = po_sim_params.market_start_time
        self.all_assets = po_sim_params.universe
        self.n = len(self.all_assets)

        asset_info_data = self.data_interface.get_asof_data(
            "asset_info", self.date
        ).loc[self.all_assets]
        self.trading_end_time_minute_since_market_start = asset_info_data[
            "trading_end_time_minute_since_sod"
        ].values

        self.alpha_horizons = np.cumsum(
            po_sim_params.optimizer_config["alpha_horizons"]
        )

        self.current_timestamp = None
        self.current_trade_desire = None

    def _time_to_minute_since_market_start(self, time):
        return int((time - (self.date_time + self.market_start_time)) / MINUTE)

    def get_ref_index(self, target_horizon):
        index = 1
        while (
            index + 1 < len(self.alpha_horizons)
            and self.alpha_horizons[index] < target_horizon
        ):
            index += 1
        return index - 1, index

    def interpolate_trade_desire(self, timestamp):
        if timestamp == self.current_timestamp:
            return self.current_trade_desire
        self.current_timestamp = timestamp

        minute_since_market_start = self._time_to_minute_since_market_start(timestamp)
        remaining_time = np.maximum(
            1,
            self.trading_end_time_minute_since_market_start - minute_since_market_start,
        )
        target_horizons = np.minimum(
            remaining_time, self.numerical_config["execution_target_horizon"]
        )

        share_position_name = str(MINUTE) + "_current_position"
        target_position_names = [
            f"{str(pd.Timedelta(minutes=horizon))}_target_position"
            for horizon in self.alpha_horizons
        ]
        Q_names = [share_position_name] + target_position_names
        Q = self.data_interface.get_intraday_data("Q", self.date, timestamp, Q_names)

        share_position = Q.sel(columns=share_position_name, drop=True).values
        target_positions = (
            Q.sel(columns=target_position_names).transpose("asset", "columns").values
        )

        if len(self.alpha_horizons) == 1:
            target_horizon = np.minimum(remaining_time, self.alpha_horizons[0])
            target_position = target_positions.squeeze(axis=1)
            self.current_trade_desire = (
                target_position - share_position
            ) / target_horizon
            return self.current_trade_desire

        trade_desires = np.zeros(self.n)
        for i in range(self.n):
            target_horizon = target_horizons[i]
            target_index_1, target_index_2 = self.get_ref_index(target_horizon)
            target_position_1 = target_positions[i, target_index_1]
            target_position_2 = target_positions[i, target_index_2]
            target_horizon_1 = self.alpha_horizons[target_index_1]
            target_horizon_2 = self.alpha_horizons[target_index_2]
            interpolated_target_position = target_position_1 + (
                target_position_2 - target_position_1
            ) * (target_horizon - target_horizon_1) / (
                target_horizon_2 - target_horizon_1
            )
            trade_desires[i] = (
                interpolated_target_position - share_position[i]
            ) / target_horizon
        self.current_trade_desire = trade_desires
        return self.current_trade_desire
