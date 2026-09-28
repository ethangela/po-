import datetime
import numpy as np
import pandas as pd
import xarray as xr
from collections import deque
from dataclasses import asdict, dataclass
from brain.portfolio.trade_desire_interpolator import TradeDesireInterpolator


@dataclass(frozen=True)
class StreetFillRow:
    timestamp: datetime.datetime | pd.Timestamp
    fill_qty: float
    fill_price: float
    account: str
    contra_account: str = "sim"
    is_manual: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def get_fill_model(po_sim_params):
    print(f"fill_model: {po_sim_params.fill_model}")
    if po_sim_params.fill_model == "fill_all_at_mid":
        return FillModelFillAllAtMid(po_sim_params)
    elif po_sim_params.fill_model in ("aggressive", "fill_all_at_aggressive"):
        return FillModelFillAllAtAggressive(po_sim_params)
    elif po_sim_params.fill_model == "fill_all_at_wmp":
        return FillModelFillAllAtWMP(po_sim_params)
    elif po_sim_params.fill_model == "rabbit":
        return FillModelRabbit(po_sim_params)
    elif po_sim_params.fill_model == "at_mid_adj":
        return FillModelAtMidAdj(po_sim_params)
    elif po_sim_params.fill_model == "execution_algo":
        from brain.apps.batch.execution_service.algo import FillModelExecutionAlgo

        return FillModelExecutionAlgo(po_sim_params)
    elif po_sim_params.fill_model == "algo_container_spirit":
        from brain.apps.batch.execution_service.algo_container_spirit import (
            FillModelAlgoContainerSpirit,
        )

        return FillModelAlgoContainerSpirit(po_sim_params)
    else:
        raise ValueError(f"Invalid fill model: {po_sim_params.fill_model}")


class FillModel:
    def __init__(self, po_sim_params):
        self.all_assets = po_sim_params.universe
        _cm = po_sim_params.config_manager
        self.execution_universe = (
            set(_cm.get_list(_cm.section, "execution_universe"))
            if str(_cm.get_with_default(_cm.section, "execution_universe", "")).strip()
            else set(self.all_assets)
        )
        self.liquidate_at_eod = po_sim_params.optimizer_config["numerical_config"][
            "liquidate_at_eod"
        ]

        self.date = po_sim_params.date

        self.data_interface = po_sim_params.data_interface
        self.portfolio_name = po_sim_params.config_manager.get_sequencer_name()

        self.share_position = np.zeros(len(self.all_assets))
        self.share_trade_desire = np.zeros(len(self.all_assets))
        self.risk_gradient = np.zeros(len(self.all_assets))
        self.alpha_gradient = np.zeros(len(self.all_assets))
        self.hurdle = np.zeros(len(self.all_assets))
        self.alpha = np.zeros(len(self.all_assets))
        self.target_set_time = None
        # TODO: Apply this to other fill models too
        self.fill_model_delay = datetime.timedelta(
            seconds=_cm.get_with_default(
                _cm.section, "fill_model_delay_s", 0, type_converter=float
            )
        )

        self.time_list = deque(maxlen=3)
        self.minutebar_list = deque(maxlen=3)

        self.trade_desire_interpolator = TradeDesireInterpolator(po_sim_params)

        # FillModels using these indirectly through those get_* methods won't be supported in aio
        # if self.data_interface.mode == "GCS":
        #     state_names = [
        #         "s1_mid_price",
        #         "s1_bid_price",
        #         "s1_ask_price",
        #         "s1_bid_size",
        #         "s1_ask_size",
        #         "s2_mid_price",
        #         "s2_would_price",
        #     ]
        #     hurdle_alpha_names = [f"v_{horizon}" for horizon in [10, 30, 60]]
        #     self.state_data = self.data_interface.get_asof_data(
        #         "state", self.date, state_names
        #     )
        #     self.hurdle_alpha_data = self.data_interface.get_asof_data(
        #         "hurdle_alpha", self.date, hurdle_alpha_names
        #     )

    def on_end_of_day(self):
        pass

    def update_minutebar_data(self, time):
        minutebar_names = [
            "s1_mid_price_open",
            "s1_mid_price_close",
            "s1_mid_price_high",
            "s1_mid_price_low",
            "s1_mid_price_mean",
            "s1_bid_price_open",
            "s1_bid_price_close",
            "s1_bid_price_high",
            "s1_bid_price_low",
            "s1_bid_price_mean",
            "s1_ask_price_open",
            "s1_ask_price_close",
            "s1_ask_price_high",
            "s1_ask_price_low",
            "s1_ask_price_mean",
            "s3_overall_twap_mean",
            "s1_bid_size_mean",
            "s1_ask_size_mean",
            "s1_half_spread_frac_mean",
            "s3_share_volume_total",
            "s1_would_price_close",
        ]
        self.time_list.append(time)
        self.minutebar_list.append(
            self.data_interface.get_intraday_data(
                "minutebar", self.date, time, minutebar_names
            )
        )
        self.minutebar_data = xr.concat(
            list(self.minutebar_list), dim="timestamp"
        ).assign_coords(timestamp=list(self.time_list))

    def clip_qty(self, limit, desired_qty, truncate=True):
        fill_qty = np.clip(desired_qty, -limit, limit)
        if truncate:
            fill_qty = np.trunc(fill_qty)
        return fill_qty

    def set_share_trade_desire(self, timestamp):
        horizon = pd.Timedelta(minutes=1)
        prefix = str(horizon) + "_"
        share_position_name = prefix + "current_position"
        share_trade_desire_name = prefix + "trade_desire"
        risk_gradient_name = prefix + "risk_gradient"
        alpha_gradient_name = prefix + "alpha_gradient"
        cost_gradient_name = prefix + "cost_gradient"
        hurdle_name = prefix + "hurdle"
        alpha_name = prefix + "alpha"
        risk_hurdle_name = prefix + "risk_hurdle"
        alpha_hurdle_name = prefix + "alpha_hurdle"
        max_hurdle_name = prefix + "max_hurdle"
        decay_rate_name = prefix + "decay_rate"

        Q_names = [
            share_position_name,
            share_trade_desire_name,
            risk_gradient_name,
            alpha_gradient_name,
            cost_gradient_name,
            hurdle_name,
            alpha_name,
            risk_hurdle_name,
            alpha_hurdle_name,
            max_hurdle_name,
            decay_rate_name,
        ]
        Q = self.data_interface.get_intraday_data("Q", self.date, timestamp, Q_names)
        self.target_set_time = timestamp
        self.share_position = Q.sel(columns=share_position_name, drop=True).values

        self.share_trade_desire = (
            self.trade_desire_interpolator.interpolate_trade_desire(timestamp)
        )

        self.risk_gradient = Q.sel(columns=risk_gradient_name, drop=True)
        self.alpha_gradient = Q.sel(columns=alpha_gradient_name, drop=True)
        self.cost_gradient = Q.sel(columns=cost_gradient_name, drop=True)
        self.hurdle = Q.sel(columns=hurdle_name, drop=True)
        self.alpha = Q.sel(columns=alpha_name, drop=True)
        self.risk_hurdle = Q.sel(columns=risk_hurdle_name, drop=True)
        self.alpha_hurdle = Q.sel(columns=alpha_hurdle_name, drop=True)
        self.max_hurdle = Q.sel(columns=max_hurdle_name, drop=True)
        self.decay_rate = Q.sel(columns=decay_rate_name, drop=True)
        if not self.time_list:
            self.update_minutebar_data(timestamp)

    def _build_street_fill_row(self, timestamp, fill_qty, fill_price):
        return StreetFillRow(
            timestamp=timestamp,
            fill_qty=fill_qty,
            fill_price=fill_price,
            account=self.portfolio_name,
        ).to_dict()

    # TODO: see if we can extend the sim driver to still optimize at the top of the minute
    # but we can assume that the fill models only see the targets when they get to delay
    @property
    def execution_start_time(self):
        return self.target_set_time + self.fill_model_delay

    def on_get_fill_start(self, target_set_time, timestamp):
        pass

    def on_get_fill_end(self, target_set_time, timestamp):
        pass

    def get_fills(self, timestamp: datetime.datetime, eod):
        # time stamp is the get fill time, for example, target set at 9:30, fill at 9:31, then timestamp is 9:31
        # md_timestamp is 9:31 representing period of 9:30-9:31
        if self.target_set_time is None or timestamp < self.target_set_time:
            street_fill = {}
            for asset in self.all_assets:
                street_fill[asset] = [self._build_street_fill_row(timestamp, 0, 0)]
            return {"street_fill": street_fill}

        self.update_minutebar_data(timestamp)

        current_pos = self.share_position
        # get the mkt data for the period between target set time and fill time
        interval_mkt_data = self.minutebar_data.sel(
            timestamp=slice(self.target_set_time, timestamp)
        )
        street_fill = {}
        self.on_get_fill_start(self.target_set_time, timestamp)
        for i, asset in enumerate(self.all_assets):
            if asset not in self.execution_universe:
                street_fill[asset] = [self._build_street_fill_row(timestamp, 0, 0)]
                continue
            mds = interval_mkt_data.sel(asset=asset).to_pandas()
            if eod and self.liquidate_at_eod:
                result = self.eod_fill(
                    timestamp,
                    asset,
                    -current_pos[i],
                    mds,
                )
            else:
                result = self.get_asset_fill(
                    timestamp,
                    asset,
                    self.share_trade_desire[i],
                    mds,
                )

            if isinstance(result, list):
                street_fill[asset] = result
            else:
                fill_qty, fill_price = result
                street_fill[asset] = [
                    self._build_street_fill_row(timestamp, fill_qty, fill_price)
                ]
        self.on_get_fill_end(self.target_set_time, timestamp)
        return {"street_fill": street_fill}

    def eod_fill(self, timestamp, asset, share_trade_desire, mds):
        return share_trade_desire, mds.iloc[0]["s1_mid_price_close"]

    def get_asset_fill(self, timestamp, asset, share_trade_desire, mds):
        raise NotImplementedError("get_asset_fill is not implemented")

    def aggressive_fill(self, timestamp, asset, share_trade_desire, md):
        side = "buy" if share_trade_desire > 0 else "sell"
        if side == "buy":
            fill_price = md["s1_ask_price_open"]
            fill_qty = self.clip_qty(md["s1_ask_size_mean"], share_trade_desire)
        else:
            fill_price = md["s1_bid_price_open"]
            fill_qty = self.clip_qty(md["s1_bid_size_mean"], share_trade_desire)
        return fill_qty, fill_price, share_trade_desire - fill_qty

    # def get_market_snapshot(self, time, asset):
    #     """Batch-fetch all market data columns in a single .sel() call."""
    #     cols = [
    #         "s1_ask_price",
    #         "s1_bid_price",
    #         "s1_mid_price",
    #         "s2_mid_price",
    #         "s2_would_price",
    #         "s1_ask_size",
    #         "s1_bid_size",
    #     ]
    #     vals = self.state_data.sel(timestamp=time, asset=asset, columns=cols).values
    #     return dict(zip(cols, vals))


class FillModelAtMidAdj(FillModel):
    def __init__(self, po_sim_params):
        super().__init__(po_sim_params)
        self.po_sim_params = po_sim_params

    def get_asset_fill(self, end_time, asset, share_trade_desire, mds):
        if share_trade_desire == 0:
            return 0.0, 0.0

        max_hurdle = self.max_hurdle.sel(asset=asset, drop=True).item()
        base_decay_rate = self.decay_rate.sel(asset=asset, drop=True).item()
        coeff = max(0.0, 1.0 - max_hurdle / base_decay_rate)
        share_trade_desire = share_trade_desire * coeff
        return share_trade_desire, mds.iloc[0]["s1_mid_price_close"]  # without trunc


class FillModelFillAllAtMid(FillModel):
    def __init__(self, po_sim_params):
        super().__init__(po_sim_params)

    def get_asset_fill(self, timestamp, asset, share_trade_desire, mds):
        return share_trade_desire, mds.iloc[0]["s1_mid_price_close"]  # without trunc


class FillModelFillAllAtAggressive(FillModel):
    def __init__(self, po_sim_params):
        super().__init__(po_sim_params)

    def get_asset_fill(self, timestamp, asset, share_trade_desire, mds):
        fill_qty, fill_price, _ = self.aggressive_fill(
            timestamp, asset, share_trade_desire, mds.iloc[-1]
        )
        return fill_qty, fill_price


class FillModelFillAllAtWMP(FillModel):
    def __init__(self, po_sim_params):
        super().__init__(po_sim_params)

    def get_asset_fill(self, timestamp, asset, share_trade_desire, mds):
        return share_trade_desire, mds.iloc[0]["s1_would_price_close"]  # without trunc


class FillModelRabbit(FillModel):
    def __init__(self, po_sim_params):
        super().__init__(po_sim_params)

    def get_asset_fill(self, timestamp, asset, share_trade_desire, mds):
        return np.trunc(share_trade_desire), mds.iloc[0]["s1_mid_price_close"]
