import copy

import numpy as np
import pandas as pd
import xarray as xr

from brain.portfolio.trade_desire_interpolator import TradeDesireInterpolator


class PositionAndPnlTracker:
    def __init__(self, po_sim_params):
        self.date = po_sim_params.date
        self.all_assets = po_sim_params.universe
        self.data_interface = po_sim_params.data_interface
        self.trade_desire_interpolator = TradeDesireInterpolator(po_sim_params)

        # notice in config, this ratio 1 means half spread
        # and now the spreadfrac data is based on full spread
        self.spread_cost_ratio = po_sim_params.realized_spread_cost_ratio / 2
        self.cost_ema_alpha = po_sim_params.cost_ema_alpha

        self.column_names = [
            "timestamp",
            "ArrivalMid",
            "ArrivalFarTouch",
            "CurrentMid",
            "Position",
            "PositionVal",
            "PositionCost",
            "RealizedPnl",
            "UnRealizedPnl",
            "MarkToMarketVal",
            "GMV",
            "PnL",
            "QtyOnStreet",
            "ValOnStreet",
            "PnLPerShare",
            "ShareBot",
            "ShareSold",
            "ValBot",
            "ValSold",
            "AvgPriceBot",
            "AvgPriceSold",
            "NotionalVal",
            "Cashflow",
            "TradeDesire",
            "UnfilledQty",
            "SpreadCost",
            "FarTouchCost",
            "ExecCost",
            "MinuteExecCost",
            "OppCost",
            "MinuteOppCost",
            "CostPerTradeDesire",
        ]

        self.mid_price_close_name = "s1_mid_price_close"
        self.would_price_close_name = "s1_would_price_close"
        self.bid_price_close_name = "s1_bid_price_close"
        self.ask_price_close_name = "s1_ask_price_close"
        self.spread_fraction_name = "s2_spread_pct_mean"
        self.minutebar_names = [
            self.mid_price_close_name,
            self.would_price_close_name,
            self.bid_price_close_name,
            self.ask_price_close_name,
            self.spread_fraction_name,
        ]

        self.fill_qty_name = "fill_qty"
        self.fill_price_name = "fill_price"
        self.street_fill_names = [self.fill_qty_name, self.fill_price_name]

        if "mid" in po_sim_params.ref_price_name:
            self.ref_price_name = self.mid_price_close_name
        elif "wmp" in po_sim_params.ref_price_name:
            self.ref_price_name = self.would_price_close_name
        else:
            raise ValueError(f"Invalid ref price name: {po_sim_params.ref_price_name}")
        print(f"ref_price_name: {self.ref_price_name}")

        asset_info_data = self.data_interface.get_asof_data("asset_info", self.date)
        self.contract_unit = asset_info_data.loc[self.all_assets, "handle_value"]
        self.spread_cost_ratio_multiplier = asset_info_data.loc[
            self.all_assets, "spread_cost_ratio_multiplier"
        ]

        self.p_list = []
        self.p_time_list = []
        self.last_ep_row_dict = {}
        # self.ep_dict = {}
        self.cost_ema_dict = {}
        self.trade_desire_ema_dict = {}
        for asset in self.all_assets:
            self.last_ep_row_dict[asset] = {
                col: np.float64(0) for col in self.column_names
            }
            # self.ep_dict[asset] = list()
            self.cost_ema_dict[asset] = 0
            self.trade_desire_ema_dict[asset] = 0

    def on_trade_desire_cancel(self, time):
        self.__update_p_table(time)
        return copy.deepcopy(self.last_ep_row_dict)

    def __update_p_table(self, time):
        p_data = []
        for asset in self.all_assets:
            p_data.append(self.update_ep_on_trade_desire_cancel(time, asset))
        df_p = pd.DataFrame(p_data)

        self.p_time_list.append(time)
        self.p_list.append(
            xr.DataArray(
                df_p[self.column_names],
                dims=["asset", "column"],
                coords={"asset": self.all_assets, "column": self.column_names},
            )
            .drop_sel(column=["timestamp"])
            .astype(np.float64)
        )

    @staticmethod
    def safe_div(a, b):
        return a / b if b != 0 else 0

    def __refresh_ep_row(self, time, asset):
        # This is place holder for current mid, later we will use the state data
        if self.data_interface.mode == "GCS":
            current_mid_price = (
                self.data_interface.get_intraday_data(
                    "minutebar", self.date, time, [self.ref_price_name]
                )
                .sel(asset=asset, drop=True)
                .item()
            )
        else:
            current_mid_price = self.data_interface.get_intraday_data_per_asset(
                "minutebar", time, asset
            )[self.ref_price_name]

        contract_unit = self.contract_unit.loc[asset].item()
        row = self.last_ep_row_dict[asset]
        row["timestamp"] = time
        row["CurrentMid"] = current_mid_price
        row["MarkToMarketVal"] = row["Position"] * current_mid_price * contract_unit
        row["GMV"] = np.abs(row["MarkToMarketVal"])
        row["PnL"] = row["Cashflow"] + row["MarkToMarketVal"]
        row["PnLPerShare"] = (
            self.safe_div(row["PnL"], (row["ShareBot"] - row["ShareSold"]))
            / contract_unit
        )
        row["UnRealizedPnl"] = (
            row["Position"]
            * (row["CurrentMid"] - row["PositionCost"])
            * contract_unit
        )
        row["QtyOnStreet"] = row["UnfilledQty"]
        row["ValOnStreet"] = row["QtyOnStreet"] * row["ArrivalMid"] * contract_unit

        output = row.copy()
        # self.ep_dict[asset].append(output)
        return output

    def update_ep_on_trade_desire_cancel(self, time, asset):
        if self.data_interface.mode == "GCS":
            mid_price_close = (
                self.data_interface.get_intraday_data(
                    "minutebar", self.date, time, [self.ref_price_name]
                )
                .sel(asset=asset, drop=True)
                .item()
            )
        else:
            mid_price_close = self.data_interface.get_intraday_data_per_asset(
                "minutebar", time, asset
            )[self.ref_price_name]

        contract_unit = self.contract_unit.loc[asset].item()
        row = self.last_ep_row_dict[asset]
        row["ExitMid"] = mid_price_close
        row["MinuteOppCost"] = (
            row["UnfilledQty"]
            * (row["ExitMid"] - row["ArrivalMid"])
            * contract_unit
        )
        row["OppCost"] += row["MinuteOppCost"]

        self.cost_ema_dict[asset] = self.cost_ema_dict[asset] * self.cost_ema_alpha + (
            row["MinuteOppCost"] + row["MinuteExecCost"]
        ) * (1 - self.cost_ema_alpha)
        self.trade_desire_ema_dict[asset] = self.trade_desire_ema_dict[
            asset
        ] * self.cost_ema_alpha + np.abs(row["TradeDesire"]) * (
            1 - self.cost_ema_alpha
        )
        row["CostPerTradeDesire"] = self.safe_div(
            self.cost_ema_dict[asset], self.trade_desire_ema_dict[asset]
        )
        row["UnfilledQty"] = 0
        return self.__refresh_ep_row(time, asset)

    def on_trade_desire_new(self, time, asset):
        trade_desires = self.trade_desire_interpolator.interpolate_trade_desire(time)
        index = self.all_assets.index(asset)
        trade_desire = trade_desires[index]

        if self.data_interface.mode == "GCS":
            minutebar_data = self.data_interface.get_intraday_data(
                "minutebar", self.date, time, self.minutebar_names
            ).sel(asset=asset, drop=True)
            mid_price_close = minutebar_data.sel(
                columns=self.ref_price_name, drop=True
            ).item()
            bid_price_close = minutebar_data.sel(
                columns=self.bid_price_close_name, drop=True
            ).item()
            ask_price_close = minutebar_data.sel(
                columns=self.ask_price_close_name, drop=True
            ).item()
        else:
            minutebar_data = self.data_interface.get_intraday_data_per_asset(
                "minutebar", time, asset
            )
            mid_price_close = minutebar_data[self.ref_price_name]
            bid_price_close = minutebar_data[self.bid_price_close_name]
            ask_price_close = minutebar_data[self.ask_price_close_name]

        row = self.last_ep_row_dict[asset]
        row["ArrivalMid"] = mid_price_close
        row["ArrivalFarTouch"] = (
            bid_price_close if trade_desire < 0 else ask_price_close
        )
        row["ExitMid"] = mid_price_close
        row["TradeDesire"] = trade_desire
        row["UnfilledQty"] = trade_desire
        row["MinuteOppCost"] = 0
        row["MinuteExecCost"] = 0

        return self.__refresh_ep_row(time, asset)

    def __insert_ep_row_on_fill(
        self, time, asset, fill_qty, fill_price, spread_fraction
    ):
        assert fill_qty != 0, "Fill quantity is 0"
        contract_unit = self.contract_unit.loc[asset].item()
        row = self.last_ep_row_dict[asset]
        row["Position"] += fill_qty

        spread_cost = (
            np.abs(fill_qty)
            * fill_price
            * spread_fraction
            * self.spread_cost_ratio
            * self.spread_cost_ratio_multiplier.loc[asset].item()
            * contract_unit
        )
        row["SpreadCost"] += spread_cost

        fill_price = fill_price * (
            1
            + spread_fraction
            * self.spread_cost_ratio
            * self.spread_cost_ratio_multiplier.loc[asset].item()
            * np.sign(fill_qty)
        )

        QtyBot = fill_qty if fill_qty > 0 else 0
        QtySold = fill_qty if fill_qty < 0 else 0
        share_value = fill_price * contract_unit
        row["ShareBot"] += QtyBot
        row["ShareSold"] += QtySold
        row["ValBot"] += QtyBot * share_value
        row["ValSold"] += QtySold * share_value
        row["AvgPriceBot"] = (
            self.safe_div(row["ValBot"], row["ShareBot"]) / contract_unit
        )
        row["AvgPriceSold"] = (
            self.safe_div(row["ValSold"], row["ShareSold"]) / contract_unit
        )
        row["NotionalVal"] += np.abs(fill_qty) * share_value
        row["Cashflow"] -= fill_qty * share_value

        if row["Position"] * fill_qty <= 0:
            row["PositionVal"] += fill_qty * row["PositionCost"] * contract_unit
            row["RealizedPnl"] -= (
                fill_qty * (fill_price - row["PositionCost"]) * contract_unit
            )
        else:
            row["PositionVal"] += fill_qty * fill_price * contract_unit
        row["PositionCost"] = (
            self.safe_div(row["PositionVal"], row["Position"]) / contract_unit
        )

        row["FarTouchCost"] += (
            np.abs(fill_qty)
            * np.abs(row["ArrivalFarTouch"] - row["ArrivalMid"])
            * contract_unit
        )
        exec_cost = fill_qty * (fill_price - row["ArrivalMid"]) * contract_unit
        row["ExecCost"] += exec_cost
        row["MinuteExecCost"] += exec_cost
        row["UnfilledQty"] -= fill_qty

        return self.__refresh_ep_row(time, asset)

    def on_fill(self, time, asset, fill):
        fill_qty = fill[self.fill_qty_name]
        fill_price = fill[self.fill_price_name]

        if fill_qty == 0:
            return []

        # to reconcile two modes better, we probably should record the spread_fraction in on_trade_desire_new
        if self.data_interface.mode == "GCS":
            spread_fraction = (
                self.data_interface.get_intraday_data(
                    "minutebar", self.date, time, [self.spread_fraction_name]
                )
                .sel(asset=asset, drop=True)
                .item()
                / 6
            )
        else:
            spread_fraction = (
                self.data_interface.get_intraday_data_per_asset(
                    "minutebar", time, asset
                )[self.spread_fraction_name]
                / 6
            )

        last_position = self.last_ep_row_dict[asset]["Position"]
        new_position = last_position + fill_qty

        d_list = []
        if new_position * last_position < 0:
            d_list.append(
                self.__insert_ep_row_on_fill(
                    time,
                    asset,
                    -last_position,
                    fill_price,
                    spread_fraction,
                )
            )
            d_list.append(
                self.__insert_ep_row_on_fill(
                    time,
                    asset,
                    new_position,
                    fill_price,
                    spread_fraction,
                )
            )
        else:
            d_list.append(
                self.__insert_ep_row_on_fill(
                    time, asset, fill_qty, fill_price, spread_fraction
                )
            )
        return d_list

    def on_street_fill(self, time, asset):
        street_fill = self.data_interface.get_intraday_data_per_asset(
            "street_fill", time, asset
        )
        return self.on_fill(time, asset, street_fill)

    # def get_ep_table(self):
    #     return {asset: pd.DataFrame(self.ep_dict[asset]) for asset in self.all_assets}

    def get_p_table(self):
        return xr.concat(self.p_list, dim="timestamp").assign_coords(
            timestamp=self.p_time_list
        )
