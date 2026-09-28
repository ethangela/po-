import pandas as pd

from brain.utils import Utils
from brain.portfolio.fill_model import get_fill_model
from brain.portfolio.external_fill_driver import SimExternalFillDriver
from brain.portfolio.po_executor import POExecutor
from brain.portfolio.position_and_pnl_tracker import PositionAndPnlTracker


class SimDriver:
    def __init__(self, po_sim_params):
        print("Initializing SimDriver")
        self.data_interface = po_sim_params.data_interface
        self.universe = po_sim_params.universe

        self.date = po_sim_params.date
        self.tz = po_sim_params.tz
        self.start_time = po_sim_params.start_time
        self.end_time = po_sim_params.end_time
        self.time_gap = po_sim_params.time_gap

        self.external_fill_only = po_sim_params.external_fill_only
        self.fill_model = get_fill_model(po_sim_params)
        self.po_executor = POExecutor(po_sim_params)
        self.position_and_pnl_tracker = PositionAndPnlTracker(po_sim_params)
        self.external_fill_driver = None
        if po_sim_params.per_product_external_fill_dim:
            print(
                f"Loading per-product external fill data from "
                f"{po_sim_params.per_product_external_fill_dim} for {self.date}"
            )
            self.external_fill_driver = SimExternalFillDriver(po_sim_params)
        elif Utils.path_exists(
            self.data_interface.paths["external_fill_dir"] + f"/{self.date}.pkl"
        ):
            print(f"Loading external fill data for {self.date}")
            self.external_fill_driver = SimExternalFillDriver(po_sim_params)

        assert (
            self.external_fill_driver or not self.external_fill_only
        ), "external_fill_only is True but no external fill data found"

        self.produce_ep = po_sim_params.produce_ep
        self.produce_constraint_info = po_sim_params.produce_constraint_info

    def on_street_fill(self, time, asset, fill):
        self.data_interface.put("street_fill", asset, fill)
        output = self.position_and_pnl_tracker.on_street_fill(time, asset)
        if self.produce_ep:
            for ep_row in output:
                self.data_interface.put("EPI", asset, ep_row)

    def run_snapshot(self, time, eod=False):
        if self.external_fill_driver:
            street_fill = self.external_fill_driver.get_fills(time)
            for asset in self.universe:
                for row in street_fill[asset]:
                    self.on_street_fill(time, asset, row)

        # people should eventually cleanup the side effect of the fill model
        output = self.fill_model.get_fills(time, eod)
        if not self.external_fill_only:
            street_fill = output["street_fill"]
            # notice here the ordering is first asset then timestamp, fine for sim for now
            for asset in self.universe:
                for row in street_fill[asset]:
                    timestamp = row["timestamp"]
                    self.on_street_fill(timestamp, asset, row)

        # logics after this follows the minute schedule
        output = self.position_and_pnl_tracker.on_trade_desire_cancel(time)
        for asset in self.universe:
            self.data_interface.put("P", asset, output[asset])
            if self.produce_ep:
                self.data_interface.put("EPI", asset, output[asset])

        output = self.po_executor.run_snapshot(time, not eod)
        Q = output["Q"]
        PP = output["PP"]
        for asset in self.universe:
            self.data_interface.put("Q", asset, Q[asset])
        self.data_interface.put("PP", self.universe[0], PP)

        self.fill_model.set_share_trade_desire(time)

        for asset in self.universe:
            output = self.position_and_pnl_tracker.on_trade_desire_new(time, asset)
            if self.produce_ep:
                self.data_interface.put("EPI", asset, output)

        self.po_executor.post_order_process(gc_collect=False)
        # return self.data_interface.get_output(time)
        return

    def run(self):
        print("Start running PO simulation")
        time = self.start_time
        while time < self.end_time:
            self.run_snapshot(pd.Timestamp(self.date, tz=self.tz) + time)
            time += self.time_gap
        end_time = pd.Timestamp(self.date, tz=self.tz) + self.end_time
        self.run_snapshot(end_time, eod=True)

        # Save execution model checkpoint at end of day (if supported)
        self.fill_model.on_end_of_day()
        # this is for run_po_sim.py
        output = self.data_interface.get_output_table()
        if self.produce_constraint_info:
            output["constraint_info"] = self.po_executor.get_constraint_info()
        return output
