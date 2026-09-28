import numpy as np


class ExecutionOptimizer:
    def __init__(self, po_sim_params, asset, mktstats_timebin_data):
        self.date = po_sim_params.date
        self.alpha_horizons = po_sim_params.execution_optimizer_config["alpha_horizons"]
        self.H = self.alpha_horizons[-1]
        self.h_list = np.ones(self.H)
        self.gamma = po_sim_params.execution_optimizer_config["gamma"]
        # when we have better impact models, we could compute it per asset
        # and put into the asset_info data
        # now only try some similar logic as quadratic transact cost, with a global multiplier
        self.eta = po_sim_params.execution_optimizer_config["eta_multiplier"]

        self.decay = 1

        self.data_interface = po_sim_params.data_interface
        self.load_data(asset, mktstats_timebin_data)
        self.compute_helper_matrix()

    def load_data(self, asset, mktstats_timebin_data):
        cov_data = self.data_interface.get_asof_data("cov", self.date)
        self.cov_per_second = (
            cov_data.sel(asset1=asset, asset2=asset, drop=True).mean(dim="time").item()
            / 60
        )

        price_name = "timebin_mean_s1_mid_price_mean"
        vlm_name = "timebin_mean_s3_share_volume_total"
        price = (
            mktstats_timebin_data.sel(asset=asset, columns=price_name, drop=True)
            .mean(dim="timestamp")
            .item()
        )
        vlm = (
            mktstats_timebin_data.sel(asset=asset, columns=vlm_name, drop=True)
            .mean(dim="timestamp")
            .item()
        )
        contract_unit = (
            self.data_interface.get_asof_data("asset_info", self.date)
            .loc[asset]["handle_value"]
            .item()
        )
        dollar_value_per_share = price * contract_unit

        # (TODO) potentially make PO quadratic cost like this
        self.eta *= (
            1 / dollar_value_per_share / (vlm / 60) * np.sqrt(self.cov_per_second)
        )

        self.gamma *= dollar_value_per_share
        self.eta *= dollar_value_per_share

    def compute_helper_matrix(self):
        # the gradient will be computed as A @ x = b
        # b will be b_helper_hh @ alpha + b_helper_h * target_trade_desire
        self.A_hh = np.zeros((self.H, self.H))
        self.b_helper_hh = np.zeros((self.H, self.H))
        self.b_helper_h = np.zeros(self.H)

        risk_coeff = self.cov_per_second * self.gamma
        decay_to_the_i = 1
        for i in range(self.H):
            self.A_hh[i, i] = (
                2 * self.eta / self.h_list[i] + risk_coeff * self.h_list[i] * 2 / 3
            )
            self.b_helper_hh[i, i] = 1 / 2 * decay_to_the_i
            if i == 0:
                self.b_helper_h[i] = -(
                    -2 * self.eta / self.h_list[i] + risk_coeff * self.h_list[i] / 3
                )
            else:
                self.A_hh[i, i - 1] = (
                    -2 * self.eta / self.h_list[i] + risk_coeff * self.h_list[i] / 3
                )
            if i != self.H - 1:
                decay_to_the_i *= self.decay
                self.A_hh[i, i] += (
                    2 * self.eta / self.h_list[i + 1]
                    + risk_coeff * self.h_list[i + 1] * 2 / 3
                )
                self.A_hh[i, i + 1] = (
                    -2 * self.eta / self.h_list[i + 1]
                    + risk_coeff * self.h_list[i + 1] / 3
                )
                self.b_helper_hh[i, i + 1] = 1 / 2 * decay_to_the_i
        self.A_hh[self.H - 1, self.H - 1] += risk_coeff * 60 * 10000

        self.A_inv_hh = np.linalg.inv(self.A_hh)
        self.A_inv_hh_b_helper_hh = self.A_inv_hh @ self.b_helper_hh
        self.A_inv_hh_b_helper_h = self.A_inv_hh @ self.b_helper_h

    def compute_interpolated_execution_alpha(self, execution_alpha):
        execution_alpha_h = np.zeros(self.H)
        last_alpha_horizon = 0
        last_alpha = 0
        for current_alpha_horizon, current_alpha in zip(
            self.alpha_horizons, execution_alpha
        ):
            slope = (current_alpha - last_alpha) / (
                current_alpha_horizon - last_alpha_horizon
            )
            execution_alpha_h[last_alpha_horizon:current_alpha_horizon] = slope
            last_alpha_horizon = current_alpha_horizon
            last_alpha = current_alpha
        return execution_alpha_h

    def get_execution_plan(
        self,
        seconds_since_target_set_time,
        target_trade_desire,
        hurdle_alpha,
        execution_alpha,
    ):
        hurdle_alpha_h = np.zeros(self.H)
        hurdle_alpha_h[: 60 - seconds_since_target_set_time] = hurdle_alpha / 60
        execution_alpha_h = self.compute_interpolated_execution_alpha(execution_alpha)
        # (TODO) for now, make hurdle weaker
        alpha_h = hurdle_alpha_h / 10 + execution_alpha_h
        """
        b_h = self.b_helper_hh @ alpha_h + self.b_helper_h * target_trade_desire
        q_h = self.A_inv_hh @ b_h
        """
        q_h = self.A_inv_hh_b_helper_hh @ alpha_h + self.A_inv_hh_b_helper_h * (
            -target_trade_desire
        )
        q_h = np.clip(
            q_h, -np.abs(target_trade_desire), np.abs(target_trade_desire) * 0.9
        )
        return q_h + target_trade_desire
