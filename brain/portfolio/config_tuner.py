import numpy as np
import xarray as xr


class ConfigTuner:
    def __init__(self, po_sim_params):
        self.data_interface = po_sim_params.data_interface
        self.numerical_config = po_sim_params.optimizer_config["numerical_config"]

        self.verbose_level = po_sim_params.verbose_level
        self.universe = po_sim_params.universe
        self.date = po_sim_params.date
        self.covariance_data = self.data_interface.get_asof_data(
            "timebin_cov", self.date
        ).sel(asset1=self.universe, asset2=self.universe)

        self.dynamic_gamma_vol_adapt = self.numerical_config["dynamic_gamma_vol_adapt"]
        # SOD anchor + running day-variance accumulators for the vol adaptation.
        self.sod_gamma = self.numerical_config["gamma"]
        self._dyn_intraday_dayvar = 0.0
        self._dyn_timebin_dayvar = 0.0
        self._dyn_intraday_diag = None

    def _ensure_intraday_diag(self):
        if self._dyn_intraday_diag is not None:
            return
        cov = self.data_interface.get_asof_data("intraday_cov", self.date)
        diag = xr.concat(
            [cov.sel(asset=a, columns=a, drop=True) for a in self.universe],
            dim="asset",
        ).assign_coords(asset=self.universe)
        self._dyn_intraday_diag = diag.transpose("timestamp", "asset")

    def _update_dynamic_gamma(self, time, share_position, dollar_value_per_share):
        # Bounded on both sides (>3x gamma) to (<0.5x gamma).
        self._ensure_intraday_diag()
        p_dollar = share_position * dollar_value_per_share
        if not np.any(p_dollar):
            return
        intr_diag = self._dyn_intraday_diag.sel(
            timestamp=time, method="nearest", drop=True
        ).values
        intraday_var = float(np.nansum((p_dollar**2) * intr_diag))
        cov_t = self.covariance_data.sel(
            timestamp=time, method="nearest", drop=True
        ).values
        timebin_var = float(np.nansum((p_dollar**2) * np.diag(cov_t)))
        if intraday_var <= 0.0 or timebin_var <= 0.0:
            return
        self._dyn_intraday_dayvar += intraday_var
        self._dyn_timebin_dayvar += timebin_var
        if self._dyn_timebin_dayvar <= 0.0:
            return
        vol_ratio = self._dyn_intraday_dayvar / self._dyn_timebin_dayvar
        gamma_multiplier = float(np.clip(vol_ratio, 0.5, 3.0))
        new_gamma = self.sod_gamma * gamma_multiplier
        self.numerical_config["gamma"] = new_gamma
        # TODO(dynamic-gamma -> spirit): the companion horizon adaptation below only reaches
        # the toy interpolator fill models, not the live spirit executor (whose target_horizon
        # is fixed at construction). To drive spirit it must be published through the po_sim Q
        # dim (or another dim the sequencer forwards to spirit) and read per update in the
        # executor. Commented out until we see value from it.
        # new_horizon = float(np.clip(120.0 / gamma_multiplier, 5.0, 120.0))
        # self.numerical_config["target_horizon"] = new_horizon
        # self.numerical_config["execution_target_horizon"] = new_horizon
        if self.verbose_level >= 1:
            print(
                f"#DynGamma {time} vol_ratio={vol_ratio:.3f} mult={gamma_multiplier:.3f}"
                f" gamma={new_gamma:.3e}"
            )

    def update(self, time, in_range, pp_dict, share_position, dollar_value_per_share):
        if self.dynamic_gamma_vol_adapt and in_range:
            self._update_dynamic_gamma(time, share_position, dollar_value_per_share)

        if (
            not self.numerical_config["liquidation_mode"]
            and pp_dict["PnL"] < self.numerical_config["liquidation_entrance_threshold"]
        ):
            self.numerical_config["liquidation_mode"] = True
            self.numerical_config["liquidation_remaining_time"] = np.inf
            print(f"POExecutor entering liquidation mode at {time}")
        if (
            self.numerical_config["liquidation_mode"]
            and pp_dict["PnL"] > self.numerical_config["liquidation_exit_threshold"]
        ):
            self.numerical_config["liquidation_mode"] = False
