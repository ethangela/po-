import gc
import pathlib
import logging as logger
import time as timer

import numpy as np
from scipy.linalg import block_diag, cholesky, pinv
from scipy.sparse import bsr_matrix

import mosek
from mosek.fusion import (
    AccSolutionStatus,
    Domain,
    Expr,
    Matrix,
    Model,
    ObjectiveSense,
    SolutionStatus,
)
import mosek.fusion.pythonic


# Translate the raw data into the final format for the numerical solver
# Also organize some high level logics here
class Optimizer:
    def __init__(self, optimizer_config: dict):
        self.numerical_config = optimizer_config["numerical_config"]
        self.numerical_solver = MosekOptimizer(self.numerical_config)
        self.n = len(optimizer_config["all_assets"])
        self.components_to_remove = optimizer_config["components_to_remove"]
        self.checkpoint_minute_since_market_start = optimizer_config[
            "checkpoint_minute_since_market_start"
        ]
        self.alpha_horizons = optimizer_config["alpha_horizons"]

    def set_log_file(self, log_file):
        self.numerical_solver.set_log_file(log_file)

    def setup_start_of_day_as_of(
        self,
        sod_data: dict,
        trading_end_time_minute_since_market_start: np.ndarray,
        vlm_data: np.ndarray,
    ):
        self.return_cov = sod_data["return_cov"]
        self.orthogonal_factor_exposures = sod_data["orthogonal_factor_exposures"]
        self.trading_end_time_minute_since_market_start = (
            trading_end_time_minute_since_market_start
        )
        self.vlm_data = vlm_data

        for t in range(len(self.return_cov)):
            self.return_cov[t] = self._adjust_risk_matrix(self.return_cov[t], t)

    def _adjust_risk_matrix(self, return_cov, minute_since_market_start):
        overnight_minutes = 60 * 4
        minute_to_trading_end_time = (
            self.trading_end_time_minute_since_market_start - minute_since_market_start
        )
        minute_to_trading_end_time[minute_to_trading_end_time < 10] = 10
        sqrt_multiplier = np.sqrt(
            (minute_to_trading_end_time + overnight_minutes)
            / minute_to_trading_end_time
        ).reshape(1, -1)
        multiplier = sqrt_multiplier.T @ sqrt_multiplier
        return multiplier * (return_cov + np.eye(self.n) * 1e-12)

    def _compute_h_list(self, minute_since_market_start: int):
        eod = self.checkpoint_minute_since_market_start[-1]
        self.h_list = []
        self.vlm_list = []
        for h in self.alpha_horizons:
            if minute_since_market_start + h < eod:
                self.h_list.append(h)
                self.vlm_list.append(
                    np.sum(
                        self.vlm_data[
                            minute_since_market_start : minute_since_market_start + h
                        ],
                        axis=0,
                    )
                )
                minute_since_market_start += h
            else:
                self.h_list.append(eod - minute_since_market_start)
                self.vlm_list.append(
                    np.sum(self.vlm_data[minute_since_market_start:eod], axis=0)
                )
                minute_since_market_start = eod
                break
        if self.numerical_config["additional_horizon_from_checkpoint"]:
            for t in self.checkpoint_minute_since_market_start:
                if minute_since_market_start >= t:
                    continue
                self.h_list.append(t - minute_since_market_start)
                self.vlm_list.append(
                    np.sum(self.vlm_data[minute_since_market_start:t], axis=0)
                )
                minute_since_market_start = t
        self.H = len(self.h_list)

    def _compute_return_cov_list(self, minute_since_market_start: int):
        return_cov_list = []
        minutes_since_market_start = np.cumsum(
            [minute_since_market_start] + self.h_list
        )
        for s, e in zip(
            minutes_since_market_start[:-1], minutes_since_market_start[1:]
        ):
            return_cov_list.append(self.return_cov[s:e].mean(axis=0))
        return return_cov_list

    def optimize(self, intraday_data: dict, trade_limit_multiplier: float = 1):
        self.numerical_solver.update_numerical_config()

        minute_since_market_start = intraday_data["minute_since_market_start"]
        self._compute_h_list(minute_since_market_start)

        return_cov = self._compute_return_cov_list(minute_since_market_start)
        orthogonal_factor_exposures = [self.orthogonal_factor_exposures] * self.H

        components_to_remove = [
            component * intraday_data["dollar_value_per_share"]
            for component in self.components_to_remove
        ]
        self.numerical_solver.set_basic_model(
            self.n,
            self.h_list,
            return_cov,
            components_to_remove,
        )

        dollar_position = (
            intraday_data["share_position"] * intraday_data["dollar_value_per_share"]
        )
        self.numerical_solver.set_init_position(dollar_position)

        # position and trade limits
        position_limit = np.asarray(intraday_data["position_limit"])
        gmv_limit = np.asarray(intraday_data["gmv_limit"])
        nmv_limit = np.asarray(intraday_data["nmv_limit"])
        num_rows_to_add = self.H - len(position_limit)
        if num_rows_to_add > 0:
            position_limit = np.vstack(
                [
                    position_limit,
                    np.repeat(position_limit[-1:, :], num_rows_to_add, axis=0),
                ]
            )
            gmv_limit = np.concatenate([gmv_limit, [gmv_limit[-1]] * num_rows_to_add])
            nmv_limit = np.concatenate([nmv_limit, [nmv_limit[-1]] * num_rows_to_add])
        else:
            position_limit = position_limit[: self.H, :]
            gmv_limit = gmv_limit[: self.H]
            nmv_limit = nmv_limit[: self.H]

        # share the same retry multiplier for now, so it doesn't spam too much retries
        portfolio_volatility_limit = (
            intraday_data["portfolio_volatility_limit"] * trade_limit_multiplier
        )
        if portfolio_volatility_limit < np.inf:
            self.numerical_solver.set_portfolio_volatility_limit(
                portfolio_volatility_limit
            )

        if (
            self.numerical_config["enable_vlm_limit"]
            and trade_limit_multiplier < np.inf
        ):
            if intraday_data["turnover_limit"] < np.inf:
                total_vlm_limit = (
                    intraday_data["turnover_limit"]
                    * gmv_limit[0]
                    * trade_limit_multiplier
                )
                vlm_limit_h = [
                    np.dot(vlms, intraday_data["dollar_value_per_share"])
                    for vlms in self.vlm_list
                ]
                self.numerical_solver.add_total_vlm_limit(total_vlm_limit, vlm_limit_h)

            if "intraday_vlm_limit" in intraday_data:
                self.vlm_list[0] = np.maximum(
                    self.vlm_list[0] / 5,
                    intraday_data["intraday_vlm_limit"] * self.h_list[0],
                )

        trade_limit = (
            np.asarray(self.vlm_list) * intraday_data["dollar_value_per_share"]
        )
        trade_limit_1m = self.numerical_config["trade_limit_1m"]
        np.minimum(
            trade_limit,
            trade_limit_1m * np.asarray(self.h_list)[:, None],
            out=trade_limit,
        )
        trade_limit *= trade_limit_multiplier

        for n_index in range(self.n):
            remaining_time = (
                self.trading_end_time_minute_since_market_start[n_index]
                - minute_since_market_start
            )
            if remaining_time < 10:
                np.maximum(
                    trade_limit[:, n_index],
                    np.abs(dollar_position[n_index]) * 2,
                    out=trade_limit[:, n_index],
                )
        self.numerical_solver.add_trade_limit(trade_limit.reshape(-1))

        upper_position_limit = position_limit
        lower_position_limit = -position_limit
        self.numerical_solver.add_position_limit(
            upper_position_limit.reshape(-1), lower_position_limit.reshape(-1)
        )

        self.numerical_solver.add_gmv_limit(gmv_limit)

        self.numerical_solver.add_nmv_limit(nmv_limit)

        tc_cost_multiplier = np.ones((self.H, self.n))
        for n_index in range(self.n):
            remaining_time = (
                self.trading_end_time_minute_since_market_start[n_index]
                - minute_since_market_start
            )
            for h_index in range(self.H + 1):
                if h_index < self.H:
                    remaining_time -= self.h_list[h_index]
                if self.numerical_config["liquidate_at_eod"]:
                    if h_index == 0:
                        continue
                    if remaining_time < 0 or (
                        h_index == self.H and remaining_time == 0
                    ):
                        self.numerical_solver.add_customized_position_constraint(
                            n_index,
                            h_index - 1,
                            set_value_flag=True,
                            value_to_set=0,
                        )
                        for h_index_2 in range(h_index, self.H):
                            self.numerical_solver.add_customized_trade_constraint(
                                n_index,
                                h_index_2,
                                set_value_flag=True,
                                value_to_set=0,
                            )
                        break
                else:
                    if remaining_time < 0:
                        intraday_data["alpha"][h_index:, n_index] = 0
                        tc_cost_multiplier[h_index:, n_index] *= 100
                        break

        if self.numerical_config["liquidation_mode"]:
            for n_index in range(self.n):
                cum_h = 0
                for h_index in range(self.H):
                    cum_h += self.h_list[h_index]
                    if cum_h >= self.numerical_config["liquidation_remaining_time"]:
                        self.numerical_solver.add_customized_position_constraint(
                            n_index,
                            h_index,
                            set_value_flag=True,
                            value_to_set=0,
                        )
                        continue
                    if dollar_position[n_index] == 0:
                        self.numerical_solver.add_customized_trade_constraint(
                            n_index,
                            h_index,
                            set_value_flag=True,
                            value_to_set=0,
                        )
                    elif dollar_position[n_index] > 0:
                        self.numerical_solver.add_customized_trade_constraint(
                            n_index,
                            h_index,
                            no_buy_flag=True,
                        )
                        self.numerical_solver.add_customized_position_constraint(
                            n_index,
                            h_index,
                            lower_bound=0,
                        )
                    else:
                        self.numerical_solver.add_customized_trade_constraint(
                            n_index,
                            h_index,
                            no_sell_flag=True,
                        )
                        self.numerical_solver.add_customized_position_constraint(
                            n_index,
                            h_index,
                            upper_bound=0,
                        )
                self.numerical_config["liquidation_remaining_time"] -= 1

        self.numerical_solver.add_orthogonal_factor_exposures(
            np.asarray(orthogonal_factor_exposures)
        )
        self.numerical_solver.add_transact_cost(
            intraday_data["transact_cost_fraction"],
            intraday_data["quadratic_tc_coeff"],
            tc_cost_multiplier.reshape(-1),
        )

        self.numerical_solver.set_alpha(intraday_data["alpha"])

        numerical_result = self.numerical_solver.numerical_solve()
        numerical_result["share_trade_desires"] = (
            numerical_result["dollar_trade_desire"]
            / intraday_data["dollar_value_per_share"]
        )

        numerical_result["constraint_info"]["h_list"] = self.h_list

        return numerical_result

    def post_order_process(self, gc_collect=True):
        self.numerical_solver.post_order_process(gc_collect=gc_collect)

    def get_solver(self):
        return self.numerical_solver


class OptimizerError(Exception):
    pass


class NamedCallback:
    def __init__(self, name, callback):
        self.name = name
        self.callback = callback

    def __call__(self):
        return {self.name: self.callback()}


class NamedCallbacks:
    def __init__(self):
        self.callback_list = []

    def add_callback(self, callback):
        self.callback_list.append(callback)

    def __call__(self):
        return {callback.name: callback() for callback in self.callback_list}


# help organize each component of the objective function
class MosekUtilityTerm:
    def __init__(self, name, type, mosek_expr, value_callbacks, gradient_callback):
        self.name = name
        self.type = type
        self.mosek_expr = mosek_expr
        self.value_callbacks = value_callbacks
        self.gradient_callback = gradient_callback


# help organize important constraints that we want to track and analyze
class MosekConstraintTerm:
    def __init__(self, name, mosek_expr, additional_callbacks):
        self.name = name
        self.mosek_expr = mosek_expr
        self.additional_callbacks = additional_callbacks

    def get_constraint_info(self):
        constraint_info = {}
        constraint_info[f"{self.name}_level"] = self.mosek_expr.level()
        constraint_info[f"{self.name}_dual"] = self.mosek_expr.dual()
        constraint_info.update(self.additional_callbacks())
        return constraint_info


# Deals with all numerical logics based on Mosek. numpy array as interface
class MosekOptimizer:
    def __init__(self, numerical_config: dict):
        # Force the license to the repo-root mosek.lic ONLY when it exists.
        # In installed images brain lives in site-packages; if the file wasn't
        # baked there, putlicensepath() overrides MOSEKLM_LICENSE_FILE with a
        # dead path and every solve fails -> po_executor swallows the error ->
        # silent all-zero positions ("fake success"). Fall back to MOSEK's
        # default search (env var / system paths) instead.
        _lic = pathlib.Path(__file__).absolute().parent.parent.parent / "mosek.lic"
        if _lic.is_file():
            mosek.Env().putlicensepath(_lic.as_posix())

        self.numerical_config = numerical_config

        self.log_file = None
        self.M = None
        self.last_obj = 0

    # make sure certain widely used parameters are updated only here so always synced
    def update_numerical_config(self):
        # turns out this scale is important for the optimizer to be numerically stable
        # we also need to be careful with the position limit and gmv limit, not too wild
        self.scale = self.numerical_config["scale"]
        self.gam = self.numerical_config["gamma"] * self.scale

    def set_log_file(self, log_file):
        self.log_file = log_file

    def _set_helper_matrix(self):
        l = int(self.n * self.H * (self.H + 1) / 2)
        data = np.ndarray((l))
        ij = np.ndarray((2, l), dtype=int)
        index = 0
        for k in range(self.H * self.n):
            data[index] = self.numerical_config["alpha_realization_coef"]
            ij[0, index] = k
            ij[1, index] = k
            index += 1
        for i in range(self.H):
            for j in range(i + 1, self.H):
                for k in range(self.n):
                    data[index] = 1
                    ij[0, index] = i * self.n + k
                    ij[1, index] = j * self.n + k
                    index += 1
        self.alpha_matrix_hn_hn = bsr_matrix(
            (data, ij), shape=(self.n * self.H, self.n * self.H)
        )
        # 0.5 0   1   0   1   0
        # 0   0.5 0   1   0   1
        # 0   0   0.5 0   1   0
        # 0   0   0   0.5 0   1
        # 0   0   0   0   0.5 0
        # 0   0   0   0   0   0.5
        # when n = 2, H = 3

        v_list = [1] * l
        i_list = []
        j_list = []
        for i in range(self.H):
            for j in range(i, self.H):
                for k in range(self.n):
                    i_list.append(j * self.n + k)
                    j_list.append(i * self.n + k)
        self.position_matrix_hn_hn = Matrix.sparse(
            self.n * self.H, self.n * self.H, i_list, j_list, v_list
        )
        # 1 0 0 0 0 0
        # 0 1 0 0 0 0
        # 1 0 1 0 0 0
        # 0 1 0 1 0 0
        # 1 0 1 0 1 0
        # 0 1 0 1 0 1
        # when n = 2, H = 3

    # Quadratic risk on X
    def _compute_risk_matrix(self, return_cov, components_to_remove):
        # remove components_to_remove from each of return_cov
        proj = np.eye(self.n)
        for v in components_to_remove:
            u = v / np.linalg.norm(v)
            P = np.outer(u, u)
            proj = proj @ (np.eye(self.n) - P)
        # have to do this after the cholesky decomposition,
        # since it makes the matrix from PD to PSD
        # for i in range(len(return_cov)):
        #     return_cov[i] = proj @ return_cov[i] @ proj.T

        M = [[None] * self.H for _ in range(self.H)]
        for i in range(self.H):
            M[i][i] = return_cov[i] * 2 / 3 * self.h_list[i]
            for k in range(i + 1, self.H):
                M[i][i] += return_cov[k] * 2 * self.h_list[k]
        for i in range(self.H - 1):
            for j in range(i + 1, self.H):
                M[i][j] = return_cov[j] * self.h_list[j]
                for k in range(j + 1, self.H):
                    M[i][j] += return_cov[k] * 2 * self.h_list[k]
                M[j][i] = M[i][j]

        # this scipy.linalg.cholesky does L.T@L
        self.L_hn_hn = cholesky(np.block(M) * self.gam) @ block_diag(
            *([proj] * self.H)
        )

        N = [None] * self.H
        for i in range(self.H):
            N[i] = return_cov[i] * self.h_list[i]
            for k in range(i + 1, self.H):
                N[i] += return_cov[k] * 2 * self.h_list[k]
            N[i] = proj @ N[i] @ proj
        self.init_risk_matrix_n_hn = np.block(N) * self.gam
        self.init_risk_matrix_n_n = (
            proj @ return_cov[0] @ proj * self.h_list[0] * self.gam
        )

        # a pure integral of the return covariance matrix along the time
        L = np.zeros((self.n, self.n))
        for i in range(self.H):
            L += return_cov[i] * self.h_list[i]
        self.constant_risk_matrix_n_n = proj @ L @ proj.T * self.gam

    def set_basic_model(self, n, h_list, return_cov, components_to_remove):
        self.n = n
        self.H = len(h_list)
        self.h_list = h_list
        self.h_cumsum = np.cumsum(h_list)
        self.T = [0] + self.h_cumsum.tolist()
        self._set_helper_matrix()
        self._compute_risk_matrix(return_cov, components_to_remove)

        assert self.M is None
        self.M = Model()
        self.M.setLogHandler(self.log_file)
        self.M.setSolverParam("logCutSecondOpt", 0)

        # Mosek auto-scaling seems ok here

        # these three may not apply any more after we modify current problem, 07/27/2021
        # self.M.setSolverParam("presolveLindepUse", "off")
        # self.M.setSolverParam("presolveEliminatorMaxNumTries", 0)
        # self.M.setSolverParam("intpntSolveForm", "dual")

        # Mosek defaults to all cores per solve; [PO_SIM] mosek_num_threads caps it (0 = all cores).
        # Cap it for local parallel sims so concurrent solves don't oversubscribe.
        mosek_threads = self.numerical_config["mosek_num_threads"]
        if mosek_threads > 0:
            self.M.setSolverParam("numThreads", mosek_threads)

        # https://docs.mosek.com/9.3/pythonfusion/solving-conic.html#adjusting-optimality-criteria
        multiplier = self.numerical_config["optimality_criteria_multiplier"]
        if multiplier != 1:
            self.M.setSolverParam("intpntCoTolPfeas", 1.0e-8 * multiplier)
            self.M.setSolverParam("intpntCoTolDfeas", 1.0e-8 * multiplier)
            self.M.setSolverParam("intpntCoTolRelGap", 1.0e-8 * multiplier)
            self.M.setSolverParam("intpntCoTolInfeas", 1.0e-12 * multiplier)

        # or we carefully tune the tolerance for the optimal solution
        self.M.acceptedSolutionStatus(AccSolutionStatus.Anything)

        self.X_hn = self.M.variable(self.n * self.H)
        self.init_positions_hn = self.M.parameter(self.n * self.H)
        self.mu1_hn = self.M.parameter(self.n * self.H)
        self.mu2_hn = self.M.parameter(self.n * self.H)

        self.dollar_positions_hn = (
            self.position_matrix_hn_hn @ self.X_hn + self.init_positions_hn
        )

        self.utility_list = []
        self.constraint_list = []

        linear_alpha = Expr.dot(self.mu1_hn, self.X_hn)
        value_callbacks = [
            lambda: np.dot(self.f_list_hn, self.init_positions_hn.getValue()),
            lambda: np.dot(
                self.mu1_hn.getValue()[: self.n], self.X_hn.level()[: self.n]
            ),
            lambda: np.dot(
                self.mu1_hn.getValue()[self.n :], self.X_hn.level()[: -self.n]
            ),
        ]
        gradient_callback = lambda: self.mu1_hn.getValue()
        self.utility_list.append(
            MosekUtilityTerm(
                "linear_alpha",
                "alpha",
                linear_alpha,
                value_callbacks,
                gradient_callback,
            )
        )

        self.linear_risk = Expr.dot(self.mu2_hn, self.X_hn)
        value_callbacks = [
            lambda: -self.x_init_n
            @ self.constant_risk_matrix_n_n
            @ self.x_init_n.T,
            lambda: np.dot(self.mu2_hn.getValue(), self.X_hn.level()),
        ]
        gradient_callback = lambda: self.mu2_hn.getValue()
        self.utility_list.append(
            MosekUtilityTerm(
                "linear_risk",
                "risk",
                self.linear_risk,
                value_callbacks,
                gradient_callback,
            )
        )

        self.q = self.M.variable()
        self.M.constraint(
            Expr.vstack(1, self.q, self.L_hn_hn @ self.X_hn),
            Domain.inRotatedQCone(),
        )
        self.quadratic_risk = -self.q
        value_callback = lambda: -self.q.level()[0]
        gradient_callback = lambda: -self.L_hn_hn.T @ (
            self.L_hn_hn @ self.X_hn.level()
        )

        self.utility_list.append(
            MosekUtilityTerm(
                "quadratic_risk",
                "risk",
                self.quadratic_risk,
                [value_callback],
                gradient_callback,
            )
        )

    def set_init_position(self, x_init: np.ndarray):
        self.x_init_n = x_init / self.scale
        self.init_positions_hn.setValue(np.block([self.x_init_n] * self.H))
        init_risk = self.x_init_n @ self.init_risk_matrix_n_hn
        self.mu2_hn.setValue(-init_risk)
        self.mu2_n = -self.x_init_n @ self.init_risk_matrix_n_n

    def set_portfolio_volatility_limit(self, portfolio_volatility_limit):
        constant_risk = (
            -self.x_init_n @ self.constant_risk_matrix_n_n @ self.x_init_n.T
        )
        total_risk = self.linear_risk + self.quadratic_risk + constant_risk
        adjusted_portfolio_volatility_limit = (
            portfolio_volatility_limit * self.gam / self.scale / self.scale
        )
        self.portfolio_volatility_limit_constraint = self.M.constraint(
            adjusted_portfolio_volatility_limit + total_risk >= 0.0
        )
        callbacks = NamedCallbacks()
        callbacks.add_callback(
            NamedCallback(
                "adjusted_portfolio_volatility_limit",
                lambda: adjusted_portfolio_volatility_limit,
            )
        )
        self.constraint_list.append(
            MosekConstraintTerm(
                "portfolio_volatility_limit",
                self.portfolio_volatility_limit_constraint,
                callbacks,
            )
        )

    def add_orthogonal_factor_exposures(
        self, orthogonal_factor_exposures_h_x_n: np.ndarray
    ):
        self.orthogonal_factor_exposures_x_h_n = (
            orthogonal_factor_exposures_h_x_n.transpose(1, 0, 2)
        )
        orthogonal_factor_exposure_matrices_x_h_hn = [
            Matrix.sparse(
                block_diag(
                    *(
                        [
                            orthogonal_factor_exposure_n
                            for orthogonal_factor_exposure_n in orthogonal_factor_exposure_h_n
                        ]
                    )
                )[self.numerical_config["num_of_horizons_to_ignore_factor_neutral"] :]
            )
            for orthogonal_factor_exposure_h_n in self.orthogonal_factor_exposures_x_h_n
        ]
        self.factor_orthogonal_constraints = [
            self.M.constraint(
                orthogonal_factor_exposure_matrix @ self.dollar_positions_hn == 0.0
            )
            for orthogonal_factor_exposure_matrix in orthogonal_factor_exposure_matrices_x_h_hn
        ]
        """
        for i, factor_orthogonal_constraint in enumerate(
            self.factor_orthogonal_constraints
        ):
            self.constraint_list.append(
                MosekConstraintTerm(
                    f"factor_orthogonal_constraint_{i}",
                    factor_orthogonal_constraint,
                    NamedCallbacks(),
                )
            )
        """

    def add_position_limit(
        self, upper_position_limit: np.ndarray, lower_position_limit: np.ndarray
    ):
        self.dp_upper_bound_hn = self.M.parameter(self.n * self.H)
        self.dp_lower_bound_hn = self.M.parameter(self.n * self.H)
        self.dp_upper_bound_hn.setValue(upper_position_limit / self.scale)
        self.dp_lower_bound_hn.setValue(lower_position_limit / self.scale)

        self.dp_upper_bound_constraint = self.M.constraint(
            self.dp_upper_bound_hn - self.dollar_positions_hn >= 0.0
        )
        self.dp_lower_bound_constraint = self.M.constraint(
            self.dollar_positions_hn - self.dp_lower_bound_hn >= 0.0
        )
        callbacks = NamedCallbacks()
        callbacks.add_callback(
            NamedCallback(
                "dp_upper_bound_hn", lambda: self.dp_upper_bound_hn.getValue()
            )
        )
        self.constraint_list.append(
            MosekConstraintTerm(
                "dp_upper_bound", self.dp_upper_bound_constraint, callbacks
            )
        )
        callbacks = NamedCallbacks()
        callbacks.add_callback(
            NamedCallback(
                "dp_lower_bound_hn", lambda: self.dp_lower_bound_hn.getValue()
            )
        )
        self.constraint_list.append(
            MosekConstraintTerm(
                "dp_lower_bound", self.dp_lower_bound_constraint, callbacks
            )
        )

    def add_trade_limit(self, trade_limit: np.ndarray):
        self.X_upper_bound_hn = self.M.parameter(self.n * self.H)
        self.X_lower_bound_hn = self.M.parameter(self.n * self.H)
        self.X_upper_bound_hn.setValue(trade_limit / self.scale)
        self.X_lower_bound_hn.setValue(-trade_limit / self.scale)

        self.X_upper_bound_constraint = self.M.constraint(
            self.X_upper_bound_hn - self.X_hn >= 0.0
        )
        self.X_lower_bound_constraint = self.M.constraint(
            self.X_hn - self.X_lower_bound_hn >= 0.0
        )
        callbacks = NamedCallbacks()
        callbacks.add_callback(
            NamedCallback(
                "X_upper_bound_hn", lambda: self.X_upper_bound_hn.getValue()
            )
        )
        self.constraint_list.append(
            MosekConstraintTerm(
                "X_upper_bound", self.X_upper_bound_constraint, callbacks
            )
        )
        callbacks = NamedCallbacks()
        callbacks.add_callback(
            NamedCallback(
                "X_lower_bound_hn", lambda: self.X_lower_bound_hn.getValue()
            )
        )
        self.constraint_list.append(
            MosekConstraintTerm(
                "X_lower_bound", self.X_lower_bound_constraint, callbacks
            )
        )

    def add_total_vlm_limit(self, total_vlm_limit, vlm_limit_h):
        self.vlm_limit_h = self.M.parameter(self.H)
        self.vlm_limit_h.setValue(
            total_vlm_limit
            * np.asarray(vlm_limit_h)
            / np.sum(vlm_limit_h)
            / self.scale
        )
        abs_X_hn = self.M.variable(self.n * self.H)
        self.M.constraint(abs_X_hn + self.X_hn >= 0.0)
        self.M.constraint(abs_X_hn - self.X_hn >= 0.0)
        self.vlm_h = Expr.sum(Expr.reshape(abs_X_hn, self.H, self.n), 1)
        self.vlm_limit_constraint = self.M.constraint(
            self.vlm_limit_h - self.vlm_h >= 0.0
        )
        callbacks = NamedCallbacks()
        callbacks.add_callback(
            NamedCallback("vlm_limit_h", lambda: self.vlm_limit_h.getValue())
        )
        self.constraint_list.append(
            MosekConstraintTerm("vlm_limit", self.vlm_limit_constraint, callbacks)
        )

    def add_gmv_limit(self, gmv_limit: np.ndarray):
        if self.numerical_config["enable_gmv_constraint"]:
            self.gmv_limit_h = self.M.parameter(self.H)
            self.gmv_limit_h.setValue(gmv_limit / self.scale)
            abs_dollar_positions_hn = self.M.variable(self.n * self.H)
            self.M.constraint(
                abs_dollar_positions_hn + self.dollar_positions_hn >= 0.0
            )
            self.M.constraint(
                abs_dollar_positions_hn - self.dollar_positions_hn >= 0.0
            )
            self.gmv_h = Expr.sum(
                Expr.reshape(abs_dollar_positions_hn, self.H, self.n), 1
            )
            self.gmv_constraint = self.M.constraint(
                self.gmv_limit_h - self.gmv_h >= 0.0
            )
            callbacks = NamedCallbacks()
            callbacks.add_callback(
                NamedCallback("gmv_limit_h", lambda: self.gmv_limit_h.getValue())
            )
            self.constraint_list.append(
                MosekConstraintTerm("gmv_limit", self.gmv_constraint, callbacks)
            )

    def add_nmv_limit(self, nmv_limit: np.ndarray):
        self.nmv_limit_h = self.M.parameter(self.H)
        self.nmv_limit_h.setValue(nmv_limit / self.scale)
        self.nmv_h = Expr.sum(Expr.reshape(self.dollar_positions_hn, self.H, self.n), 1)
        self.nmv_upper_bound_constraint = self.M.constraint(
            self.nmv_limit_h - self.nmv_h >= 0.0
        )
        self.nmv_lower_bound_constraint = self.M.constraint(
            self.nmv_limit_h + self.nmv_h >= 0.0
        )
        callbacks = NamedCallbacks()
        callbacks.add_callback(
            NamedCallback("nmv_limit_h", lambda: self.nmv_limit_h.getValue())
        )
        self.constraint_list.append(
            MosekConstraintTerm(
                "nmv_upper_bound", self.nmv_upper_bound_constraint, callbacks
            )
        )
        self.constraint_list.append(
            MosekConstraintTerm(
                "nmv_lower_bound", self.nmv_lower_bound_constraint, callbacks
            )
        )

    def add_transact_cost(
        self,
        transact_cost_fraction: np.ndarray,
        quadratic_tc_coeff: np.ndarray,
        tc_cost_multiplier: np.ndarray,
    ):
        if self.numerical_config["include_transact_cost"]:
            self.transact_cost_vector_hn = self.M.parameter(self.n * self.H)
            self.transact_cost_vector_hn.setValue(
                np.block([-transact_cost_fraction] * self.H) * tc_cost_multiplier
            )
            self.absX_hn = self.M.variable(self.n * self.H)
            self.M.constraint(self.absX_hn + self.X_hn >= 0.0)
            self.M.constraint(self.absX_hn - self.X_hn >= 0.0)
            transact_cost = Expr.dot(self.transact_cost_vector_hn, self.absX_hn)
            value_callback = lambda: np.dot(
                self.transact_cost_vector_hn.getValue(), self.absX_hn.level()
            )
            gradient_callback = (
                lambda: self.transact_cost_vector_hn.getValue()
                * np.sign(self.X_hn.level())
            )
            self.utility_list.append(
                MosekUtilityTerm(
                    "transact_cost",
                    "cost",
                    transact_cost,
                    [value_callback],
                    gradient_callback,
                )
            )

        if self.numerical_config["include_quadratic_transact_cost"]:
            self.quadratic_tc_coeff_hn = self.M.parameter(self.n * self.H)
            self.quadratic_tc_coeff_hn.setValue(
                np.block(
                    [
                        quadratic_tc_coeff * np.sqrt(self.scale / h)
                        for h in self.h_list
                    ]
                )
                * tc_cost_multiplier
            )
            self.q2 = self.M.variable()
            self.M.constraint(
                Expr.vstack(
                    1,
                    self.q2,
                    Expr.mulElm(self.quadratic_tc_coeff_hn, self.X_hn),
                ),
                Domain.inRotatedQCone(),
            )
            quadratic_transact_cost = -self.q2
            value_callback = lambda: -self.q2.level()[0]
            gradient_callback = (
                lambda: -self.quadratic_tc_coeff_hn.getValue()
                * self.quadratic_tc_coeff_hn.getValue()
                * self.X_hn.level()
            )
            self.utility_list.append(
                MosekUtilityTerm(
                    "quadratic_transact_cost",
                    "cost",
                    quadratic_transact_cost,
                    [value_callback],
                    gradient_callback,
                )
            )

    # be careful with future user cases if we need the self.scale here
    def add_customized_trade_constraint(
        self,
        n_index,
        h_index,
        no_buy_flag=False,
        no_sell_flag=False,
        set_value_flag=False,
        value_to_set=0,
    ):
        index = h_index * self.n + n_index
        if set_value_flag:
            self.M.constraint(self.X_hn[index] == value_to_set)
        elif no_buy_flag and no_sell_flag:
            self.M.constraint(self.X_hn[index] == 0.0)
        elif no_buy_flag and (not no_sell_flag):
            self.M.constraint(self.X_hn[index] <= 0.0)
        elif (not no_buy_flag) and no_sell_flag:
            self.M.constraint(self.X_hn[index] >= 0.0)

    # be careful with future user cases if we need the self.scale here
    def add_customized_position_constraint(
        self,
        n_index,
        h_index,
        upper_bound=None,
        lower_bound=None,
        set_value_flag=False,
        value_to_set=0,
    ):
        index = h_index * self.n + n_index
        if set_value_flag:
            self.M.constraint(self.dollar_positions_hn[index] == value_to_set)
            return
        if upper_bound is not None:
            self.M.constraint(self.dollar_positions_hn[index] <= upper_bound)
        if lower_bound is not None:
            self.M.constraint(self.dollar_positions_hn[index] >= lower_bound)

    def set_alpha(self, alpha: np.ndarray):
        self.f_list_hn = alpha.copy()
        self.f_list_hn.resize(self.H * self.n)
        self.mu1_hn.setValue(self.alpha_matrix_hn_hn @ self.f_list_hn)
        self.mu1_n = self.f_list_hn[: self.n]

    def _solve_once(self):
        self.M.solve()
        admissible_status = [SolutionStatus.Optimal, SolutionStatus.Unknown]
        if (
            self.M.getPrimalSolutionStatus() not in admissible_status
            or self.M.getDualSolutionStatus() not in admissible_status
        ):
            exception_msg = f"Problem status: {self.M.getProblemStatus()}\n"
            exception_msg += (
                f"Primal solution status: {self.M.getPrimalSolutionStatus()}\n"
            )
            exception_msg += f"Dual solution status: {self.M.getDualSolutionStatus()}\n"
            # exception_msg += f"Current position: {self.init_positions_hn.getValue()[:self.n]}\n"
            # exception_msg += f"Transact cost ratio: {self.transact_cost_vector_hn.getValue()[:self.n]}\n"
            raise OptimizerError(exception_msg)
        # try:
        #     if (
        #         self.M.getPrimalSolutionStatus() != SolutionStatus.Optimal
        #         or self.M.getDualSolutionStatus() != SolutionStatus.Optimal
        #     ):
        #         raise Exception("None Optimal Solution")
        # # can't catch keyboard interrupt since M.solve() doesn't tell
        # except Exception as solver_error:
        #     try:
        #         primal = self.M.primalObjValue()
        #         dual = self.M.dualObjValue()
        #     except Exception as solution_error:
        #         raise OptimizerError("Can't solve.\n") from solution_error
        #     else:
        #         if np.abs(primal - dual) > 0.1:
        #             raise (
        #                 OptimizerError(
        #                     f"Duality gap too large, Primal: {primal}, Dual: {dual}\nLast obj: {self.last_obj}\n"
        #                 )
        #             ) from solver_error
    def numerical_solve(self) -> np.ndarray:
        self.M.objective(
            ObjectiveSense.Maximize, Expr.add([u.mosek_expr for u in self.utility_list])
        )
        self._solve_once()

        if (
            self.M.getPrimalSolutionStatus() != SolutionStatus.Optimal
            or self.M.getDualSolutionStatus() != SolutionStatus.Optimal
        ):
            # we can do some counter here
            logger.info("Did not find optimal solution!")

        alpha_utility = 0
        risk_utility = 0
        cost_utility = 0
        for u in self.utility_list:
            utility = 0
            for callback in u.value_callbacks:
                utility += callback()
            utility = utility * self.scale
            if u.type == "alpha":
                alpha_utility += utility
            elif u.type == "risk":
                risk_utility += utility
            elif u.type == "cost":
                cost_utility += utility

        alpha_gradient = np.zeros(self.n * self.H)
        cost_gradient = np.zeros(self.n * self.H)
        risk_gradient = np.zeros(self.n * self.H)
        for u in self.utility_list:
            if "quadratic" in u.name:
                continue
            gradient = u.gradient_callback()
            if u.type == "alpha":
                alpha_gradient += gradient
            elif u.type == "cost":
                cost_gradient += gradient
            elif u.type == "risk":
                risk_gradient += gradient
        alpha_gradient = alpha_gradient.reshape(self.H, self.n)
        cost_gradient = cost_gradient.reshape(self.H, self.n)
        risk_gradient = risk_gradient.reshape(self.H, self.n)

        # self.last_obj = self.M.primalObjValue()
        self.numerical_result_hn = self.X_hn.level()
        self.dollar_trade_desire = (
            self.numerical_result_hn.reshape(self.H, self.n) * self.scale
        )
        # self.solver_status = dict(
        #     num_iterations=self.M.getSolverIntInfo("intpntIter"),
        #     primal_value=self.M.primalObjValue(),
        #     dual_value=self.M.dualObjValue(),
        # )

        alpha_hurdle_raw = self.mu1_n
        risk_hurdle_raw = self.mu2_n

        constraint_info = {}
        if self.numerical_config["produce_constraint_info"]:
            for constraint in self.constraint_list:
                constraint_info.update(constraint.get_constraint_info())

        numerical_result = dict(
            dollar_trade_desire=self.dollar_trade_desire,
            alpha_utility=alpha_utility,
            risk_utility=risk_utility,
            cost_utility=cost_utility,
            alpha_gradient=alpha_gradient,
            cost_gradient=cost_gradient,
            risk_gradient=risk_gradient,
            alpha_hurdle_raw=alpha_hurdle_raw,
            risk_hurdle_raw=risk_hurdle_raw,
            constraint_info=constraint_info,
        )
        return numerical_result

    def get_solver_status(self) -> dict:
        return self.solver_status

    def post_order_process(self, gc_collect=True):
        if self.M is not None:
            self.M.dispose()
            self.M = None
            if gc_collect:
                gc.collect()
        return

        # basic info
        logger.info(
            f"Time: {self.M.getSolverDoubleInfo('optimizerTime')} Iterations: {self.M.getSolverIntInfo('intpntIter')}"
        )
        logger.info(
            f"Primal Objective {self.M.primalObjValue()}, Dual Objective {self.M.dualObjValue()}"
        )
        logger.info(
            f"Init position: {self.x_init_n}, initial GMV: {np.sum(np.abs(self.x_init_n))}"
        )
        logger.info(
            f"Trade desire: {self.X_hn.level()}, total size: {np.sum(np.abs(self.X_hn.level()[:self.n]))}"
        )
        logger.info(
            f"Target position: {self.x_init_n + self.X_hn.level().reshape(self.H, self.n).cumsum(axis=0)}"
        )
        logger.info(
            f"target GMV: {np.sum(np.abs(self.x_init_n + self.X_hn.level()[:self.n]))}"
        )

        # utility decomposition
        s = 0
        self.last_utility_snapshot = {}
        for u in self.utility_list:
            v = 0
            for index, callback in enumerate(u.value_callbacks):
                value = callback()
                logger.info(f"{u.name}_{index} = {value}")
                v += value
                s += value
            self.last_utility_snapshot[u.name] = v
        logger.info(f"total utility = {s}")

        if self.numerical_config["gradient_analysis"]:
            self.last_gradient_snapshot = {}
            start_time = timer.monotonic()

            selected_index = np.argsort(-np.abs(self.X_hn.level()[: self.n]))[:10]
            selected_index = np.arange(self.n)
            # gradient decomposition
            g = np.zeros(self.n * self.H)
            for u in self.utility_list:
                if u.gradient_callback is not None:
                    gradient = u.gradient_callback()
                    logger.info(f"{u.name} gradient = {gradient[selected_index]}")
                    self.last_gradient_snapshot[f"{u.name}_gradient"] = (
                        gradient.reshape(-1, self.n)
                    )
                    g += gradient
            logger.info(f"total gradient = {g[selected_index]}")

            # flags for boundary and discontinuity on gradient
            # Not clear how to dynamically pick the thresh
            # Maybe we will even find more continuous treatment as if weighted regression
            numerical_thresh = 1000
            zero_trade_flag = np.abs(self.X_hn.level()[: self.n]) > numerical_thresh
            zero_dp_flag = (
                np.abs(self.X_hn.level()[: self.n] + self.x_init_n) > numerical_thresh
            )
            final_mask = zero_trade_flag & zero_dp_flag
            for constraint in [
                self.dp_lower_bound_constraint,
                self.dp_upper_bound_constraint,
            ]:
                mask = constraint.level()[: self.n] > numerical_thresh
                final_mask = final_mask & mask
                logger.info(
                    f"{self.n - sum(mask)} of {self.n} assets are within {numerical_thresh} of the limit"
                )
                logger.info(f"constraint level: {constraint.level()[selected_index]}")
                logger.info(f"mask: {final_mask[selected_index]}")
            self.last_gradient_snapshot["interior_flag"] = final_mask.reshape(
                -1, self.n
            )

            # gradient on constraints
            constraint_gradient = np.asarray(self.orthogonal_factor_exposures_x_h_n)[
                :, 0, :
            ]
            for exposure in self.orthogonal_factor_exposures_x_h_n:
                logger.info(
                    f"orthogonal factor exposure: {exposure[0][selected_index]}"
                )
            if self.numerical_config["enable_gmv_constraint"]:
                gmv_gradient = np.sign(self.X_hn.level()[: self.n] + self.x_init_n)
                logger.info(f"GMV constraint: {gmv_gradient[selected_index]}")
                constraint_gradient = np.vstack([constraint_gradient, gmv_gradient])

            # A Lagrangian process or say a regression
            gradient = g[: self.n]
            Y = gradient * final_mask
            X = constraint_gradient * final_mask
            constraint_gradient = np.negative(constraint_gradient) * (
                pinv(X @ X.T) @ (X @ Y)
            ).reshape(-1, 1)
            residual_gradient = gradient + np.sum(constraint_gradient, axis=0)
            logger.info(f"residual gradient = {residual_gradient[selected_index]}")

            reshaped = constraint_gradient.reshape(-1, self.n)
            for i in range(reshaped.shape[0]):
                self.last_gradient_snapshot[f"constraint_gradient_{i}"] = reshaped[
                    i
                ]
            self.last_gradient_snapshot["residual_gradient"] = (
                residual_gradient.reshape(-1, self.n)
            )

            end_time = timer.monotonic()
            logger.info(f"Gradient analysis time: {end_time - start_time}")
            logger.info("")

        # misc items
        for constraint in self.factor_orthogonal_constraints:
            logger.info(
                f"Constraint value and dual: {constraint.level()} {constraint.dual()}"
            )
        if self.numerical_config["enable_gmv_constraint"]:
            logger.info(
                f"GMV constraint value and dual: {self.gmv_constraint.level()} {self.gmv_constraint.dual()}"
            )

        logger.info("")
