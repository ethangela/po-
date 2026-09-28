import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict

from brain.utils import Utils


class DataInterface:
    intraday_data_names = ["street_fill", "EP", "P", "PP", "Q"]

    def _get_external_data_name(self, data_name):
        if data_name == "P":
            return "po_sim_p_dim"
        elif data_name == "EP":
            return "po_sim_ep_dim"
        elif data_name == "Q":
            return "po_sim_q_dim"
        elif data_name == "PP":
            return "po_sim_pp_dim"
        else:
            return data_name + "_dim"

    def __init__(self, universe):
        self.universe = universe
        # TODO: when we later don't need to maintain the whole history for the get_output
        # we can just maintain the last row of the data
        self.data = defaultdict(dict)
        self.input_data = None
        for data_name in self.intraday_data_names:
            for asset in self.universe:
                self.data[data_name][asset] = []
        self.last_time = None

    def put(self, dim, asset, data):
        self.data[dim][asset].append(data)

    def get_output(self, time):
        output = defaultdict(dict)
        for data_name in self.intraday_data_names:
            for asset in self.universe:
                output[data_name][asset] = []
                for row in self.data[data_name][asset]:
                    if (
                        self.last_time is not None
                        and row["timestamp"] <= self.last_time
                    ):
                        continue
                    output[data_name][asset].append(row)
        self.last_time = time
        return output

    def get_intraday_data_per_asset(self, data_name, time, asset):
        external_data_name = self._get_external_data_name(data_name)
        if self.input_data and external_data_name in self.input_data:
            return self.input_data[external_data_name][asset]
        else:
            return self.data[data_name][asset][-1]

    # here time won't be respected, we will use the last row of the data
    def _get_intraday_data(self, data_name, column_names=None, source=None):
        if source is None:
            source = self.data

        def _latest_row(asset):
            row = source[data_name][asset]
            return row[-1] if isinstance(row, list) else row

        data_list = []
        if column_names is not None:
            keys = column_names
        else:
            keys = list(_latest_row(self.universe[0]).keys())
            keys.remove("timestamp")

        for asset in self.universe:
            data_dict = _latest_row(asset)
            data = xr.DataArray(
                [data_dict[k] for k in keys], dims=["columns"], coords={"columns": keys}
            )
            data_list.append(data)
        intraday_data = xr.concat(data_list, dim="asset").assign_coords(
            asset=self.universe
        )
        return intraday_data

    def get_asof_data(self, data_name, date, column_names=None):
        raise NotImplementedError

    def get_intraday_data(self, data_name, date, time, column_names=None):
        raise NotImplementedError


class GCSDataInterface(DataInterface):
    direct_data_names = [
        "cov",
        "timebin_cov",
        "asset_info",
        "fundamental_alpha_panel",
        "external_fill",
    ]
    per_asset_df_data_names = [
        "minutebar",
        "mktstats_timebin",
        "alpha_po",
        "econevents",
        "intraday_cov",
        "state",
        "hurdle_alpha",
        "execution_alpha",
    ]

    def __init__(self, paths, universe, read_resolver=None):
        super().__init__(universe)
        self.mode = "GCS"
        self.paths = paths
        self._resolve = read_resolver or (lambda d, dt: f"{d.rstrip('/')}/{dt}.pkl")
        self.intraday_data_cache = {}

    def get_asof_data(self, data_name, date=None, column_names=None):
        if data_name in self.direct_data_names:
            return self._get_direct_data(data_name, date)
        elif data_name in self.per_asset_df_data_names:
            return self._get_per_asset_df_data(data_name, date, column_names)
        else:
            raise ValueError(f"Data name {data_name} not supported")

    def get_intraday_data(self, data_name, date, timestamp, column_names=None):
        if data_name in self.intraday_data_cache:
            data = self.intraday_data_cache[data_name]
            intraday_data = data.sel(timestamp=timestamp, method="ffill", drop=True)
            if column_names is not None:
                intraday_data = intraday_data.sel(columns=column_names)
        elif data_name in self.per_asset_df_data_names or data_name.startswith(
            "alpha_po_"
        ):
            data = self._get_per_asset_df_data(data_name, date)
            self.intraday_data_cache[data_name] = data
            intraday_data = data.sel(timestamp=timestamp, method="ffill", drop=True)
            if column_names is not None:
                intraday_data = intraday_data.sel(columns=column_names)
        elif data_name in self.intraday_data_names:
            intraday_data = self._get_intraday_data(data_name, column_names)
        else:
            raise ValueError(f"Data name {data_name} not supported")
        return intraday_data

    def _get_direct_data(self, data_name, date):
        return Utils.read_pickle(self._resolve(self.paths[data_name + "_dir"], date))

    def _get_per_asset_df_data(self, data_name, date, column_names=None):
        data_list = []
        for asset in self.universe:
            data = Utils.read_pickle(
                self._resolve(self.paths[data_name + "_dir"], f"{asset}/{date}")
            )
            if data_name.startswith("alpha_po_"):
                data.index.name = "timestamp"

            if column_names is None:
                column_names = data.columns.sort_values()
            data_list.append(
                xr.DataArray(data[column_names], dims=["timestamp", "columns"])
            )
        return xr.concat(data_list, dim="asset", join="exact").assign_coords(
            asset=self.universe
        )

    def get_output_table(self):
        output = {}
        for data_name in self.intraday_data_names:
            if data_name in ["P", "Q"]:
                output[data_name] = self._get_output_table_with_universe(data_name)
            elif data_name in ["PP"]:
                output[data_name] = self._get_output_table_per_asset(
                    data_name, self.universe[0]
                )
            elif data_name in ["EP"]:
                output[data_name] = self._get_output_dataframe(data_name)
        return output

    def _get_output_table_with_universe(self, data_name):
        universe_data_list = []
        for asset in self.universe:
            data = self._get_output_table_per_asset(data_name, asset)
            universe_data_list.append(data)
        return (
            xr.concat(universe_data_list, dim="asset")
            .assign_coords(asset=self.universe)
            .astype(np.float64)
        )

    def _get_output_table_per_asset(self, data_name, asset):
        data_list = []
        time_list = []
        for row in self.data[data_name][asset]:
            data = xr.DataArray(
                list(row.values()),
                dims=["columns"],
                coords={"columns": list(row.keys())},
            )
            data_list.append(data)
            time_list.append(row["timestamp"])
        return (
            xr.concat(data_list, dim="timestamp")
            .drop_sel(columns="timestamp")
            .assign_coords(timestamp=time_list)
            .astype(np.float32)
        )

    def _get_output_dataframe(self, data_name):
        return {
            asset: pd.DataFrame(self.data[data_name][asset]) for asset in self.universe
        }


class BrainAppDataInterface(DataInterface):
    def __init__(self, universe, asof_inputs, input_data):
        super().__init__(universe)
        self.mode = "BrainApp"
        self.asof_inputs = asof_inputs
        self.input_data = input_data

    def get_asof_data(self, data_name, date=None, column_names=None):
        data_name = self._get_external_data_name(data_name)
        if data_name in self.asof_inputs:
            if isinstance(self.asof_inputs[data_name], dict):
                data_list = []
                for asset in self.universe:
                    data = self.asof_inputs[data_name][asset]
                    if column_names is not None:
                        data = data[column_names]
                    data_list.append(xr.DataArray(data, dims=["timestamp", "columns"]))
                data = xr.concat(data_list, dim="asset").assign_coords(
                    asset=self.universe
                )
            else:
                data = self.asof_inputs[data_name]
                if column_names is not None:
                    data = data[column_names]
            return data
        else:
            raise ValueError(f"Data name {data_name} not supported")

    def get_intraday_data(self, data_name, date, timestamp, column_names=None):
        external_data_name = self._get_external_data_name(data_name)
        if external_data_name in self.input_data:
            intraday_data = self._get_intraday_data(
                external_data_name, column_names, source=self.input_data
            )
            return intraday_data
        elif external_data_name in self.asof_inputs:
            data = self.get_asof_data(data_name, date, column_names)
            return data.sel(timestamp=timestamp, drop=True)
        elif data_name in self.intraday_data_names:
            intraday_data = self._get_intraday_data(
                data_name, column_names, source=self.data
            )
            return intraday_data
        else:
            raise ValueError(f"Data name {data_name} not supported")
