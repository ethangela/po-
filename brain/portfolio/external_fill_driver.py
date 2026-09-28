import pandas as pd


class SimExternalFillDriver:
    def __init__(self, po_sim_params):
        self.universe = po_sim_params.universe
        self.external_fill_data = (
            self._load_per_product_external_fills(po_sim_params)
            if po_sim_params.per_product_external_fill_dim
            else po_sim_params.data_interface.get_asof_data(
                "external_fill", po_sim_params.date
            )
        )
        self.index = 0

    @property
    def has_fills(self):
        return bool(self.external_fill_data)

    def _load_per_product_external_fills(self, po_sim_params):
        fills = []
        dim = po_sim_params.per_product_external_fill_dim
        tz = po_sim_params.tz
        for asset in self.universe:
            df = po_sim_params.config_manager.load_dataframe_by_dim(
                dim, asset, po_sim_params.date
            )
            if "timestamp" not in df.columns:
                df = df.reset_index()

            for row in df[df["fill_qty"] != 0].to_dict(orient="records"):
                timestamp = pd.Timestamp(row["timestamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(tz)
                else:
                    timestamp = timestamp.tz_convert(tz)
                fills.append(
                    {
                        "timestamp": timestamp,
                        "asset": asset,
                        "fill_qty": row["fill_qty"],
                        "fill_price": row["fill_price"],
                    }
                )

        fills.sort(key=lambda fill: fill["timestamp"])
        return fills

    def get_fills(self, time):
        fills = {asset: [] for asset in self.universe}
        while (
            self.index < len(self.external_fill_data)
            and self.external_fill_data[self.index]["timestamp"] <= time
        ):
            fill = self.external_fill_data[self.index]
            asset = fill["asset"]
            if asset in self.universe:
                fills[asset].append(
                    {key: value for key, value in fill.items() if key != "asset"}
                )
            self.index += 1
        return fills
