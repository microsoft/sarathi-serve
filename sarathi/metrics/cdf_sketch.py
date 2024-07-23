import logging

import numpy as np
import pandas as pd
import plotly_express as px
import wandb
from ddsketch.ddsketch import DDSketch

logger = logging.getLogger(__name__)


class CDFSketch:

    def __init__(
        self,
        metric_name: str,
        relative_accuracy: float = 0.001,
        num_quantiles_in_df: int = 101,
    ) -> None:
        # metrics are a data series of two-dimensional (x, y) datapoints
        self.sketch = DDSketch(relative_accuracy=relative_accuracy)
        # column name
        self.metric_name = metric_name

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self._last_data = 0

        self._num_quantiles_in_df = num_quantiles_in_df

    @property
    def mean(self) -> float:
        return self.sketch.avg

    @property
    def median(self) -> float:
        return self.sketch.get_quantile_value(0.5)

    @property
    def sum(self) -> float:
        return self.sketch.sum

    def __len__(self):
        return int(self.sketch.count)

    def merge(self, other: "CDFSketch") -> None:
        assert self.metric_name == other.metric_name

        self.sketch.merge(other.sketch)

    # add a new datapoint
    def put(self, data: float) -> None:
        self._last_data = data
        self.sketch.add(data)

    # add a new x, y datapoint only for the x value to be discarded
    def put_pair(self, data_x: float, data_y: float) -> None:
        self._last_data = data_y
        self.sketch.add(data_y)

    # add a new datapoint as an incremental (delta) update to
    # recently collected datapoint
    def put_delta(self, delta: float) -> None:
        data = self._last_data + delta
        self.put(data)

    def print_distribution_stats(self, plot_name: str) -> None:
        if self.sketch._count == 0:
            return

        logger.info(
            f"{plot_name}: {self.metric_name} stats:"
            f" min: {self.sketch._min},"
            f" max: {self.sketch._max},"
            f" mean: {self.sketch.avg},"
            f" 25th percentile: {self.sketch.get_quantile_value(0.25)},"
            f" median: {self.sketch.get_quantile_value(0.5)},"
            f" 75th percentile: {self.sketch.get_quantile_value(0.75)},"
            f" 95th percentile: {self.sketch.get_quantile_value(0.95)},"
            f" 99th percentile: {self.sketch.get_quantile_value(0.99)}"
            f" 99.9th percentile: {self.sketch.get_quantile_value(0.999)}"
            f" count: {self.sketch._count}"
            f" sum: {self.sketch.sum}"
        )
        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": self.sketch._min,
                    f"{plot_name}_max": self.sketch._max,
                    f"{plot_name}_mean": self.sketch.avg,
                    f"{plot_name}_25th_percentile": self.sketch.get_quantile_value(
                        0.25
                    ),
                    f"{plot_name}_median": self.sketch.get_quantile_value(0.5),
                    f"{plot_name}_75th_percentile": self.sketch.get_quantile_value(
                        0.75
                    ),
                    f"{plot_name}_95th_percentile": self.sketch.get_quantile_value(
                        0.95
                    ),
                    f"{plot_name}_99th_percentile": self.sketch.get_quantile_value(
                        0.99
                    ),
                    f"{plot_name}_99.9th_percentile": self.sketch.get_quantile_value(
                        0.999
                    ),
                    f"{plot_name}_count": self.sketch.count,
                    f"{plot_name}_sum": self.sketch.sum,
                },
                step=0,
            )

    def to_df(self) -> pd.DataFrame:
        # get quantiles at 1% intervals
        quantiles = np.linspace(0, 1, self._num_quantiles_in_df)
        # get quantile values
        quantile_values = [self.sketch.get_quantile_value(q) for q in quantiles]
        # create dataframe
        df = pd.DataFrame({"cdf": quantiles, self.metric_name: quantile_values})

        return df

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv", index=False)

    def plot_cdf(self, path: str, plot_name: str, x_axis_label: str = None) -> None:

        if self.sketch._count == 0:
            return

        if x_axis_label is None:
            x_axis_label = self.metric_name

        df = self.to_df()

        self.print_distribution_stats(plot_name)

        fig = px.line(
            df, x=self.metric_name, y="cdf", markers=True, labels={"x": x_axis_label}
        )
        fig.update_traces(marker=dict(color="red", size=2))

        if wandb.run:
            wandb_df = df.copy()
            # rename the self.metric_name column to x_axis_label
            wandb_df = wandb_df.rename(columns={self.metric_name: x_axis_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        "cdf",
                        x_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")
        self._save_df(df, path, plot_name)
