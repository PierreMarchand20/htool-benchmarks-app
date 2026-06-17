from enum import Enum, auto

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px


class ReferenceCurves(Enum):
    InvN = auto()
    NLog2N = auto()
    NLogN = auto()


pn.extension("plotly")


def make_dashboard(
    data: pd.DataFrame,
    title: str,
    x_axis: str,
    filters: list[str],
    metrics: list[str],
    reference_curves: dict[str, list[ReferenceCurves]],
    defaults: dict[str, str],
):
    widgets = dict()

    for filter in filters:
        options = sorted(data[filter].dropna().unique())
        widgets[filter] = pn.widgets.MultiChoice(
            name=filter,
            options=options,
            value=defaults.get(filter, options),
        )
    metric = pn.widgets.Select(
        name="metric",
        options=metrics,
    )

    @pn.depends(
        *(widgets.values()),
        metric,
    )
    def plot_bench(*values):
        *filter_values, metric_value = values

        # Build filtering mask dynamically
        mask = pd.Series(True, index=data.index)

        for col, selected in zip(filters, filter_values):
            mask &= data[col].isin(selected)

        filtered = data[mask].copy()

        if filtered.empty:
            return pn.pane.Markdown("No data selected.")

        filtered["curve"] = (
            filtered[filters]
            .astype(str)
            .agg(
                " | ".join,
                axis=1,
            )
        )

        fig = px.line(
            filtered.sort_values(x_axis),
            x=x_axis,
            y=metric_value,
            color="curve",
            markers=True,
            log_x=True,
            log_y=True,
        )

        # --- reference curve ---
        if metric_value in reference_curves.keys():
            filtered = filtered.sort_values(x_axis)
            for reference_curve in reference_curves[metric_value]:
                if reference_curve == ReferenceCurves.NLog2N:
                    ref_nlogn = (
                        np.log(filtered[x_axis])
                        * np.log(filtered[x_axis])
                        * filtered[x_axis]
                        * filtered[metric_value].iloc[0]
                        / (
                            np.log(filtered[x_axis].iloc[0])
                            * np.log(filtered[x_axis].iloc[0])
                            * filtered[x_axis].iloc[0]
                        )
                    )
                    fig.add_scatter(
                        x=filtered[x_axis],
                        y=ref_nlogn,
                        mode="lines",
                        name="O(nlog^2(n)) reference",
                        line=dict(dash="dash", width=3),
                    )
                elif reference_curve == ReferenceCurves.NLogN:
                    ref_nlogn = (
                        np.log(filtered[x_axis])
                        * filtered[x_axis]
                        * filtered[metric_value].iloc[0]
                        / (np.log(filtered[x_axis].iloc[0]) * filtered[x_axis].iloc[0])
                    )
                    fig.add_scatter(
                        x=filtered[x_axis],
                        y=ref_nlogn,
                        mode="lines",
                        name="O(nlog(n)) reference",
                        line=dict(dash="dash", width=3),
                    )
                elif reference_curve == ReferenceCurves.InvN:
                    ref_invn = (
                        filtered[x_axis].iloc[0]
                        / filtered[x_axis]
                        * filtered[metric_value].iloc[0]
                    )
                    fig.add_scatter(
                        x=filtered[x_axis],
                        y=ref_invn,
                        mode="lines",
                        name="O(1/n) reference",
                        line=dict(dash="dash", width=3),
                    )

        fig.update_layout(
            height=700,
            legend_title="Configuration",
        )

        return pn.pane.Plotly(
            fig,
            sizing_mode="stretch_width",
            height=600,
        )

    filter_box = pn.WidgetBox(
        "### Filters",
        metric,
        *widgets.values(),
        width=320,
    )

    return pn.Card(
        pn.Row(
            filter_box,
            plot_bench,
            sizing_mode="stretch_width",
        ),
        title=title,
        collapsible=False,
    )


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

data_build_vs_pbl_size = pd.read_csv("data/bench_hmatrix_build_vs_pbl_size.csv")
data_build_vs_thread = pd.read_csv("data/bench_hmatrix_build_vs_thread.csv")

data_product_vs_pbl_size = pd.read_csv(
    "data/bench_hmatrix_matrix_product_vs_pbl_size.csv"
)
data_product_vs_thread = pd.read_csv("data/bench_hmatrix_matrix_product_vs_thread.csv")

data_facto_vs_pbl_size = pd.read_csv("data/bench_hmatrix_factorization_vs_pbl_size.csv")
data_facto_vs_thread = pd.read_csv("data/bench_hmatrix_factorization_vs_thread.csv")

data_build_vs_pbl_size = data_build_vs_pbl_size[
    data_build_vs_pbl_size["id_rep"] == "mean"
]
data_build_vs_thread = data_build_vs_thread[data_build_vs_thread["id_rep"] == "mean"]

data_facto_vs_pbl_size = data_facto_vs_pbl_size[
    data_facto_vs_pbl_size["id_rep"] == "mean"
]
data_facto_vs_thread = data_facto_vs_thread[data_facto_vs_thread["id_rep"] == "mean"]

data_product_vs_pbl_size = data_product_vs_pbl_size[
    data_product_vs_pbl_size["id_rep"] == "mean"
]
data_product_vs_thread = data_product_vs_thread[
    data_product_vs_thread["id_rep"] == "mean"
]

# data_product_vs_pbl_size = data_build_vs_pbl_size
# data_product_vs_thread = data_build_vs_thread

# data_facto_vs_pbl_size = data_build_vs_pbl_size
# data_facto_vs_thread = data_build_vs_thread

metrics = [
    "time",
    "compression_ratio",
    "space_saving",
]

filters_for_size_scaling = [
    "epsilon",
    "policy_type",
    "generator_type",
    "symmetry_type",
    "low_rank_generator_type",
    "clustering_type",
    "number_of_threads",
    "hardware_type",
    "version",
]

filters_for_thread_scaling = [
    "epsilon",
    "policy_type",
    "generator_type",
    "symmetry_type",
    "low_rank_generator_type",
    "clustering_type",
    "size",
    "hardware_type",
    "version",
]

# ------------------------------------------------------------------
# Dashboards
# ------------------------------------------------------------------

dashboard_build_vs_size = make_dashboard(
    data=data_build_vs_pbl_size,
    title="Scaling of assembly with sizes",
    x_axis="size",
    filters=filters_for_size_scaling,
    metrics=metrics,
    reference_curves={"time": [ReferenceCurves.NLog2N]},
    defaults={},
)

dashboard_build_vs_thread = make_dashboard(
    data=data_build_vs_thread,
    title="Scaling of assembly with threads",
    x_axis="number_of_threads",
    filters=filters_for_thread_scaling,
    metrics=metrics,
    reference_curves={"time": [ReferenceCurves.InvN]},
    defaults={},
)

dashboard_facto_vs_size = make_dashboard(
    data=data_facto_vs_pbl_size,
    title="Scaling of factorization with sizes",
    x_axis="size",
    filters=filters_for_size_scaling,
    metrics=[
        "factorization_time",
        "compression_ratio",
        "space_saving",
    ],
    reference_curves={"factorization_time": [ReferenceCurves.NLog2N]},
    defaults={},
)

dashboard_facto_vs_thread = make_dashboard(
    data=data_facto_vs_thread,
    title="Scaling of factorization with threads",
    x_axis="number_of_threads",
    filters=filters_for_thread_scaling,
    metrics=[
        "factorization_time",
        "compression_ratio",
        "space_saving",
    ],
    reference_curves={"factorization_time": [ReferenceCurves.InvN]},
    defaults={},
)

dashboard_product_vs_size = make_dashboard(
    data=data_product_vs_pbl_size,
    title="Scaling of 30 hmatrix vector with sizes",
    x_axis="size",
    filters=filters_for_size_scaling,
    metrics=metrics,
    reference_curves={"time": [ReferenceCurves.NLogN]},
    defaults={},
)

dashboard_product_vs_thread = make_dashboard(
    data=data_product_vs_thread,
    title="Scaling of 30 hmatrix vector with threads",
    x_axis="number_of_threads",
    filters=filters_for_thread_scaling,
    metrics=metrics,
    reference_curves={"time": [ReferenceCurves.InvN]},
    defaults={},
)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------

assembly_tab = pn.Tabs(
    ("Scaling with Size", dashboard_build_vs_size),
    ("Scaling with Threads", dashboard_build_vs_thread),
    dynamic=True,
)

factorization_tab = pn.Tabs(
    ("Scaling with Size", dashboard_facto_vs_size),
    ("Scaling with Threads", dashboard_facto_vs_thread),
    dynamic=True,
)

product_tab = pn.Tabs(
    ("Scaling with Size", dashboard_product_vs_size),
    ("Scaling with Threads", dashboard_product_vs_thread),
    dynamic=True,
)

tabs = pn.Tabs(
    ("Assembly", assembly_tab),
    ("Factorization", factorization_tab),
    ("Product", product_tab),
    dynamic=True,
)

# ------------------------------------------------------------------
# Template
# ------------------------------------------------------------------

template = pn.template.FastListTemplate(
    title="Htool-DDM Benchmark Explorer",
    accent_base_color="#1f77b4",
    header_background="#1f77b4",
    main=[tabs],
)

template.servable()
