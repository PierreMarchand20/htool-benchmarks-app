import pandas as pd
import panel as pn
import plotly.express as px

pn.extension("plotly")


def make_dashboard(data, title, defaults):

    eps_opts = sorted(data["epsilon"].dropna().unique())
    pol_opts = sorted(data["policy_type"].dropna().unique())
    gen_opts = sorted(data["generator_type"].dropna().unique())
    lr_opts = sorted(data["low_rank_generator_type"].dropna().unique())
    clust_opts = sorted(data["clustering_type"].dropna().unique())
    thread_opts = sorted(data["number_of_threads"].dropna().unique())

    epsilon = pn.widgets.MultiChoice(
        name="epsilon",
        options=eps_opts,
        value=eps_opts,
    )

    policy = pn.widgets.MultiChoice(
        name="policy_type",
        options=pol_opts,
        value=defaults.get("policy_type", pol_opts),
    )

    generator = pn.widgets.MultiChoice(
        name="generator_type",
        options=sorted(gen_opts),
        value=defaults.get("generator_type", gen_opts),
    )

    low_rank = pn.widgets.MultiChoice(
        name="low_rank_generator_type",
        options=lr_opts,
        value=defaults.get("low_rank_generator_type", lr_opts),
    )

    clustering = pn.widgets.MultiChoice(
        name="clustering_type",
        options=clust_opts,
        value=defaults.get("clustering_type", clust_opts),
    )

    threads = pn.widgets.MultiChoice(
        name="number_of_threads",
        options=thread_opts,
        value=defaults.get("number_of_threads", thread_opts),
    )

    metric = pn.widgets.Select(
        name="metric",
        options=[
            "time (s)",
            "compression_ratio",
            "space_saving",
        ],
    )

    @pn.depends(
        epsilon,
        policy,
        generator,
        low_rank,
        clustering,
        threads,
        metric,
    )
    def plot_bench(
        epsilon,
        policy,
        generator,
        low_rank,
        clustering,
        threads,
        metric,
    ):

        filtered = data[
            data["epsilon"].isin(epsilon)
            & data["policy_type"].isin(policy)
            & data["generator_type"].isin(generator)
            & data["low_rank_generator_type"].isin(low_rank)
            & data["clustering_type"].isin(clustering)
            & data["number_of_threads"].isin(threads)
        ].copy()

        if filtered.empty:
            return pn.pane.Markdown("No data selected.")

        # One line per configuration
        curve_cols = [
            "epsilon",
            "policy_type",
            "generator_type",
            "low_rank_generator_type",
            "clustering_type",
            "number_of_threads",
        ]

        filtered["curve"] = (
            filtered[curve_cols]
            .astype(str)
            .agg(
                " | ".join,
                axis=1,
            )
        )

        fig = px.line(
            filtered.sort_values("size"),
            x="size",
            y=metric,
            color="curve",
            markers=True,
            log_x=True,
            log_y=True,
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

    sidebar = pn.Column(
        "## Filters",
        metric,
        epsilon,
        policy,
        generator,
        low_rank,
        clustering,
        threads,
        width=350,
    )

    return pn.Column(
        f"## {title}",
        pn.Row(
            sidebar,
            plot_bench,
        ),
    )


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

data = pd.read_csv("data/bench_hmatrix_build_vs_pbl_size.csv")

data = data[data["id_rep"] == "mean"].copy()

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

dashboard_build = make_dashboard(
    data,
    title="HMatrix assembly",
    defaults={},
)

dashboard_facto = make_dashboard(
    data,
    title="HMatrix factorization",
    defaults={},
)

app = pn.Tabs(
    ("HMatrix assembly", dashboard_build),
    ("HMatrix factorization", dashboard_facto),
)

app.servable()
