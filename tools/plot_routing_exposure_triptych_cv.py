#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routing_exposure_revised_v8 import (
    CURVE_LINEWIDTH,
    FIGURE_FACE_COLOR,
    LAYER_COLORS,
    LEGEND_EDGE_COLOR,
    SAVEFIG_FACE_COLOR,
    add_log_bands,
    paper_style,
    plot_exposure_curve,
    style_common_axes,
)


OLMOE_SCORES = (
    "/home/lwk/HeteroQuant_Repro/experiments/"
    "final_routed_tail_downstream_group2048_20260717_213819/"
    "olmoe_final_routed_tail_group2048/plans/expert_distribution_scores.csv"
)
QWEN_SCORES = (
    "/home/lwk/HeteroQuant_Repro/experiments/"
    "final_routed_tail_downstream_group2048_20260717_213819/"
    "qwen_final_routed_tail_group2048/plans/expert_distribution_scores.csv"
)
OUT_DIR = Path("/home/lwk/HeteroQuant_Repro/figures/routing_exposure_triptych")


def normalized_exposure(df, layer, sort=False):
    group = df[df["layer_index"] == layer].copy()
    if group.empty:
        raise ValueError(f"layer {layer} not found")
    group = group.sort_values("expert_index")
    y = group["route_mass_fraction"].to_numpy(dtype=float)
    y = y / max(float(y.mean()), 1e-12)
    x = group["expert_index"].to_numpy(dtype=int)
    if sort:
        y = np.sort(y)[::-1]
        x = np.arange(1, len(y) + 1)
    return x, y


def style_exposure_axis(ax, xlabel):
    ax.set_yscale("log")
    ax.set_ylim(0.025, 5.4)
    ax.set_yticks([0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0])
    ax.set_yticklabels(["0.03x", "0.1x", "0.3x", "0.5x", "1x", "1.5x", "3x"])
    ax.set_xlabel(xlabel)
    style_common_axes(ax)


def add_paper_legend(ax, loc, framealpha=0.96):
    legend = ax.legend(
        loc=loc,
        frameon=True,
        framealpha=framealpha,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
    )
    for handle in legend.get_lines():
        handle.set_alpha(1.0)
        handle.set_linewidth(CURVE_LINEWIDTH + 0.35)
    return legend


def add_panel_caption(ax, text):
    ax.text(
        0.5,
        -0.28,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )


def plot_sorted_panel(ax, df):
    layers = [6, 12, 15]
    colors = LAYER_COLORS
    linestyles = ["-", "--", "-."]
    add_log_bands(ax, 0.025, 5.4)
    for i, layer in enumerate(layers):
        x, y = normalized_exposure(df, layer, sort=True)
        plot_exposure_curve(
            ax,
            x,
            y,
            line_color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"Layer {layer}",
        )
    ax.set_xlim(1, 64)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    style_exposure_axis(ax, "Expert rank")
    add_paper_legend(ax, "lower left")
    add_panel_caption(ax, "(b) Sorted exposure")


def plot_unsorted_panel(ax, df):
    layers = [0, 6, 15]
    colors = LAYER_COLORS
    linestyles = ["-", "--", "-."]
    add_log_bands(ax, 0.025, 5.4)
    for i, layer in enumerate(layers):
        x, y = normalized_exposure(df, layer, sort=False)
        plot_exposure_curve(
            ax,
            x,
            y,
            line_color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"Layer {layer}",
        )
    ax.set_xlim(0, 63)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63])
    ax.set_ylabel("Routing exposure / layer mean")
    style_exposure_axis(ax, "Expert index")
    add_paper_legend(ax, "lower left")
    add_panel_caption(ax, "(a) Expert-index exposure")


def cv_by_layer(df):
    rows = []
    for layer, group in df.groupby("layer_index"):
        vals = group["route_mass_fraction"].to_numpy(dtype=float)
        rows.append({"layer": int(layer), "cv": float(vals.std(ddof=0) / max(vals.mean(), 1e-12))})
    return pd.DataFrame(rows).sort_values("layer")


def plot_cv_panel(ax, olmoe_df, qwen_df):
    olmoe = cv_by_layer(olmoe_df)
    qwen = cv_by_layer(qwen_df)
    olmoe_mean = float(olmoe["cv"].mean())
    qwen_mean = float(qwen["cv"].mean())
    ax.plot(
        olmoe["layer"],
        olmoe["cv"],
        color=LAYER_COLORS[0],
        marker="o",
        markersize=4.1,
        linewidth=CURVE_LINEWIDTH + 1.0,
        markerfacecolor="white",
        markeredgewidth=1.05,
        label="OLMoE",
    )
    ax.plot(
        qwen["layer"],
        qwen["cv"],
        color=LAYER_COLORS[3],
        marker="s",
        markersize=4.1,
        linewidth=CURVE_LINEWIDTH + 1.0,
        markerfacecolor="white",
        markeredgewidth=1.05,
        label="Qwen-MoE",
    )
    ax.axhline(
        olmoe_mean,
        color=LAYER_COLORS[0],
        linewidth=CURVE_LINEWIDTH + 0.35,
        linestyle=(0, (4, 3)),
        alpha=0.78,
        label="OLMoE mean",
    )
    ax.axhline(
        qwen_mean,
        color=LAYER_COLORS[3],
        linewidth=CURVE_LINEWIDTH + 0.35,
        linestyle=(0, (4, 3)),
        alpha=0.78,
        label="Qwen-MoE mean",
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Exposure CV")
    ax.set_xlim(-0.5, max(qwen["layer"].max(), olmoe["layer"].max()) + 0.5)
    ax.set_ylim(0.15, 0.72)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    style_common_axes(ax)
    add_paper_legend(ax, "upper left", framealpha=0.72)
    add_panel_caption(ax, "(c) Layer-wise imbalance")
    return pd.concat(
        [
            olmoe.assign(model="OLMoE"),
            qwen.assign(model="Qwen-MoE"),
        ],
        ignore_index=True,
    )


def main():
    paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    olmoe_df = pd.read_csv(OLMOE_SCORES)
    qwen_df = pd.read_csv(QWEN_SCORES)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 3.85),
        facecolor=FIGURE_FACE_COLOR,
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.0]},
    )
    plot_unsorted_panel(axes[0], olmoe_df)
    plot_sorted_panel(axes[1], olmoe_df)
    summary = plot_cv_panel(axes[2], olmoe_df, qwen_df)
    fig.tight_layout(w_pad=1.05, pad=0.45, rect=(0.0, 0.10, 1.0, 0.98))

    png = OUT_DIR / "routing_exposure_triptych_cv.png"
    pdf = OUT_DIR / "routing_exposure_triptych_cv.pdf"
    csv = OUT_DIR / "routing_exposure_triptych_cv_summary.csv"
    fig.savefig(png, dpi=420, bbox_inches="tight", facecolor=SAVEFIG_FACE_COLOR)
    fig.savefig(pdf, bbox_inches="tight", facecolor=SAVEFIG_FACE_COLOR)
    summary.to_csv(csv, index=False)
    print(f"saved png: {png}")
    print(f"saved pdf: {pdf}")
    print(f"saved summary: {csv}")


if __name__ == "__main__":
    main()
