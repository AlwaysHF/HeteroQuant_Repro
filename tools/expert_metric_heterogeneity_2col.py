#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle


DEFAULT_OLMOE_SCORES = (
    "/home/lwk/HeteroQuant_Repro/experiments/"
    "final_routed_tail_downstream_group2048_20260717_213819/"
    "olmoe_final_routed_tail_group2048/plans/expert_distribution_scores.csv"
)
DEFAULT_QWEN_SCORES = (
    "/home/lwk/HeteroQuant_Repro/experiments/"
    "final_routed_tail_downstream_group2048_20260717_213819/"
    "qwen_final_routed_tail_group2048/plans/expert_distribution_scores.csv"
)


METRICS = {
    "tail": "tail_score",
    "range": "range_score",
}


def paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def score_ratio_matrix(df, score_col):
    layers = sorted(int(x) for x in df["layer_index"].unique())
    experts = sorted(int(x) for x in df["expert_index"].unique())
    mat = np.zeros((len(layers), len(experts)), dtype=float)
    for row_idx, layer in enumerate(layers):
        group = df[df["layer_index"] == layer].copy()
        mean = max(float(group[score_col].mean()), 1e-12)
        group["score_ratio"] = group[score_col].astype(float) / mean
        group = group.set_index("expert_index")
        for col_idx, expert in enumerate(experts):
            mat[row_idx, col_idx] = float(group.loc[expert, "score_ratio"])
    return layers, experts, mat


def strongest_layer(layers, ratio):
    best_idx = int(np.argmax(np.std(ratio, axis=1)))
    return int(layers[best_idx]), best_idx


def ratio_to_heat(ratio):
    ratio = np.asarray(ratio, dtype=float)
    ratio = np.maximum(ratio, 1e-12)
    return np.log2(ratio)


def style_heatmap_axis(ax):
    ax.tick_params(axis="both", which="both", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_zoom_ticks(ax, experts):
    n = len(experts)
    ticks = np.unique(np.linspace(0, n - 1, min(5, n), dtype=int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(experts[i]) for i in ticks])
    ax.set_yticks([])


def add_row_zoom_box(fig, ax_full, ax_zoom, row_idx, n_experts):
    color = "#D00000"
    rect = Rectangle(
        (-0.5, row_idx - 0.5),
        n_experts,
        1.0,
        fill=False,
        edgecolor=color,
        linewidth=1.05,
        zorder=5,
    )
    ax_full.add_patch(rect)

    for y in (row_idx - 0.5, row_idx + 0.5):
        con = ConnectionPatch(
            xyA=(n_experts - 0.5, y),
            coordsA=ax_full.transData,
            xyB=(-0.5, y - row_idx),
            coordsB=ax_zoom.transData,
            color=color,
            linewidth=0.85,
            linestyle=(0, (4, 3)),
            alpha=0.88,
            zorder=6,
        )
        fig.add_artist(con)


def add_case(
    fig,
    ax_full,
    ax_zoom,
    df,
    model_label,
    metric_label,
    score_col,
    cmap,
    norm,
    show_xlabel=False,
):
    layers, experts, ratio = score_ratio_matrix(df, score_col)
    heat = np.clip(ratio_to_heat(ratio), norm.vmin, norm.vmax)
    zoom_layer, zoom_idx = strongest_layer(layers, ratio)

    im = ax_full.imshow(
        heat,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        rasterized=True,
    )
    ax_full.set_ylabel(f"{model_label}\n{metric_label}", rotation=0, ha="right", va="center", labelpad=20)
    ytick_idx = np.unique(np.linspace(0, len(layers) - 1, min(7, len(layers)), dtype=int))
    ax_full.set_yticks(ytick_idx)
    ax_full.set_yticklabels([str(layers[i]) for i in ytick_idx])
    xticks = np.unique(np.linspace(0, len(experts) - 1, min(5, len(experts)), dtype=int))
    ax_full.set_xticks(xticks)
    ax_full.set_xticklabels([str(experts[i]) for i in xticks])
    if show_xlabel:
        ax_full.set_xlabel("Expert index")
    else:
        ax_full.set_xticklabels([])
    style_heatmap_axis(ax_full)

    zoom = heat[zoom_idx : zoom_idx + 1, :]
    ax_zoom.imshow(
        zoom,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        rasterized=True,
    )
    ax_zoom.set_title(f"Layer {zoom_layer}", pad=2, fontsize=8.6)
    format_zoom_ticks(ax_zoom, experts)
    if show_xlabel:
        ax_zoom.set_xlabel("Expert index")
    else:
        ax_zoom.set_xticklabels([])
    style_heatmap_axis(ax_zoom)
    add_row_zoom_box(fig, ax_full, ax_zoom, zoom_idx, len(experts))

    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--olmoe_scores", default=DEFAULT_OLMOE_SCORES)
    parser.add_argument("--qwen_scores", default=DEFAULT_QWEN_SCORES)
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/expert_metric_heterogeneity_2col")
    parser.add_argument("--vmax_log2", type=float, default=1.0, help="color clipping in log2 ratio units; 1 means 0.5x to 2x")
    args = parser.parse_args()

    paper_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    olmoe = pd.read_csv(args.olmoe_scores)
    qwen = pd.read_csv(args.qwen_scores)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "metric_ratio",
        ["#0B3B75", "#8DB8DA", "#F8F4EB", "#F4A79D", "#C7352A"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-args.vmax_log2, vcenter=0.0, vmax=args.vmax_log2)

    cases = [
        (olmoe, "OLMoE", "Tail", METRICS["tail"]),
        (olmoe, "OLMoE", "Range", METRICS["range"]),
        (qwen, "Qwen", "Tail", METRICS["tail"]),
        (qwen, "Qwen", "Range", METRICS["range"]),
    ]

    fig = plt.figure(figsize=(7.6, 5.65))
    gs = fig.add_gridspec(
        4,
        3,
        width_ratios=[3.55, 1.45, 0.16],
        height_ratios=[1, 1, 1, 1],
        left=0.09,
        right=0.93,
        bottom=0.10,
        top=0.92,
        wspace=0.18,
        hspace=0.34,
    )

    axes_full = []
    axes_zoom = []
    last_im = None
    for idx, (df, model_label, metric_label, score_col) in enumerate(cases):
        ax_full = fig.add_subplot(gs[idx, 0])
        ax_zoom = fig.add_subplot(gs[idx, 1])
        axes_full.append(ax_full)
        axes_zoom.append(ax_zoom)
        last_im = add_case(
            fig,
            ax_full,
            ax_zoom,
            df,
            model_label,
            metric_label,
            score_col,
            cmap,
            norm,
            show_xlabel=(idx == len(cases) - 1),
        )

    fig.text(0.325, 0.965, "Full model", ha="center", va="bottom", fontsize=11.0)
    fig.text(0.650, 0.965, "Zoomed layer", ha="center", va="bottom", fontsize=11.0)

    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(last_im, cax=cax, extend="both")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.5x", "0.71x", "1x", "1.41x", "2x"])
    cbar.set_label("Score / layer mean", rotation=270, labelpad=13)
    cbar.outline.set_linewidth(0.65)

    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"expert_metric_heterogeneity_2col.{ext}",
            dpi=520 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)

    summary_rows = []
    for df, model_label, metric_label, score_col in cases:
        layers, _, ratio = score_ratio_matrix(df, score_col)
        zoom_layer, zoom_idx = strongest_layer(layers, ratio)
        summary_rows.append(
            {
                "model": model_label,
                "metric": metric_label,
                "zoom_layer": zoom_layer,
                "zoom_layer_std": float(np.std(ratio[zoom_idx], ddof=0)),
                "zoom_layer_min": float(np.min(ratio[zoom_idx])),
                "zoom_layer_max": float(np.max(ratio[zoom_idx])),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "expert_metric_heterogeneity_2col_summary.csv", index=False)
    print(f"saved {out_dir / 'expert_metric_heterogeneity_2col.png'}")
    print(f"saved {out_dir / 'expert_metric_heterogeneity_2col.pdf'}")
    print(f"saved {out_dir / 'expert_metric_heterogeneity_2col_summary.csv'}")


if __name__ == "__main__":
    main()
