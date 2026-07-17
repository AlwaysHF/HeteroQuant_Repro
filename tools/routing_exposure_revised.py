#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCORES = (
    "/home/lwk/HeteroQuant_Repro/experiments/"
    "metric_ablation_olmoe_ppl_group2048_20260716_101427/"
    "routed_tail/plans/expert_distribution_scores.csv"
)


def sorted_norm_exposure(df, layer):
    group = df[df["layer_index"] == layer]
    if group.empty:
        raise ValueError(f"layer {layer} not found")
    vals = group["route_mass_fraction"].to_numpy(dtype=float)
    vals = vals / max(float(vals.mean()), 1e-12)
    return np.sort(vals)[::-1]


def expert_id_norm_exposure(df, layer):
    group = df[df["layer_index"] == layer].copy()
    if group.empty:
        raise ValueError(f"layer {layer} not found")
    group = group.sort_values("expert_index")
    vals = group["route_mass_fraction"].to_numpy(dtype=float)
    vals = vals / max(float(vals.mean()), 1e-12)
    experts = group["expert_index"].to_numpy(dtype=int)
    return experts, vals


HIGH_THRESHOLD = 1.5
LOW_THRESHOLD = 0.5
CURVE_LINEWIDTH = 1.20
EXTREME_MARKER_SIZE = 13.0
EXTREME_MARKER_WIDTH = 0.75


def add_extreme_markers(ax, x, y, color, high=HIGH_THRESHOLD, low=LOW_THRESHOLD):
    """Mark only points in the high- and low-exposure regions."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x)
    mask = (y > high) | (y < low)
    if np.any(mask):
        ax.scatter(
            x[mask],
            y[mask],
            s=EXTREME_MARKER_SIZE,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=EXTREME_MARKER_WIDTH,
            zorder=4,
            label="_nolegend_",
        )


def add_log_bands(ax, ymin, ymax):
    # Three solid, pale background regions.
    ax.axhspan(HIGH_THRESHOLD, ymax, color="#EAF2FF", alpha=0.92, linewidth=0, zorder=-10)
    ax.axhspan(LOW_THRESHOLD, HIGH_THRESHOLD, color="#F3F4F6", alpha=0.92, linewidth=0, zorder=-10)
    ax.axhspan(ymin, LOW_THRESHOLD, color="#FFF1E8", alpha=0.92, linewidth=0, zorder=-10)

    ax.axhline(1.0, color="#7C8491", linewidth=0.9, linestyle=(0, (4, 3)), label="Layer mean", zorder=-1)
    ax.axhline(HIGH_THRESHOLD, color="#C8D2DF", linewidth=0.7, alpha=0.85, zorder=-1)
    ax.axhline(LOW_THRESHOLD, color="#D8CFC8", linewidth=0.7, alpha=0.85, zorder=-1)

    tx = ax.get_yaxis_transform()
    ax.text(0.018, 3.9, "High\nexposure", transform=tx, ha="left", va="center", color="#315A8C", fontsize=9.2)
    ax.text(0.018, 0.86, "Near\nmean", transform=tx, ha="left", va="center", color="#5F6772", fontsize=9.2)
    ax.text(0.018, 0.23, "Low\nexposure", transform=tx, ha="left", va="center", color="#9A5A2D", fontsize=9.2)


def add_log2_bands(ax, ymin, ymax):
    # y is log2(exposure). These spans correspond to exposure >1.5, 0.5-1.5, and <0.5.
    hi = math.log2(HIGH_THRESHOLD)
    lo = math.log2(LOW_THRESHOLD)
    ax.axhspan(hi, ymax, color="#EAF2FF", alpha=0.92, linewidth=0, zorder=-10)
    ax.axhspan(lo, hi, color="#F3F4F6", alpha=0.92, linewidth=0, zorder=-10)
    ax.axhspan(ymin, lo, color="#FFF1E8", alpha=0.92, linewidth=0, zorder=-10)

    ax.axhline(0.0, color="#7C8491", linewidth=0.9, linestyle=(0, (4, 3)), label="Layer mean", zorder=-1)
    ax.axhline(hi, color="#C8D2DF", linewidth=0.7, alpha=0.85, zorder=-1)
    ax.axhline(lo, color="#D8CFC8", linewidth=0.7, alpha=0.85, zorder=-1)

    tx = ax.get_yaxis_transform()
    ax.text(0.018, 1.72, "High\nexposure", transform=tx, ha="left", va="center", color="#315A8C", fontsize=9.2)
    ax.text(0.018, 0.0, "Near\nmean", transform=tx, ha="left", va="center", color="#5F6772", fontsize=9.2)
    ax.text(0.018, -2.15, "Low\nexposure", transform=tx, ha="left", va="center", color="#9A5A2D", fontsize=9.2)


def paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax, legend_loc="lower left"):
    ax.set_xlim(1, 64)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    ax.set_xlabel("Expert rank within each layer (sorted)")
    ax.grid(axis="y", which="major", color="#D8DEE9", linewidth=0.75, alpha=0.76)
    ax.grid(axis="x", which="major", color="#EDF0F5", linewidth=0.65, alpha=0.70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.94, edgecolor="#D1D5DB")


def plot_log_variant(df, out_dir, name, layers, title, ylim=(0.025, 5.4), legend_loc="lower left"):
    colors = ["#1B4D89", "#E76F51", "#5C677D", "#2A9D8F", "#7B2CBF"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))]

    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    add_log_bands(ax, ylim[0], ylim[1])
    for i, layer in enumerate(layers):
        y = sorted_norm_exposure(df, layer)
        x = np.arange(1, len(y) + 1)
        ax.plot(
            x,
            y,
            color=colors[i % len(colors)],
            linewidth=CURVE_LINEWIDTH,
            linestyle=linestyles[i % len(linestyles)],
            label=f"Layer {layer}",
            zorder=2,
        )
        add_extreme_markers(ax, x, y, colors[i % len(colors)])
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks([0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0])
    ax.set_yticklabels(["0.03x", "0.1x", "0.3x", "0.5x", "1x", "1.5x", "3x"])
    ax.set_ylabel("Routing exposure / layer mean")
    ax.set_title(title)
    finish_axes(ax, legend_loc)
    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_unsorted_log_variant(df, out_dir, name, layers):
    colors = ["#1B4D89", "#E76F51", "#5C677D", "#2A9D8F", "#7B2CBF"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))]
    ylim = (0.025, 5.4)

    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    add_log_bands(ax, ylim[0], ylim[1])
    for i, layer in enumerate(layers):
        x, y = expert_id_norm_exposure(df, layer)
        ax.plot(
            x,
            y,
            color=colors[i % len(colors)],
            linewidth=CURVE_LINEWIDTH,
            linestyle=linestyles[i % len(linestyles)],
            label=f"Layer {layer}",
            zorder=2,
        )
        add_extreme_markers(ax, x, y, colors[i % len(colors)])
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 63)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63])
    ax.set_yticks([0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0])
    ax.set_yticklabels(["0.03x", "0.1x", "0.3x", "0.5x", "1x", "1.5x", "3x"])
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Routing exposure / layer mean")
    ax.set_title("Routing Exposure by Expert Index")
    ax.grid(axis="y", which="major", color="#D8DEE9", linewidth=0.75, alpha=0.76)
    ax.grid(axis="x", which="major", color="#EDF0F5", linewidth=0.65, alpha=0.70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, framealpha=0.94, edgecolor="#D1D5DB")
    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_log2_variant(df, out_dir, name, layers):
    colors = ["#1B4D89", "#E76F51", "#5C677D"]
    linestyles = ["-", "--", "-."]
    ymin, ymax = -5.2, 2.2

    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    add_log2_bands(ax, ymin, ymax)
    for i, layer in enumerate(layers):
        y = np.log2(sorted_norm_exposure(df, layer))
        x = np.arange(1, len(y) + 1)
        ax.plot(
            x,
            y,
            color=colors[i],
            linewidth=CURVE_LINEWIDTH,
            linestyle=linestyles[i],
            label=f"Layer {layer}",
            zorder=2,
        )
        hi = math.log2(HIGH_THRESHOLD)
        lo = math.log2(LOW_THRESHOLD)
        mask = (y > hi) | (y < lo)
        if np.any(mask):
            ax.scatter(
                x[mask],
                y[mask],
                s=EXTREME_MARKER_SIZE,
                marker="o",
                facecolors="none",
                edgecolors=colors[i],
                linewidths=EXTREME_MARKER_WIDTH,
                zorder=4,
                label="_nolegend_",
            )
    ax.set_ylim(ymin, ymax)
    ticks = [-5, -4, -3, -2, -1, 0, 1, 2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{2 ** t:g}x" for t in ticks])
    ax.set_ylabel("Routing exposure / layer mean")
    ax.set_title("Sorted Routing Exposure Across Experts")
    finish_axes(ax, "lower left")
    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", default=DEFAULT_SCORES)
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/routing_exposure_variants")
    args = parser.parse_args()

    paper_style()
    df = pd.read_csv(args.scores_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("routing_exposure_3layers_bands", [0, 6, 15], "Sorted Routing Exposure Across Experts"),
        ("routing_exposure_3layers_extreme", [6, 12, 15], "Routing Exposure Long Tail"),
        ("routing_exposure_5layers_bands", [0, 4, 6, 12, 15], "Sorted Routing Exposure Across Experts"),
    ]
    for name, layers, title in variants:
        plot_log_variant(df, out_dir, name, layers, title)
        print(f"saved {name}: layers={layers}")

    plot_log2_variant(df, out_dir, "routing_exposure_3layers_log2", [0, 6, 15])
    print("saved routing_exposure_3layers_log2: layers=[0, 6, 15]")

    plot_unsorted_log_variant(df, out_dir, "routing_exposure_3layers_unsorted_bands", [0, 6, 15])
    print("saved routing_exposure_3layers_unsorted_bands: layers=[0, 6, 15]")

    plot_unsorted_log_variant(df, out_dir, "routing_exposure_5layers_unsorted_bands", [0, 4, 6, 12, 15])
    print("saved routing_exposure_5layers_unsorted_bands: layers=[0, 4, 6, 12, 15]")

    summary_rows = []
    for layer in sorted(df["layer_index"].unique()):
        y = sorted_norm_exposure(df, int(layer))
        summary_rows.append(
            {
                "layer": int(layer),
                "max_norm": float(y.max()),
                "min_norm": float(y.min()),
                "cv_norm": float(y.std(ddof=0)),
                "top10_mean": float(y[:10].mean()),
                "bottom10_mean": float(y[-10:].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "routing_exposure_layer_summary.csv", index=False)
    print(f"output_dir: {out_dir}")


if __name__ == "__main__":
    main()
