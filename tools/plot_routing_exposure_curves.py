#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_layers(value):
    layers = []
    for item in value.split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    if not layers:
        raise ValueError("at least one layer is required")
    return layers


def choose_top_cv_layers(df, k):
    stats = []
    for layer, group in df.groupby("layer_index"):
        vals = group["route_mass_fraction"].to_numpy(dtype=float)
        mean = float(vals.mean())
        cv = float(vals.std(ddof=0) / max(mean, 1e-12))
        stats.append((cv, int(layer)))
    stats.sort(reverse=True)
    return [layer for _, layer in stats[:k]]


def main():
    parser = argparse.ArgumentParser(description="Plot sorted routing exposure curves for selected MoE layers.")
    parser.add_argument(
        "--scores_csv",
        default="/home/lwk/HeteroQuant_Repro/experiments/metric_ablation_olmoe_ppl_group2048_20260716_101427/routed_tail/plans/expert_distribution_scores.csv",
    )
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/routing_exposure")
    parser.add_argument("--layers", default="0,4,6,12,15")
    parser.add_argument("--top_cv", type=int, default=0, help="override --layers with top-k layers by CV")
    parser.add_argument("--prefix", default="routing_exposure_sorted")
    args = parser.parse_args()

    df = pd.read_csv(args.scores_csv)
    required = {"layer_index", "expert_index", "route_mass_fraction"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if args.top_cv > 0:
        layers = choose_top_cv_layers(df, args.top_cv)
    else:
        layers = parse_layers(args.layers)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compact paper style: restrained palette, no chart junk, readable after column scaling.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = ["#1B4D89", "#2A9D8F", "#E76F51", "#7B2CBF", "#5C677D"]
    fig, ax = plt.subplots(figsize=(7.0, 4.25))

    summary_rows = []
    for idx, layer in enumerate(layers):
        group = df[df["layer_index"] == layer].copy()
        if group.empty:
            raise ValueError(f"layer {layer} not found in {args.scores_csv}")
        vals = group["route_mass_fraction"].to_numpy(dtype=float)
        mean = vals.mean()
        norm = np.sort(vals / max(mean, 1e-12))[::-1]
        ranks = np.arange(1, len(norm) + 1)
        cv = float(norm.std(ddof=0))
        top = float(norm[0])
        bottom = float(norm[-1])
        summary_rows.append(
            {
                "layer": int(layer),
                "max_norm_exposure": top,
                "min_norm_exposure": bottom,
                "cv_norm_exposure": cv,
                "top10_mean_norm_exposure": float(norm[:10].mean()),
                "bottom10_mean_norm_exposure": float(norm[-10:].mean()),
            }
        )

        ax.plot(
            ranks,
            norm,
            color=colors[idx % len(colors)],
            linewidth=2.25,
            solid_capstyle="round",
            label=f"Layer {layer}",
        )

    ax.axhline(
        1.0,
        color="#8A8F98",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=0,
        label="Layer mean",
    )

    ax.set_yscale("log")
    ax.set_xlim(1, 68)
    ax.set_ylim(0.02, 5.2)
    ax.set_xlabel("Expert rank within each layer (sorted by routing exposure)")
    ax.set_ylabel("Routing exposure / layer mean")
    ax.set_title("Sorted Routing Exposure Across Experts")
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    ax.set_yticks([0.03, 0.1, 0.3, 1.0, 3.0])
    ax.set_yticklabels(["0.03x", "0.1x", "0.3x", "1x", "3x"])
    ax.grid(axis="y", which="major", color="#D8DEE9", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", which="major", color="#EDF0F5", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#D1D5DB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.7)

    png_path = out_dir / f"{args.prefix}.png"
    pdf_path = out_dir / f"{args.prefix}.pdf"
    csv_path = out_dir / f"{args.prefix}_summary.csv"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print(f"layers: {layers}")
    print(f"saved png: {png_path}")
    print(f"saved pdf: {pdf_path}")
    print(f"saved summary: {csv_path}")


if __name__ == "__main__":
    main()
