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


# =====================================================
# Main style knobs
# =====================================================
HIGH_THRESHOLD = 1.5
LOW_THRESHOLD = 0.5
CURVE_LINEWIDTH = 0.8
MID_LIGHTEN_AMOUNT = 0.68
EXTREME_MARKER_SIZE = 32.0
EXTREME_MARKER_HALO_SIZE = 50.0
EXTREME_MARKER_WIDTH = 1.05

# =====================================================
# Color control section
# Modify these variables directly to adjust colors anywhere.
# =====================================================
# Background bands
HIGH_BAND_COLOR = "#F8D8D1"
MID_BAND_COLOR = "#FFFEFC"
LOW_BAND_COLOR = "#D4DEEA"

# Band label colors
HIGH_TEXT_COLOR = "#A5483A"
LOW_TEXT_COLOR = "#36597C"

# Axes / frame / legend colors
AXES_FACE_COLOR = "#FDFBF7"
SPINE_COLOR = "#B8B1A6"
LEGEND_EDGE_COLOR = "#CFC6B8"
FIGURE_FACE_COLOR = "white"
SAVEFIG_FACE_COLOR = "white"

# Layer line colors (kept as the original semantic line colors)
LAYER_COLORS = [
    "#70757A",  # Layer 0 style color
    "#9AA0A6",  # Layer 6 style color
    "#C1C5CA",  # Layer 15 style color
    "#3C8D7B",
    "#7A5AA6",
]

# High / low highlighted marker colors (independent from line colors)
HIGH_MARKER_FACE_COLOR = "#D86F60"
HIGH_MARKER_EDGE_COLOR = "#EEB1A6"
LOW_MARKER_FACE_COLOR = "#4875AA"
LOW_MARKER_EDGE_COLOR = "#A5BCD8"
MARKER_HALO_COLOR = "white"

# Optional: if you want high / low markers to instead follow layer colors,
# set USE_SEMANTIC_MARKER_COLORS = False.
USE_SEMANTIC_MARKER_COLORS = True


def lighten_color(color, amount=MID_LIGHTEN_AMOUNT):
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    white = np.ones(3)
    return tuple(np.clip((1.0 - amount) * rgb + amount * white, 0.0, 1.0))


def add_extreme_markers(ax, x, y, line_color, high=HIGH_THRESHOLD, low=LOW_THRESHOLD):
    """Highlight high/low exposure experts with configurable marker colors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x)

    high_mask = y > high
    low_mask = y < low

    if USE_SEMANTIC_MARKER_COLORS:
        high_face, high_edge = HIGH_MARKER_FACE_COLOR, HIGH_MARKER_EDGE_COLOR
        low_face, low_edge = LOW_MARKER_FACE_COLOR, LOW_MARKER_EDGE_COLOR
    else:
        # Follow the current layer color.
        high_face = low_face = line_color
        high_edge = low_edge = line_color

    for mask, face_color, edge_color in [
        (high_mask, high_face, high_edge),
        (low_mask, low_face, low_edge),
    ]:
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=EXTREME_MARKER_HALO_SIZE,
            marker="o",
            facecolors=MARKER_HALO_COLOR,
            edgecolors=MARKER_HALO_COLOR,
            linewidths=0.0,
            alpha=0.95,
            zorder=4.5,
            label="_nolegend_",
        )
        ax.scatter(
            x[mask],
            y[mask],
            s=EXTREME_MARKER_SIZE,
            marker="o",
            facecolors=face_color,
            edgecolors=edge_color,
            linewidths=EXTREME_MARKER_WIDTH,
            alpha=1.0,
            zorder=5,
            label="_nolegend_",
        )


def _interpolate_y_from_plot_coordinate(z, scale):
    if scale == "log":
        return float(np.exp(z))
    return float(z)


def _plot_coordinate(y, scale):
    y = np.asarray(y, dtype=float)
    if scale == "log":
        if np.any(y <= 0):
            raise ValueError("log-scale exposure values must be positive")
        return np.log(y)
    return y


def split_curve_by_regions(x, y, low, high, scale="log"):
    """Split a polyline exactly where it crosses low/high thresholds.

    Returns a list of ``(region, xs, ys)`` where region is one of
    ``high``, ``middle``, or ``low``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if len(x) < 2:
        return []

    z = _plot_coordinate(y, scale)
    low_z = float(_plot_coordinate(np.asarray([low]), scale)[0])
    high_z = float(_plot_coordinate(np.asarray([high]), scale)[0])
    thresholds = (low_z, high_z)

    atomic = []
    for i in range(len(x) - 1):
        x0, x1 = float(x[i]), float(x[i + 1])
        z0, z1 = float(z[i]), float(z[i + 1])

        cuts = [0.0, 1.0]
        if z1 != z0:
            for threshold in thresholds:
                t = (threshold - z0) / (z1 - z0)
                if 0.0 < t < 1.0:
                    cuts.append(float(t))
        cuts = sorted(set(cuts))

        for ta, tb in zip(cuts[:-1], cuts[1:]):
            xa = x0 + ta * (x1 - x0)
            xb = x0 + tb * (x1 - x0)
            za = z0 + ta * (z1 - z0)
            zb = z0 + tb * (z1 - z0)
            zmid = 0.5 * (za + zb)
            if zmid > high_z:
                region = "high"
            elif zmid < low_z:
                region = "low"
            else:
                region = "middle"
            ya = _interpolate_y_from_plot_coordinate(za, scale)
            yb = _interpolate_y_from_plot_coordinate(zb, scale)
            atomic.append((region, [xa, xb], [ya, yb]))

    merged = []
    for region, xs, ys in atomic:
        if merged and merged[-1][0] == region:
            merged[-1][1].append(xs[-1])
            merged[-1][2].append(ys[-1])
        else:
            merged.append([region, list(xs), list(ys)])
    return [(region, np.asarray(xs), np.asarray(ys)) for region, xs, ys in merged]


def plot_exposure_curve(
    ax,
    x,
    y,
    line_color,
    linestyle,
    label,
    high=HIGH_THRESHOLD,
    low=LOW_THRESHOLD,
    scale="log",
):
    """Keep the original layer line color. Only the middle region is faded."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    middle_color = lighten_color(line_color)

    for region, xs, ys in split_curve_by_regions(x, y, low, high, scale=scale):
        color = middle_color if region == "middle" else line_color
        width = CURVE_LINEWIDTH if region == "middle" else CURVE_LINEWIDTH + 0.12
        zorder = 2 if region == "middle" else 3

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            alpha=1.0,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=zorder,
            label="_nolegend_",
        )

    # Legend uses the original saturated line color.
    ax.plot(
        [],
        [],
        color=line_color,
        linewidth=CURVE_LINEWIDTH + 0.20,
        linestyle=linestyle,
        alpha=1.0,
        label=label,
    )

    add_extreme_markers(ax, x, y, line_color=line_color, high=high, low=low)


def add_log_bands(ax, ymin, ymax):
    ax.axhspan(HIGH_THRESHOLD, ymax, color=HIGH_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)
    ax.axhspan(LOW_THRESHOLD, HIGH_THRESHOLD, color=MID_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)
    ax.axhspan(ymin, LOW_THRESHOLD, color=LOW_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)

    tx = ax.get_yaxis_transform()
    ax.text(
        0.018,
        3.9,
        "High\nexposure",
        transform=tx,
        ha="left",
        va="center",
        color=HIGH_TEXT_COLOR,
        fontsize=9.2,
        fontweight="semibold",
    )
    ax.text(
        0.018,
        0.23,
        "Low\nexposure",
        transform=tx,
        ha="left",
        va="center",
        color=LOW_TEXT_COLOR,
        fontsize=9.2,
        fontweight="semibold",
    )


def add_log2_bands(ax, ymin, ymax):
    hi = math.log2(HIGH_THRESHOLD)
    lo = math.log2(LOW_THRESHOLD)
    ax.axhspan(hi, ymax, color=HIGH_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)
    ax.axhspan(lo, hi, color=MID_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)
    ax.axhspan(ymin, lo, color=LOW_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)

    tx = ax.get_yaxis_transform()
    ax.text(
        0.018,
        1.72,
        "High\nexposure",
        transform=tx,
        ha="left",
        va="center",
        color=HIGH_TEXT_COLOR,
        fontsize=9.2,
        fontweight="semibold",
    )
    ax.text(
        0.018,
        -2.15,
        "Low\nexposure",
        transform=tx,
        ha="left",
        va="center",
        color=LOW_TEXT_COLOR,
        fontsize=9.2,
        fontweight="semibold",
    )


def paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.5,
            "legend.fontsize": 9.1,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "axes.facecolor": AXES_FACE_COLOR,
            "figure.facecolor": FIGURE_FACE_COLOR,
            "savefig.facecolor": SAVEFIG_FACE_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_common_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def finish_axes(ax, legend_loc="lower left"):
    ax.set_xlim(1, 64)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    ax.set_xlabel("Expert rank within each layer (sorted)")
    style_common_axes(ax)
    legend = ax.legend(
        loc=legend_loc,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
    )
    for handle in legend.get_lines():
        handle.set_alpha(1.0)
        handle.set_linewidth(CURVE_LINEWIDTH + 0.35)


def plot_log_variant(df, out_dir, name, layers, title, ylim=(0.025, 5.4), legend_loc="lower left"):
    colors = LAYER_COLORS
    linestyles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))]

    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    add_log_bands(ax, ylim[0], ylim[1])
    for i, layer in enumerate(layers):
        y = sorted_norm_exposure(df, layer)
        x = np.arange(1, len(y) + 1)
        plot_exposure_curve(
            ax,
            x,
            y,
            line_color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"Layer {layer}",
        )
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks([0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0])
    ax.set_yticklabels(["0.03x", "0.1x", "0.3x", "0.5x", "1x", "1.5x", "3x"])
    ax.set_ylabel("Routing exposure / layer mean")
    ax.set_title(title, pad=8)
    finish_axes(ax, legend_loc)
    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_unsorted_log_variant(df, out_dir, name, layers):
    colors = LAYER_COLORS
    linestyles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))]
    ylim = (0.025, 5.4)

    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    add_log_bands(ax, ylim[0], ylim[1])
    for i, layer in enumerate(layers):
        x, y = expert_id_norm_exposure(df, layer)
        plot_exposure_curve(
            ax,
            x,
            y,
            line_color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"Layer {layer}",
        )
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 63)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63])
    ax.set_yticks([0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0])
    ax.set_yticklabels(["0.03x", "0.1x", "0.3x", "0.5x", "1x", "1.5x", "3x"])
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Routing exposure / layer mean")
    ax.set_title("", pad=8)
    style_common_axes(ax)
    legend = ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
    )
    for handle in legend.get_lines():
        handle.set_alpha(1.0)
        handle.set_linewidth(CURVE_LINEWIDTH + 0.35)
    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_log2_variant(df, out_dir, name, layers):
    colors = LAYER_COLORS[:3]
    linestyles = ["-", "--", "-."]
    ymin, ymax = -5.2, 2.2

    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    add_log2_bands(ax, ymin, ymax)
    hi = math.log2(HIGH_THRESHOLD)
    lo = math.log2(LOW_THRESHOLD)
    for i, layer in enumerate(layers):
        y = np.log2(sorted_norm_exposure(df, layer))
        x = np.arange(1, len(y) + 1)
        plot_exposure_curve(
            ax,
            x,
            y,
            line_color=colors[i],
            linestyle=linestyles[i],
            label=f"Layer {layer}",
            high=hi,
            low=lo,
            scale="linear",
        )
    ax.set_ylim(ymin, ymax)
    ticks = [-5, -4, -3, -2, -1, 0, 1, 2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{2 ** t:g}x" for t in ticks])
    ax.set_ylabel("Routing exposure / layer mean")
    ax.set_title("Sorted Routing Exposure Across Experts", pad=8)
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
