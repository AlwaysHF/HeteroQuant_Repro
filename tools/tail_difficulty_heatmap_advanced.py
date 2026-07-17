#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCORES = (
    "/home/lwk/HeteroQuant_Repro/experiments/"
    "metric_ablation_olmoe_ppl_group2048_20260716_101427/"
    "routed_tail/plans/expert_distribution_scores.csv"
)


# Match the visual language in routing_exposure_revised_v7.py.
HIGH_THRESHOLD = 1.12
LOW_THRESHOLD = 0.88
CURVE_LINEWIDTH = 1.05
MID_LIGHTEN_AMOUNT = 0.68
EXTREME_MARKER_SIZE = 25.0
EXTREME_MARKER_HALO_SIZE = 42.0
EXTREME_MARKER_WIDTH = 0.9

HIGH_BAND_COLOR = "#FFEAEE"
MID_BAND_COLOR = "#FFFEFC"
LOW_BAND_COLOR = "#EFF4FF"
HIGH_TEXT_COLOR = "#A5483A"
LOW_TEXT_COLOR = "#36597C"
AXES_FACE_COLOR = "#FDFBF7"
SPINE_COLOR = "#B8B1A6"
LEGEND_EDGE_COLOR = "#CFC6B8"
FIGURE_FACE_COLOR = "white"
SAVEFIG_FACE_COLOR = "white"

LAYER_COLORS = [
    "#70757A",
    "#9AA0A6",
    "#C1C5CA",
    "#3C8D7B",
    "#7A5AA6",
]

HIGH_MARKER_FACE_COLOR = "#FA8775"
HIGH_MARKER_EDGE_COLOR = "#FEE9E8"
LOW_MARKER_FACE_COLOR = "#487DBA"
LOW_MARKER_EDGE_COLOR = "#5BA6FC"
MARKER_HALO_COLOR = "white"

# =====================================================
# Advanced heatmap controls
# =====================================================
HEATMAP_LOW_COLOR = "#3F6FA6"
HEATMAP_LOW_MID_COLOR = "#B9CCE1"
HEATMAP_CENTER_COLOR = "#FAF8F3"
HEATMAP_HIGH_MID_COLOR = "#F2BDB4"
HEATMAP_HIGH_COLOR = "#D56858"
HEATMAP_MISSING_COLOR = "#D8DADD"
HEATMAP_CLIP_PERCENTILE = 98.0
HEATMAP_CELL_ASPECT = 1.4
HEATMAP_SHOW_LAYER_SUMMARY = True
HEATMAP_SUMMARY_COLOR = "#6F767D"
HEATMAP_SUMMARY_GUIDE_COLOR = "#D6D9DC"
HEATMAP_GROUP_SIZE = 4
HEATMAP_GROUP_SEPARATOR_COLOR = "#FFFFFF"
HEATMAP_GROUP_SEPARATOR_WIDTH = 0.70
HEATMAP_RASTERIZED = True
HEATMAP_TITLE = None  # Prefer the paper caption; set a string to show a title.


def lighten_color(color, amount=MID_LIGHTEN_AMOUNT):
    rgb = np.asarray(mcolors.to_rgb(color))
    white = np.ones(3)
    return tuple(np.clip((1.0 - amount) * rgb + amount * white, 0.0, 1.0))


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


def add_tail_bands(ax, ymin, ymax):
    ax.axhspan(HIGH_THRESHOLD, ymax, color=HIGH_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)
    ax.axhspan(LOW_THRESHOLD, HIGH_THRESHOLD, color=MID_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)
    ax.axhspan(ymin, LOW_THRESHOLD, color=LOW_BAND_COLOR, alpha=0.96, linewidth=0, zorder=-10)
    ax.axhline(1.0, color="#C8BFB2", linewidth=0.85, linestyle="--", zorder=-1)

    tx = ax.get_yaxis_transform()
    ax.text(
        0.50,
        min(ymax - 0.045, HIGH_THRESHOLD + 0.22),
        "High\ntail",
        transform=tx,
        ha="center",
        va="center",
        color=HIGH_TEXT_COLOR,
        fontsize=9.2,
        fontweight="semibold",
    )
    ax.text(
        0.018,
        max(ymin + 0.045, LOW_THRESHOLD - 0.13),
        "Low\ntail",
        transform=tx,
        ha="left",
        va="center",
        color=LOW_TEXT_COLOR,
        fontsize=9.2,
        fontweight="semibold",
    )


def split_curve_by_regions(x, y, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pieces = []
    for i in range(len(x) - 1):
        x0, x1 = float(x[i]), float(x[i + 1])
        y0, y1 = float(y[i]), float(y[i + 1])
        cuts = [0.0, 1.0]
        if y1 != y0:
            for threshold in (low, high):
                t = (threshold - y0) / (y1 - y0)
                if 0.0 < t < 1.0:
                    cuts.append(float(t))
        cuts = sorted(set(cuts))
        for ta, tb in zip(cuts[:-1], cuts[1:]):
            xa = x0 + ta * (x1 - x0)
            xb = x0 + tb * (x1 - x0)
            ya = y0 + ta * (y1 - y0)
            yb = y0 + tb * (y1 - y0)
            ym = 0.5 * (ya + yb)
            if ym > high:
                region = "high"
            elif ym < low:
                region = "low"
            else:
                region = "middle"
            pieces.append((region, [xa, xb], [ya, yb]))

    merged = []
    for region, xs, ys in pieces:
        if merged and merged[-1][0] == region:
            merged[-1][1].append(xs[-1])
            merged[-1][2].append(ys[-1])
        else:
            merged.append([region, list(xs), list(ys)])
    return [(r, np.asarray(xs), np.asarray(ys)) for r, xs, ys in merged]


def add_extreme_markers(ax, x, y):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x)
    for mask, face_color, edge_color in [
        (y > HIGH_THRESHOLD, HIGH_MARKER_FACE_COLOR, HIGH_MARKER_EDGE_COLOR),
        (y < LOW_THRESHOLD, LOW_MARKER_FACE_COLOR, LOW_MARKER_EDGE_COLOR),
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


def tail_norm_by_rank(df, layer):
    group = df[df["layer_index"] == layer]
    if group.empty:
        raise ValueError(f"layer {layer} not found")
    values = group["tail_score"].to_numpy(dtype=float)
    values = values / max(float(values.mean()), 1e-12)
    return np.sort(values)[::-1]


def tail_norm_matrix(df):
    required = {"layer_index", "expert_index", "tail_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicate_mask = df.duplicated(["layer_index", "expert_index"], keep=False)
    if duplicate_mask.any():
        sample = df.loc[duplicate_mask, ["layer_index", "expert_index"]].head(10)
        raise ValueError(
            "Duplicate layer-expert pairs were found. Sample:\n"
            f"{sample.to_string(index=False)}"
        )

    pivot = (
        df.pivot(index="layer_index", columns="expert_index", values="tail_score")
        .sort_index()
        .sort_index(axis=1)
    )
    layer_mean = pivot.mean(axis=1, skipna=True).replace(0.0, np.nan)
    normalized = pivot.div(layer_mean, axis=0)

    return (
        normalized.index.to_numpy(dtype=int),
        normalized.columns.to_numpy(dtype=int),
        normalized.to_numpy(dtype=float),
    )


def plot_tail_curve(ax, x, y, line_color, linestyle, label):
    middle_color = lighten_color(line_color)
    for region, xs, ys in split_curve_by_regions(x, y):
        if region == "high":
            color = HIGH_MARKER_FACE_COLOR
            width = CURVE_LINEWIDTH + 0.16
            zorder = 3
        elif region == "low":
            color = LOW_MARKER_FACE_COLOR
            width = CURVE_LINEWIDTH + 0.16
            zorder = 3
        else:
            color = middle_color
            width = CURVE_LINEWIDTH
            zorder = 2
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
    ax.plot(
        [],
        [],
        color=line_color,
        linewidth=CURVE_LINEWIDTH + 0.25,
        linestyle=linestyle,
        alpha=1.0,
        label=label,
    )
    add_extreme_markers(ax, x, y)


def plot_sorted_tail(df, out_dir, name, layers, title, legend_loc="upper right"):
    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    ymin, ymax = 0.60, 1.46
    add_tail_bands(ax, ymin, ymax)
    linestyles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))]

    for i, layer in enumerate(layers):
        y = tail_norm_by_rank(df, layer)
        x = np.arange(1, len(y) + 1)
        plot_tail_curve(
            ax,
            x,
            y,
            line_color=LAYER_COLORS[i % len(LAYER_COLORS)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"Layer {layer}",
        )

    ax.set_xlim(1, 64)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    ax.set_yticks([0.65, 0.8, 0.88, 1.0, 1.12, 1.25, 1.4])
    ax.set_yticklabels(["0.65x", "0.8x", "0.88x", "1x", "1.12x", "1.25x", "1.4x"])
    ax.set_xlabel("Expert rank within each layer (sorted)")
    ax.set_ylabel("Tail difficulty / layer mean")
    ax.set_title(title, pad=8)
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
    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_tail_heatmap(df, out_dir, name):
    layers, experts, ratio = tail_norm_matrix(df)
    ratio = np.asarray(ratio, dtype=float)
    ratio[ratio <= 0] = np.nan

    # Multiplicative deviations become symmetric around 1x in log2 space.
    heat = np.log2(ratio)
    valid_abs = np.abs(heat[np.isfinite(heat)])
    threshold_extent = max(
        abs(np.log2(LOW_THRESHOLD)),
        abs(np.log2(HIGH_THRESHOLD)),
    )
    if valid_abs.size:
        vmax = float(np.percentile(valid_abs, HEATMAP_CLIP_PERCENTILE))
    else:
        vmax = threshold_extent
    vmax = max(vmax, threshold_extent, 0.04)

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tail_difficulty_advanced",
        [
            HEATMAP_LOW_COLOR,
            HEATMAP_LOW_MID_COLOR,
            HEATMAP_CENTER_COLOR,
            HEATMAP_HIGH_MID_COLOR,
            HEATMAP_HIGH_COLOR,
        ],
        N=256,
    )
    cmap.set_bad(HEATMAP_MISSING_COLOR)

    n_layers, n_experts = heat.shape
    show_summary = HEATMAP_SHOW_LAYER_SUMMARY and n_layers > 1
    figure_width = 7.35
    figure_height = max(2.50, 1.20 + 0.090 * n_layers)

    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=False)
    width_ratios = [1.0, 0.115, 0.030] if show_summary else [1.0, 0.030]
    gs = fig.add_gridspec(
        nrows=1,
        ncols=len(width_ratios),
        width_ratios=width_ratios,
        left=0.085,
        right=0.965,
        bottom=0.18,
        top=0.94,
        wspace=0.12,
    )

    ax = fig.add_subplot(gs[0, 0])
    if show_summary:
        summary_ax = fig.add_subplot(gs[0, 1], sharey=ax)
        cax = fig.add_subplot(gs[0, 2])
    else:
        summary_ax = None
        cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(
        heat,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
        rasterized=HEATMAP_RASTERIZED,
    )
    ax.set_box_aspect((n_layers / max(n_experts, 1)) * HEATMAP_CELL_ASPECT)

    if HEATMAP_TITLE:
        ax.set_title(HEATMAP_TITLE, loc="left", pad=7, fontweight="semibold")

    ax.set_xlabel("Expert index", labelpad=5)
    ax.set_ylabel("Layer index", labelpad=7)

    num_xticks = min(9, n_experts)
    x_positions = np.unique(np.linspace(0, n_experts - 1, num_xticks, dtype=int))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(experts[i]) for i in x_positions])

    if n_layers <= 20:
        y_positions = np.arange(n_layers)
    else:
        y_positions = np.unique(np.linspace(0, n_layers - 1, 12, dtype=int))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(layers[i]) for i in y_positions])
    ax.tick_params(axis="both", which="both", length=0, pad=4)

    # Thin separators only between layer groups, not between every cell.
    if HEATMAP_GROUP_SIZE and HEATMAP_GROUP_SIZE > 0:
        for boundary in range(HEATMAP_GROUP_SIZE, n_layers, HEATMAP_GROUP_SIZE):
            ax.axhline(
                boundary - 0.5,
                color=HEATMAP_GROUP_SEPARATOR_COLOR,
                linewidth=HEATMAP_GROUP_SEPARATOR_WIDTH,
                alpha=0.92,
                zorder=3,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compact colorbar: short title instead of a long rotated label.
    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="vertical",
        extend="both",
        extendfrac=0.025,
    )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8.2, pad=3)
    cbar.ax.set_title("Tail ratio", fontsize=8.3, pad=5, fontweight="semibold")

    candidate_ratios = np.asarray(
        [0.80, LOW_THRESHOLD, 1.00, HIGH_THRESHOLD, 1.25],
        dtype=float,
    )
    candidate_positions = np.log2(candidate_ratios)
    keep = (candidate_positions >= -vmax) & (candidate_positions <= vmax)
    cbar.set_ticks(candidate_positions[keep])
    cbar.set_ticklabels([f"{v:g}x" for v in candidate_ratios[keep]])

    if summary_ax is not None:
        # Per-layer heterogeneity: standard deviation in log2 space.
        layer_dispersion = np.nanstd(heat, axis=1)
        y = np.arange(n_layers)
        summary_ax.hlines(
            y,
            0.0,
            layer_dispersion,
            color=HEATMAP_SUMMARY_GUIDE_COLOR,
            linewidth=1.0,
            zorder=1,
        )
        summary_ax.scatter(
            layer_dispersion,
            y,
            s=13,
            color=HEATMAP_SUMMARY_COLOR,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        summary_ax.set_ylim(ax.get_ylim())
        summary_ax.tick_params(axis="y", left=False, labelleft=False)
        summary_ax.set_xticks([])
        summary_ax.set_title("Layer σ", fontsize=8.0, pad=5)
        summary_ax.tick_params(length=0)
        summary_ax.set_xlim(left=0.0)
        for spine in summary_ax.spines.values():
            spine.set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"{name}.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.025,
        )
    plt.close(fig)


def write_summary(df, out_dir):
    rows = []
    for layer in sorted(int(x) for x in df["layer_index"].unique()):
        values = df[df["layer_index"] == layer]["tail_score"].to_numpy(dtype=float)
        norm = values / max(float(values.mean()), 1e-12)
        rows.append(
            {
                "layer": layer,
                "min_norm": float(norm.min()),
                "max_norm": float(norm.max()),
                "std_norm": float(norm.std(ddof=0)),
                "high_count": int((norm > HIGH_THRESHOLD).sum()),
                "low_count": int((norm < LOW_THRESHOLD).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "tail_difficulty_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", default=DEFAULT_SCORES)
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/tail_difficulty_variants")
    args = parser.parse_args()

    paper_style()
    df = pd.read_csv(args.scores_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sorted_tail(
        df,
        out_dir,
        "tail_difficulty_sorted_3layers_matched",
        [0, 6, 15],
        "Sorted Expert Tail Difficulty",
    )
    print("saved tail_difficulty_sorted_3layers_matched: layers=[0, 6, 15]")

    plot_sorted_tail(
        df,
        out_dir,
        "tail_difficulty_sorted_3layers_extreme",
        [0, 3, 15],
        "Sorted Expert Tail Difficulty",
    )
    print("saved tail_difficulty_sorted_3layers_extreme: layers=[0, 3, 15]")

    plot_sorted_tail(
        df,
        out_dir,
        "tail_difficulty_sorted_5layers",
        [0, 3, 6, 12, 15],
        "Sorted Expert Tail Difficulty",
    )
    print("saved tail_difficulty_sorted_5layers: layers=[0, 3, 6, 12, 15]")

    plot_tail_heatmap(df, out_dir, "tail_difficulty_heatmap")
    print("saved tail_difficulty_heatmap")

    write_summary(df, out_dir)
    print(f"saved summary: {out_dir / 'tail_difficulty_summary.csv'}")


if __name__ == "__main__":
    main()
