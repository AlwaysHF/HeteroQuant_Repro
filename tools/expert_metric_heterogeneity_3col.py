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
            "font.size": 8.8,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.2,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
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


def sorted_model_ratios(df, score_col):
    ratios = []
    for _, group in df.groupby("layer_index"):
        mean = max(float(group[score_col].mean()), 1e-12)
        ratios.extend((group[score_col].astype(float) / mean).tolist())
    return np.sort(np.asarray(ratios, dtype=float))[::-1]


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


def add_heatmap_case(
    fig,
    ax_full,
    ax_zoom,
    df,
    model_label,
    metric_label,
    score_col,
    cmap,
    norm,
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
    if model_label == "OLMoE":
        ytick_idx = np.asarray([layers.index(v) for v in (0, 5, 10, 15) if v in layers], dtype=int)
    else:
        ytick_idx = np.unique(np.linspace(0, len(layers) - 1, min(7, len(layers)), dtype=int))
    ax_full.set_yticks(ytick_idx)
    ax_full.set_yticklabels([str(layers[i]) for i in ytick_idx])
    xticks = np.unique(np.linspace(0, len(experts) - 1, min(5, len(experts)), dtype=int))
    ax_full.set_xticks(xticks)
    ax_full.set_xticklabels([str(experts[i]) for i in xticks])
    ax_full.set_ylabel("Layer ID", labelpad=1, fontsize=8.2)
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
    format_zoom_ticks(ax_zoom, experts)
    style_heatmap_axis(ax_zoom)
    add_row_zoom_box(fig, ax_full, ax_zoom, zoom_idx, len(experts))
    return im


def style_curve_axis(ax, model_label, show_xlabel=False):
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.42, 8.6)
    ax.set_xlim(0, 100)
    ax.axhline(1.0, color="#BEB6AA", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks([0.5, 1.0, 2.0, 4.0, 8.0])
    ax.set_yticklabels(["0.5x", "1x", "2x", "4x", "8x"])
    ax.set_xticks([0, 25, 50, 75, 100])
    if not show_xlabel:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Expert rank (%)", labelpad=1, fontsize=8.0)
    ax.set_ylabel("Score ratio", labelpad=2, fontsize=8.0)
    ax.yaxis.set_label_position("right")
    ax.text(
        0.035,
        0.93,
        model_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="semibold",
        color="#303030",
    )
    ax.grid(axis="y", color="#E5DDD1", linewidth=0.55, alpha=0.85)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8B1A6")
        ax.spines[side].set_linewidth(0.8)


def add_sorted_curves(ax, df, model_label, show_xlabel=False):
    tail = sorted_model_ratios(df, METRICS["tail"])
    range_score = sorted_model_ratios(df, METRICS["range"])
    x_tail = np.linspace(0.0, 100.0, len(tail))
    x_range = np.linspace(0.0, 100.0, len(range_score))

    ax.plot(
        x_tail,
        tail,
        color="#C84B45",
        linewidth=1.55,
        label="Tail",
        solid_capstyle="round",
    )
    ax.plot(
        x_range,
        range_score,
        color="#2F6FB0",
        linewidth=1.55,
        label="Range",
        solid_capstyle="round",
    )
    style_curve_axis(ax, model_label, show_xlabel=show_xlabel)
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#CFC6B8",
        handlelength=1.8,
    )
    for line in legend.get_lines():
        line.set_linewidth(1.7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--olmoe_scores", default=DEFAULT_OLMOE_SCORES)
    parser.add_argument("--qwen_scores", default=DEFAULT_QWEN_SCORES)
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/expert_metric_heterogeneity_3col")
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

    fig = plt.figure(figsize=(10.35, 6.05))
    gs = fig.add_gridspec(
        4,
        4,
        width_ratios=[3.22, 1.30, 0.08, 1.56],
        height_ratios=[1, 1, 1, 1],
        left=0.055,
        right=0.955,
        bottom=0.13,
        top=0.945,
        wspace=0.14,
        hspace=0.32,
    )

    last_im = None
    heatmap_axes = []
    zoom_axes = []
    for idx, (df, model_label, metric_label, score_col) in enumerate(cases):
        ax_full = fig.add_subplot(gs[idx, 0])
        ax_zoom = fig.add_subplot(gs[idx, 1])
        heatmap_axes.append(ax_full)
        zoom_axes.append(ax_zoom)
        last_im = add_heatmap_case(
            fig,
            ax_full,
            ax_zoom,
            df,
            model_label,
            metric_label,
            score_col,
            cmap,
            norm,
        )

    ax_curve_olmoe = fig.add_subplot(gs[0:2, 3])
    ax_curve_qwen = fig.add_subplot(gs[2:4, 3])
    add_sorted_curves(ax_curve_olmoe, olmoe, "OLMoE", show_xlabel=True)
    add_sorted_curves(ax_curve_qwen, qwen, "Qwen", show_xlabel=True)

    # =====================================================
    # 调整折线图高度
    # 1.0：原始高度
    # 小于 1.0：变矮
    # 大于 1.0：变高
    # =====================================================

    OLMOE_CURVE_HEIGHT_SCALE = 0.80
    QWEN_CURVE_HEIGHT_SCALE = 0.80

    OLMOE_CURVE_Y_OFFSET = -0.015
    QWEN_CURVE_Y_OFFSET = -0.015


    def resize_curve_height(ax, height_scale, y_offset=0.0):
        p = ax.get_position()

        new_height = p.height * height_scale

        # 默认保持顶部不变
        new_bottom = p.y1 - new_height

        # y_offset < 0：向下移动
        # y_offset > 0：向上移动
        new_bottom += y_offset

        ax.set_position(
            [
                p.x0,
                new_bottom,
                p.width,
                new_height,
            ]
        )


    resize_curve_height(
        ax_curve_olmoe,
        OLMOE_CURVE_HEIGHT_SCALE,
        OLMOE_CURVE_Y_OFFSET,
    )

    resize_curve_height(
        ax_curve_qwen,
        QWEN_CURVE_HEIGHT_SCALE,
        QWEN_CURVE_Y_OFFSET,
    )

    fig.canvas.draw()

    fig.canvas.draw()

    panel_captions = [
        "(a) OLMoE tail",
        "(b) OLMoE range",
        "(c) Qwen tail",
        "(d) Qwen range",
    ]
    for caption, ax_full, ax_zoom in zip(panel_captions, heatmap_axes, zoom_axes):
        p0 = ax_full.get_position()
        p1 = ax_zoom.get_position()
        x = 0.5 * (p0.x0 + p1.x1)
        y = min(p0.y0, p1.y0) - 0.023
        fig.text(x, y, caption, ha="center", va="top", fontsize=9.2)

    # for caption, ax_curve in [
    #     ("(e) OLMoE sorted score distribution", ax_curve_olmoe),
    #     ("(f) Qwen sorted score distribution", ax_curve_qwen),
    # ]:
    #     p = ax_curve.get_position()
    #     fig.text(0.5 * (p.x0 + p.x1), p.y0 - 0.040, caption, ha="center", va="top", fontsize=9.2)
    panel_captions = [
        "(a) OLMoE tail",
        "(b) OLMoE range",
        "(c) Qwen tail",
        "(d) Qwen range",
    ]

    # 保存 a、b、c、d 四个标题的纵坐标，
    # 后面让 e 与 b 对齐，f 与 d 对齐
    heatmap_caption_y = []

    for caption, ax_full, ax_zoom in zip(
        panel_captions,
        heatmap_axes,
        zoom_axes,
    ):
        p0 = ax_full.get_position()
        p1 = ax_zoom.get_position()

        x = 0.5 * (p0.x0 + p1.x1)
        y = min(p0.y0, p1.y0) - 0.023

        heatmap_caption_y.append(y)

        fig.text(
            x,
            y,
            caption,
            ha="center",
            va="top",
            fontsize=9.2,
        )

    # 折线图标题：
    # e 与 b 使用完全相同的纵坐标
    # f 与 d 使用完全相同的纵坐标
    curve_captions = [
        (
            "(e) OLMoE sorted score distribution",
            ax_curve_olmoe,
            heatmap_caption_y[1],  # 与 (b) 对齐
        ),
        (
            "(f) Qwen sorted score distribution",
            ax_curve_qwen,
            heatmap_caption_y[3],  # 与 (d) 对齐
        ),
    ]

    for caption, ax_curve, caption_y in curve_captions:
        p = ax_curve.get_position()
        caption_x = 0.5 * (p.x0 + p.x1)

        fig.text(
            caption_x,
            caption_y,
            caption,
            ha="center",
            va="top",
            fontsize=9.2,
        )



    zoom_right = max(ax.get_position().x1 for ax in zoom_axes)
    curve_left = min(
        ax_curve_olmoe.get_position().x0,
        ax_curve_qwen.get_position().x0,
    )

    all_bottom = min(ax.get_position().y0 for ax in heatmap_axes)
    all_top = max(ax.get_position().y1 for ax in heatmap_axes)
    full_h = all_top - all_bottom

    cbar_h = full_h * 0.70
    cbar_w = 0.017

    # 比例尺在第二列和折线图坐标轴之间的几何中心
    cbar_center_x = 0.5 * (zoom_right + curve_left)

    # 视觉修正：
    # 负数向左，正数向右
    CBAR_X_OFFSET = -0.011

    cbar_x = cbar_center_x - 0.5 * cbar_w + CBAR_X_OFFSET
    cbar_y = all_bottom + 0.5 * (full_h - cbar_h)

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])


    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.015, "2x", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=8.0)
    cbar.ax.text(0.5, -0.020, "0.5x", transform=cbar.ax.transAxes, ha="center", va="top", fontsize=8.0)
    cbar.outline.set_linewidth(0.65)

    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"expert_metric_heterogeneity_3col.{ext}",
            dpi=520 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)

    print(f"saved {out_dir / 'expert_metric_heterogeneity_3col.png'}")
    print(f"saved {out_dir / 'expert_metric_heterogeneity_3col.pdf'}")


if __name__ == "__main__":
    main()
