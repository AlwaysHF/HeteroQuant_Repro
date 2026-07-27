#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch, Rectangle


OLMOE_ACT_SCALES = (
    "/home/lwk/HeteroQuant_Repro/cache/"
    "smooth_stats_v2_act_scales_olmoe_compat_olmoe_wikitext2_128_4096_2_3210c09fb414c5ef.pt"
)
OLMOE_MOE_ACT_P99S = (
    "/home/lwk/HeteroQuant_Repro/cache/"
    "smooth_stats_v2_moe_act_p99s_olmoe_compat_olmoe_wikitext2_128_4096_2_3210c09fb414c5ef.pt"
)
QWEN_ACT_SCALES = (
    "/home/lwk/HeteroQuant_Repro/cache/"
    "smooth_stats_v2_act_scales_Qwen1.5-MoE-A2.7B_qwen2_moe_wikitext2_128_4096_2_3a8317c396ba7bd0.pt"
)
QWEN_MOE_ACT_P99S = (
    "/home/lwk/HeteroQuant_Repro/cache/"
    "smooth_stats_v2_moe_act_p99s_Qwen1.5-MoE-A2.7B_qwen2_moe_wikitext2_128_4096_2_3a8317c396ba7bd0.pt"
)


PALETTE = {
    # Main colors. Adjust these first when matching another figure style.
    "figure_face": "white",
    "axes_face": "#FDFBF7",
    "spine": "#B8B1A6",
    "legend_face": "white",
    "legend_edge": "#CFC6B8",
    "grid": "#D6D9DC",
    "mean": "#9AA0A6",
    "bar": "#DDD6CB",
    "scale": "#487DBA",
    "freq": "#D56858",
    "freq_edge": "#FEE9E8",
    "text": "#4B514F",
    "olmoe": "#D56858",
    "qwen": "#487DBA",
}

LAYOUT = {
    # Figure size and subplot spacing.
    "fig_width": 11.2,
    "fig_height": 7.2,
    "left_right_width_ratio": 2.35,
    "wspace": 0.11,
    "hspace": 0.43,
    "subplots_left": 0.065,
    "subplots_right": 0.985,
    "subplots_top": 0.965,
    "subplots_bottom": 0.120,
    # Caption distance from the bottom of each axes, in figure coordinates.
    "caption_gap": 0.039,
    # Curve axes can be scaled after GridSpec placement to create more room
    # for captions and projection lines.
    "curve_width_scale": 0.92,
    "curve_height_scale": 0.80,
    "curve_x_offset": 0.012,
    "curve_y_offset": 0.000,
}

PROJECTION = {
    # Two dashed lines connect the right edge of each left panel to the left
    # edge of its highlighted rectangle in the layer-wise curve.
    "left_top_y": 0.84,
    "left_bottom_y": 0.16,
    "line_width": 0.9,
    "alpha": 0.70,
    "dash": (0, (4, 3)),
}

HIGHLIGHT_BOX = {
    "height_frac": 0.075,
    "min_height": 2.0,
    "width_frac": 0.045,
    "min_width": 0.72,
    "line_width": 1.05,
}

MEAN_LINE = {
    "line_width": 3,
    "alpha": 0.75,
    "dash": "--",
}

FONT_SIZES = {
    # Global default text size.
    "base": 15,
    # Axis label text, e.g. Count, Layer, Overlap (%).
    "axis_label": 18,
    # Tick label numbers on x/y axes.
    "tick_label": 14,
    # Legend text in the first left panel.
    "legend": 15,
    # Mean-overlap annotation in the right curve panels.
    "mean_text": 17,
    # Subfigure captions under each panel: (a), (b), ...
    "panel_caption": 19,
}

LEFT_X_TICK_COUNT = 5
DEFAULT_OLMOE_LAYERS = "4,0"
DEFAULT_QWEN_LAYERS = "0,2"


def load_tensor_dict(path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in {path}, got {type(obj)}")
    return obj


def parse_layers(value):
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(layers) != 2:
        raise ValueError(f"expected exactly two layer ids, got {value!r}")
    return layers


def topk_indices(values, topk):
    values = torch.nan_to_num(values.detach().float().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    topk = min(int(topk), values.numel())
    return torch.topk(values, topk, largest=True, sorted=True).indices.cpu().numpy()


def shared_scale_for_layer(act_scales, moe_act_p99s, layer, source):
    key = f"model.layers.{layer}.mlp.gate"
    if source == "moe_p99_gate":
        if key not in moe_act_p99s:
            raise KeyError(f"missing {key} in moe_act_p99s")
        return moe_act_p99s[key]
    if source == "act_scale_gate":
        if key not in act_scales:
            raise KeyError(f"missing {key} in act_scales")
        return act_scales[key]
    raise ValueError(f"unknown shared_source: {source}")


def compute_result(act_scales, moe_act_p99s, layer, proj, topk, num_experts, shared_source):
    shared_scale = shared_scale_for_layer(act_scales, moe_act_p99s, layer, shared_source)
    scale_top = topk_indices(shared_scale, topk)
    counts = None
    for expert in range(num_experts):
        key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        if key not in act_scales:
            raise KeyError(f"missing expert activation scale: {key}")
        values = act_scales[key].detach().float().flatten()
        if counts is None:
            counts = torch.zeros(values.numel(), dtype=torch.int64)
        counts[torch.from_numpy(topk_indices(values, topk)).long()] += 1
    counts = counts.numpy()
    freq_top = np.argsort(-counts, kind="stable")[:topk]
    overlap = len(set(scale_top.tolist()) & set(freq_top.tolist())) / float(topk)
    return {
        "layer": int(layer),
        "proj": proj,
        "topk": int(topk),
        "num_experts": int(num_experts),
        "shared_source": shared_source,
        "counts": counts,
        "scale_top": scale_top,
        "freq_top": freq_top,
        "overlap": float(overlap),
    }


def compute_curve(act_scales, moe_act_p99s, num_layers, proj, topk, num_experts, shared_source):
    return [
        compute_result(act_scales, moe_act_p99s, layer, proj, topk, num_experts, shared_source)
        for layer in range(num_layers)
    ]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZES["base"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "axes.facecolor": PALETTE["axes_face"],
            "figure.facecolor": PALETTE["figure_face"],
            "savefig.facecolor": PALETTE["figure_face"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax):
    ax.set_facecolor(PALETTE["axes_face"])
    ax.grid(axis="y", linestyle="--", linewidth=0.38, color=PALETTE["grid"], alpha=0.52)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for key in ("left", "bottom"):
        ax.spines[key].set_color(PALETTE["spine"])
        ax.spines[key].set_linewidth(0.75)


def set_left_channel_ticks(ax, num_channels):
    ticks = np.linspace(0, num_channels, LEFT_X_TICK_COUNT, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])


def plot_frequency_panel(ax, result, model_label, show_legend=False):
    counts = result["counts"]
    channels = np.arange(counts.size)
    scale_top = result["scale_top"]
    freq_top = result["freq_top"]
    topk = result["topk"]

    ax.bar(
        channels,
        counts,
        width=1.0,
        color=PALETTE["bar"],
        edgecolor=PALETTE["bar"],
        linewidth=0.0,
        alpha=0.70,
        zorder=1,
    )
    scale_heights = np.maximum(counts[scale_top], max(2.0, float(counts.max()) * 0.08))
    ax.vlines(
        scale_top,
        0,
        scale_heights,
        color=PALETTE["scale"],
        linewidth=0.55,
        alpha=0.84,
        label=f"scale top-{topk}",
        zorder=3,
    )
    ax.scatter(
        freq_top,
        counts[freq_top],
        s=10,
        color=PALETTE["freq"],
        edgecolor=PALETTE["freq_edge"],
        linewidth=0.45,
        label=f"frequency top-{topk}",
        zorder=4,
    )

    y_max = max(int(counts.max()), topk)
    ax.set_xlim(-5, counts.size + 5)
    ax.set_ylim(0, y_max + max(3, y_max * 0.10))
    set_left_channel_ticks(ax, counts.size)
    style_axes(ax)
    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            facecolor=PALETTE["legend_face"],
            edgecolor=PALETTE["legend_edge"],
            borderpad=0.25,
            handlelength=1.6,
        )


def plot_curve_panel(ax, results, model_label, color, highlight_layers):
    layers = np.asarray([result["layer"] for result in results], dtype=int)
    overlaps = np.asarray([result["overlap"] * 100.0 for result in results], dtype=float)
    ax.plot(
        layers,
        overlaps,
        color=color,
        linewidth=1.75,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label=model_label,
        zorder=3,
    )
    ax.axhline(
        overlaps.mean(),
        color=PALETTE["mean"],
        linestyle=MEAN_LINE["dash"],
        linewidth=MEAN_LINE["line_width"],
        alpha=MEAN_LINE["alpha"],
        zorder=1,
    )
    y_low = max(0.0, overlaps.min() - 8.0)
    y_high = min(100.0, overlaps.max() + 6.0)
    highlight_layers = set(int(layer) for layer in highlight_layers)
    rect_height = max((y_high - y_low) * HIGHLIGHT_BOX["height_frac"], HIGHLIGHT_BOX["min_height"])
    base_x_span = (layers.max() - layers.min()) + 1.0
    rect_width = max(base_x_span * HIGHLIGHT_BOX["width_frac"], HIGHLIGHT_BOX["min_width"])
    x_low = layers.min() - max(0.5, rect_width * 0.65)
    x_high = layers.max() + max(0.5, rect_width * 0.65)
    ax.set_ylim(y_low, y_high)
    ax.set_xlim(x_low, x_high)

    highlight_geometry = {}
    for layer, overlap in zip(layers, overlaps):
        if int(layer) not in highlight_layers:
            continue
        left_x = layer - rect_width / 2.0
        top_y = overlap + rect_height / 2.0
        bottom_y = overlap - rect_height / 2.0
        highlight_geometry[int(layer)] = {
            "left_x": left_x,
            "center_y": overlap,
            "top_y": top_y,
            "bottom_y": bottom_y,
        }
        ax.add_patch(
            Rectangle(
                (left_x, bottom_y),
                rect_width,
                rect_height,
                fill=False,
                edgecolor=color,
                linewidth=HIGHLIGHT_BOX["line_width"],
                zorder=5,
            )
        )

    ax.text(
        0.965,
        0.075,
        f"mean {overlaps.mean():.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=PALETTE["text"],
        fontsize=FONT_SIZES["mean_text"],
        bbox=dict(facecolor=PALETTE["axes_face"], edgecolor="none", alpha=0.72, pad=0.7),
    )
    ax.set_ylabel("Overlap (%)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_label_coords(1.10, 0.5)
    ax.set_xlabel("Layer")
    style_axes(ax)
    ax.set_facecolor(PALETTE["figure_face"])
    return highlight_geometry


def add_projection_links(fig, left_ax, right_ax, highlight_geometry, layer, color):
    geom = highlight_geometry[int(layer)]
    pairs = [
        ((1.0, PROJECTION["left_top_y"]), (geom["left_x"], geom["top_y"])),
        ((1.0, PROJECTION["left_bottom_y"]), (geom["left_x"], geom["bottom_y"])),
    ]
    for xy_left, xy_right in pairs:
        con = ConnectionPatch(
            xyA=xy_left,
            coordsA=left_ax.transAxes,
            xyB=xy_right,
            coordsB=right_ax.transData,
            color=color,
            linewidth=PROJECTION["line_width"],
            linestyle=PROJECTION["dash"],
            alpha=PROJECTION["alpha"],
            zorder=2,
            clip_on=False,
        )
        fig.add_artist(con)


def add_panel_caption(fig, ax, caption, y=None):
    pos = ax.get_position()
    if y is None:
        y = pos.y0 - LAYOUT["caption_gap"]
    fig.text(
        0.5 * (pos.x0 + pos.x1),
        y,
        caption,
        ha="center",
        va="top",
        fontsize=FONT_SIZES["panel_caption"],
        color=PALETTE["text"],
    )


def shrink_curve_axis(ax):
    pos = ax.get_position()
    new_width = pos.width * LAYOUT["curve_width_scale"]
    new_height = pos.height * LAYOUT["curve_height_scale"]
    new_left = pos.x0 + LAYOUT["curve_x_offset"]
    new_bottom = pos.y0 + (pos.height - new_height) / 2.0 + LAYOUT["curve_y_offset"]
    ax.set_position([new_left, new_bottom, new_width, new_height])


def write_summary(path, model_name, results):
    rows = [
        {
            "model": model_name,
            "layer": result["layer"],
            "proj": result["proj"],
            "topk": result["topk"],
            "num_experts": result["num_experts"],
            "shared_source": result["shared_source"],
            "overlap": f"{result['overlap']:.6f}",
            "overlap_percent": f"{result['overlap'] * 100.0:.3f}",
            "max_count": int(result["counts"].max()),
            "nonzero_channels": int((result["counts"] > 0).sum()),
        }
        for result in results
    ]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Create a two-column composite figure for shared-scale top-K overlap."
    )
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/scale_topk_overlap_composite")
    parser.add_argument("--output_name", default="scale_topk_overlap_composite")
    parser.add_argument("--proj", default="up_proj")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--shared_source", default="act_scale_gate", choices=["act_scale_gate", "moe_p99_gate"])
    parser.add_argument("--olmoe_layers", default=DEFAULT_OLMOE_LAYERS)
    parser.add_argument("--qwen_layers", default=DEFAULT_QWEN_LAYERS)
    parser.add_argument("--fig_width", type=float, default=LAYOUT["fig_width"])
    parser.add_argument("--fig_height", type=float, default=LAYOUT["fig_height"])
    args = parser.parse_args()

    setup_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    olmoe_act = load_tensor_dict(OLMOE_ACT_SCALES)
    olmoe_p99 = load_tensor_dict(OLMOE_MOE_ACT_P99S)
    qwen_act = load_tensor_dict(QWEN_ACT_SCALES)
    qwen_p99 = load_tensor_dict(QWEN_MOE_ACT_P99S)

    olmoe_curve = compute_curve(olmoe_act, olmoe_p99, 16, args.proj, args.topk, 64, args.shared_source)
    qwen_curve = compute_curve(qwen_act, qwen_p99, 24, args.proj, args.topk, 60, args.shared_source)

    olmoe_layers = parse_layers(args.olmoe_layers)
    qwen_layers = parse_layers(args.qwen_layers)
    left_results = [
        ("OLMoE", olmoe_curve[olmoe_layers[0]]),
        ("OLMoE", olmoe_curve[olmoe_layers[1]]),
        ("Qwen", qwen_curve[qwen_layers[0]]),
        ("Qwen", qwen_curve[qwen_layers[1]]),
    ]

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[LAYOUT["left_right_width_ratio"], 1.0],
        height_ratios=[1, 1, 1, 1],
        wspace=LAYOUT["wspace"],
        hspace=LAYOUT["hspace"],
    )
    left_axes = [fig.add_subplot(gs[row, 0]) for row in range(4)]
    right_top = fig.add_subplot(gs[0:2, 1])
    right_bottom = fig.add_subplot(gs[2:4, 1])

    for idx, (ax, (model_label, result)) in enumerate(zip(left_axes, left_results)):
        plot_frequency_panel(ax, result, model_label, show_legend=(idx == 0))
        ax.set_ylabel("Count")

    olmoe_highlights = plot_curve_panel(right_top, olmoe_curve, "OLMoE", PALETTE["olmoe"], olmoe_layers)
    qwen_highlights = plot_curve_panel(right_bottom, qwen_curve, "Qwen", PALETTE["qwen"], qwen_layers)

    fig.subplots_adjust(
        left=LAYOUT["subplots_left"],
        right=LAYOUT["subplots_right"],
        top=LAYOUT["subplots_top"],
        bottom=LAYOUT["subplots_bottom"],
    )
    shrink_curve_axis(right_top)
    shrink_curve_axis(right_bottom)
    fig.canvas.draw()

    add_projection_links(fig, left_axes[0], right_top, olmoe_highlights, olmoe_layers[0], PALETTE["olmoe"])
    add_projection_links(fig, left_axes[1], right_top, olmoe_highlights, olmoe_layers[1], PALETTE["olmoe"])
    add_projection_links(fig, left_axes[2], right_bottom, qwen_highlights, qwen_layers[0], PALETTE["qwen"])
    add_projection_links(fig, left_axes[3], right_bottom, qwen_highlights, qwen_layers[1], PALETTE["qwen"])

    left_captions = [
        f"(a) OLMoE L{olmoe_layers[0]} overlap {olmoe_curve[olmoe_layers[0]]['overlap'] * 100.0:.1f}%",
        f"(b) OLMoE L{olmoe_layers[1]} overlap {olmoe_curve[olmoe_layers[1]]['overlap'] * 100.0:.1f}%",
        f"(c) Qwen L{qwen_layers[0]} overlap {qwen_curve[qwen_layers[0]]['overlap'] * 100.0:.1f}%",
        f"(d) Qwen L{qwen_layers[1]} overlap {qwen_curve[qwen_layers[1]]['overlap'] * 100.0:.1f}%",
    ]
    left_caption_y = []
    for idx, (ax, caption) in enumerate(zip(left_axes, left_captions)):
        y = ax.get_position().y0 - LAYOUT["caption_gap"]
        left_caption_y.append(y)
        add_panel_caption(fig, ax, caption, y=y)
    add_panel_caption(fig, right_top, "(e) OLMoE layer-wise overlap", y=left_caption_y[1])
    add_panel_caption(fig, right_bottom, "(f) Qwen layer-wise overlap", y=left_caption_y[3])

    output_prefix = output_dir / args.output_name
    fig.savefig(str(output_prefix) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_prefix) + ".pdf", bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / (args.output_name + "_summary.csv")
    if summary_path.exists():
        summary_path.unlink()
    write_summary(summary_path, "OLMoE", olmoe_curve)
    write_summary(summary_path, "Qwen", qwen_curve)

    print(f"saved_png={output_prefix}.png")
    print(f"saved_pdf={output_prefix}.pdf")
    print(f"saved_summary={summary_path}")
    print(f"olmoe_mean_overlap={np.mean([r['overlap'] for r in olmoe_curve]) * 100.0:.3f}")
    print(f"qwen_mean_overlap={np.mean([r['overlap'] for r in qwen_curve]) * 100.0:.3f}")


if __name__ == "__main__":
    main()
