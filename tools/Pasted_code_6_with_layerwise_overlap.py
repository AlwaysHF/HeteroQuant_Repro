#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_ACT_SCALES = (
    "/home/lwk/HeteroQuant_Repro/cache/"
    "smooth_stats_v2_act_scales_olmoe_compat_olmoe_wikitext2_128_4096_2_3210c09fb414c5ef.pt"
)
DEFAULT_MOE_ACT_P99S = (
    "/home/lwk/HeteroQuant_Repro/cache/"
    "smooth_stats_v2_moe_act_p99s_olmoe_compat_olmoe_wikitext2_128_4096_2_3210c09fb414c5ef.pt"
)

# =====================================================
# Figure colors: modify these values to tune the style.
# =====================================================
COLOR_BG = "#FDFBF7"
COLOR_BAR = "#DDD6CB"
COLOR_SCALE = "#487DBA"
COLOR_FREQ = "#D56858"
COLOR_UP = "#487DBA"
COLOR_GATE = "#D56858"
COLOR_GRID = "#D6D9DC"
COLOR_SPINE = "#B8B1A6"
COLOR_TEXT = "#4B514F"
COLOR_MUTED = "#6F767D"


def load_tensor_dict(path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in {path}, got {type(obj)}")
    return obj


def topk_indices(values, topk):
    values = torch.nan_to_num(
        values.detach().float().flatten(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
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


def expert_frequency(act_scales, layer, proj, topk, num_experts):
    counts = None
    expert_topk = {}
    for expert in range(num_experts):
        key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        if key not in act_scales:
            raise KeyError(f"missing expert activation scale: {key}")
        values = act_scales[key].detach().float().flatten()
        if counts is None:
            counts = torch.zeros(values.numel(), dtype=torch.int64)
        idx = topk_indices(values, topk)
        counts[torch.from_numpy(idx).long()] += 1
        expert_topk[expert] = idx
    return counts.numpy(), expert_topk


def compute_overlap(
    act_scales,
    moe_act_p99s,
    layer,
    proj,
    topk,
    num_experts,
    shared_source,
):
    shared_scale = shared_scale_for_layer(
        act_scales,
        moe_act_p99s,
        layer,
        shared_source,
    )
    scale_top = topk_indices(shared_scale, topk)
    counts, expert_topk = expert_frequency(
        act_scales,
        layer,
        proj,
        topk,
        num_experts,
    )
    freq_top = np.argsort(-counts, kind="stable")[:topk]
    overlap = len(set(scale_top.tolist()) & set(freq_top.tolist())) / float(topk)
    return {
        "layer": layer,
        "proj": proj,
        "topk": topk,
        "num_experts": num_experts,
        "shared_source": shared_source,
        "shared_scale": shared_scale.detach().float().cpu().numpy().flatten(),
        "scale_top": scale_top,
        "counts": counts,
        "freq_top": freq_top,
        "overlap": overlap,
        "expert_topk": expert_topk,
    }


def compute_layerwise_overlaps(
    act_scales,
    moe_act_p99s,
    layers,
    projs,
    topk,
    num_experts,
    shared_source,
):
    """Compute overlap (%) for every requested projection at every layer."""
    layers = [int(layer) for layer in layers]
    curves = {proj: [] for proj in projs}

    for layer in layers:
        for proj in projs:
            result = compute_overlap(
                act_scales,
                moe_act_p99s,
                layer,
                proj,
                topk,
                num_experts,
                shared_source,
            )
            curves[proj].append(result["overlap"] * 100.0)

    return {
        "layers": np.asarray(layers, dtype=int),
        "curves": {
            proj: np.asarray(values, dtype=float)
            for proj, values in curves.items()
        },
    }


def find_best_layer(
    act_scales,
    moe_act_p99s,
    layers,
    proj,
    topk,
    num_experts,
    shared_source,
):
    best = None
    for layer in layers:
        result = compute_overlap(
            act_scales,
            moe_act_p99s,
            layer,
            proj,
            topk,
            num_experts,
            shared_source,
        )
        if best is None or result["overlap"] > best["overlap"]:
            best = result
    return best


def write_frequency_csv(path, result):
    rows = []
    scale_set = set(result["scale_top"].tolist())
    freq_set = set(result["freq_top"].tolist())
    for channel, count in enumerate(result["counts"].tolist()):
        rows.append(
            {
                "layer": result["layer"],
                "proj": result["proj"],
                "channel": channel,
                "selected_count": count,
                "is_scale_topk": int(channel in scale_set),
                "is_frequency_topk": int(channel in freq_set),
            }
        )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "proj",
                "channel",
                "selected_count",
                "is_scale_topk",
                "is_frequency_topk",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_layerwise_overlap_csv(path, trend_result):
    projs = list(trend_result["curves"].keys())
    fieldnames = ["layer"] + [f"{proj}_overlap_pct" for proj in projs]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, layer in enumerate(trend_result["layers"].tolist()):
            row = {"layer": layer}
            for proj in projs:
                row[f"{proj}_overlap_pct"] = float(
                    trend_result["curves"][proj][idx]
                )
            writer.writerow(row)


def _style_axis(ax):
    ax.set_facecolor(COLOR_BG)
    for spine in ax.spines.values():
        spine.set_color(COLOR_SPINE)
        spine.set_linewidth(0.8)


def _projection_label(proj):
    labels = {
        "up_proj": "Up matrix",
        "gate_proj": "Gate matrix",
        "down_proj": "Down matrix",
    }
    return labels.get(proj, proj.replace("_", " "))


def _layer_tick_step(num_layers):
    if num_layers <= 16:
        return 1
    if num_layers <= 32:
        return 2
    return max(1, int(np.ceil(num_layers / 12.0)))


def plot(
    result,
    trend_result,
    output_prefix,
    fig_width=10.4,
    fig_height=3.45,
    segments=1,
    left_width_ratio=2.15,
    right_width_ratio=1.0,
):
    counts = result["counts"]
    channels = np.arange(counts.size)
    scale_top = result["scale_top"]
    freq_top = result["freq_top"]
    overlap_pct = result["overlap"] * 100.0
    topk = result["topk"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.7,
            "axes.titlesize": 9.8,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": COLOR_BG,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    segments = max(1, int(segments))
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        nrows=segments,
        ncols=2,
        width_ratios=[left_width_ratio, right_width_ratio],
        hspace=0.10,
        wspace=0.25,
    )
    left_axes = [fig.add_subplot(grid[idx, 0]) for idx in range(segments)]
    trend_ax = fig.add_subplot(grid[:, 1])
    fig.patch.set_facecolor("white")

    y_max = max(int(counts.max()), topk)
    y_upper = y_max + max(4, y_max * 0.08)
    segment_edges = np.linspace(0, counts.size, segments + 1, dtype=int)

    for seg_idx, ax in enumerate(left_axes):
        start = int(segment_edges[seg_idx])
        end = int(segment_edges[seg_idx + 1])
        mask = (channels >= start) & (channels < end)
        _style_axis(ax)

        ax.bar(
            channels[mask],
            counts[mask],
            width=1.0,
            color=COLOR_BAR,
            edgecolor=COLOR_BAR,
            linewidth=0.0,
            alpha=0.78,
            label="_nolegend_",
            zorder=1,
        )

        seg_scale = np.asarray(
            [idx for idx in scale_top if start <= idx < end],
            dtype=int,
        )
        if seg_scale.size:
            scale_heights = np.maximum(
                counts[seg_scale],
                max(2.0, float(counts.max()) * 0.08),
            )
            ax.vlines(
                seg_scale,
                0,
                scale_heights,
                color=COLOR_SCALE,
                linewidth=0.72,
                alpha=0.84,
                label=f"scale top-{topk}" if seg_idx == 0 else "_nolegend_",
                zorder=3,
            )

        seg_freq = np.asarray(
            [idx for idx in freq_top if start <= idx < end],
            dtype=int,
        )
        if seg_freq.size:
            ax.scatter(
                seg_freq,
                counts[seg_freq],
                s=14,
                color=COLOR_FREQ,
                edgecolor="#FEE9E8",
                linewidth=0.55,
                label=f"frequency top-{topk}" if seg_idx == 0 else "_nolegend_",
                zorder=4,
            )

        ax.set_xlim(start - 2, end + 2)
        ax.set_ylim(0, y_upper)
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.42,
            color=COLOR_GRID,
            alpha=0.55,
        )

        if segments > 1:
            ax.text(
                0.012,
                0.78,
                f"{start}-{end - 1}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=7.6,
                color=COLOR_MUTED,
                bbox=dict(
                    facecolor=COLOR_BG,
                    edgecolor="none",
                    alpha=0.76,
                    pad=0.5,
                ),
                zorder=5,
            )

        if seg_idx < len(left_axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Channel index")

    left_axes[0].set_title(
        f"Layer {result['layer']} {result['proj']}: top-{topk} selected-channel frequency",
        pad=8,
    )

    left_axes[0].text(
        0.012,
        0.55 if segments > 1 else 0.90,
        f"Top-{topk} overlap = {overlap_pct:.1f}%",
        transform=left_axes[0].transAxes,
        ha="left",
        va="center",
        fontsize=8.4,
        color=COLOR_TEXT,
        bbox=dict(
            facecolor=COLOR_BG,
            edgecolor="none",
            alpha=0.78,
            pad=1.0,
        ),
        zorder=5,
    )

    if segments == 1:
        left_axes[0].set_ylabel(
            f"Selected count across experts (n={result['num_experts']})"
        )
    else:
        fig.text(
            0.012,
            0.52,
            f"Selected count across experts (n={result['num_experts']})",
            rotation=90,
            va="center",
            ha="center",
            fontsize=9.2,
        )

    left_axes[0].legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CFC6B8",
        borderpad=0.35,
        handlelength=1.8,
    )

    # -------------------------------------------------
    # Right panel: layer-wise overlap curves.
    # -------------------------------------------------
    _style_axis(trend_ax)
    layers = trend_result["layers"]
    curve_colors = {
        "up_proj": COLOR_UP,
        "gate_proj": COLOR_GATE,
    }

    for curve_idx, (proj, values) in enumerate(trend_result["curves"].items()):
        color = curve_colors.get(proj, f"C{curve_idx}")
        trend_ax.plot(
            layers,
            values,
            color=color,
            linewidth=1.75,
            marker="o",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=_projection_label(proj),
            zorder=3,
        )

    selected_layer = int(result["layer"])
    if layers.size and layers.min() <= selected_layer <= layers.max():
        trend_ax.axvspan(
            selected_layer - 0.28,
            selected_layer + 0.28,
            color="#A9ADB2",
            alpha=0.13,
            linewidth=0.0,
            zorder=1,
        )
        trend_ax.axvline(
            selected_layer,
            color="#8D9297",
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
            zorder=2,
        )

        layer_pos = np.where(layers == selected_layer)[0]
        if layer_pos.size:
            pos = int(layer_pos[0])
            for curve_idx, (proj, values) in enumerate(
                trend_result["curves"].items()
            ):
                color = curve_colors.get(proj, f"C{curve_idx}")
                trend_ax.scatter(
                    [selected_layer],
                    [values[pos]],
                    s=38,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.35,
                    zorder=5,
                )

    trend_ax.set_title("Layer-wise top-K overlap", pad=8)
    trend_ax.set_xlabel("Layer index")
    trend_ax.set_ylabel("Overlap (%)")
    trend_ax.set_ylim(0, 100)

    if layers.size:
        tick_step = _layer_tick_step(len(layers))
        trend_ax.set_xticks(layers[::tick_step])
        trend_ax.set_xlim(layers.min() - 0.45, layers.max() + 0.45)

    trend_ax.set_yticks(np.arange(0, 101, 20))
    trend_ax.grid(
        axis="both",
        linestyle="--",
        linewidth=0.42,
        color=COLOR_GRID,
        alpha=0.55,
    )
    trend_ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CFC6B8",
        borderpad=0.35,
        handlelength=1.8,
    )
    trend_ax.text(
        0.97,
        0.04,
        f"Highlighted: layer {selected_layer}",
        transform=trend_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color=COLOR_MUTED,
    )

    if segments == 1:
        fig.subplots_adjust(
            left=0.072,
            right=0.985,
            top=0.88,
            bottom=0.18,
            wspace=0.25,
        )
    else:
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            top=0.90,
            bottom=0.12,
            hspace=0.10,
            wspace=0.25,
        )

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_prefix) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_prefix) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a selected layer's top-K channel frequency and layer-wise "
            "overlap curves for up_proj and gate_proj."
        )
    )
    parser.add_argument("--act_scales", default=DEFAULT_ACT_SCALES)
    parser.add_argument("--moe_act_p99s", default=DEFAULT_MOE_ACT_P99S)
    parser.add_argument(
        "--output_dir",
        default="/home/lwk/HeteroQuant_Repro/figures/scale_topk_overlap",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--auto_best", action="store_true")
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=15)
    parser.add_argument("--proj", default="up_proj")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument(
        "--trend_projs",
        nargs="+",
        default=["up_proj", "gate_proj"],
        help="Projection matrices shown in the right-hand layer-wise trend panel.",
    )
    parser.add_argument(
        "--shared_source",
        default="act_scale_gate",
        choices=["act_scale_gate", "moe_p99_gate"],
        help=(
            "act_scale_gate uses the cached activation max scale for "
            "model.layers.L.mlp.gate; moe_p99_gate uses the p99 scale used "
            "by fc1_scale_merge=act_p99."
        ),
    )
    parser.add_argument("--prefix", default="")
    parser.add_argument("--fig_width", type=float, default=10.4)
    parser.add_argument("--fig_height", type=float, default=3.45)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--left_width_ratio", type=float, default=2.15)
    parser.add_argument("--right_width_ratio", type=float, default=1.0)
    args = parser.parse_args()

    if args.layer_end < args.layer_start:
        raise ValueError("layer_end must be greater than or equal to layer_start")
    if not args.trend_projs:
        raise ValueError("trend_projs must contain at least one projection name")

    act_scales = load_tensor_dict(args.act_scales)
    moe_act_p99s = load_tensor_dict(args.moe_act_p99s)
    layers = range(int(args.layer_start), int(args.layer_end) + 1)

    if args.auto_best:
        result = find_best_layer(
            act_scales,
            moe_act_p99s,
            layers,
            args.proj,
            args.topk,
            args.num_experts,
            args.shared_source,
        )
    else:
        result = compute_overlap(
            act_scales,
            moe_act_p99s,
            args.layer,
            args.proj,
            args.topk,
            args.num_experts,
            args.shared_source,
        )

    trend_result = compute_layerwise_overlaps(
        act_scales,
        moe_act_p99s,
        layers,
        args.trend_projs,
        args.topk,
        args.num_experts,
        args.shared_source,
    )

    prefix = args.prefix
    if not prefix:
        prefix = (
            f"layer{result['layer']}_{args.proj}_top{args.topk}_"
            f"{args.shared_source}_with_layerwise_overlap"
        )
    output_prefix = Path(args.output_dir) / prefix

    plot(
        result,
        trend_result,
        output_prefix,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        segments=args.segments,
        left_width_ratio=args.left_width_ratio,
        right_width_ratio=args.right_width_ratio,
    )
    write_frequency_csv(str(output_prefix) + ".csv", result)
    write_layerwise_overlap_csv(
        str(output_prefix) + "_layerwise_overlap.csv",
        trend_result,
    )

    print(f"layer={result['layer']}")
    print(f"proj={result['proj']}")
    print(f"topk={result['topk']}")
    print(f"shared_source={result['shared_source']}")
    print(f"overlap={result['overlap']:.6f}")
    print(f"max_count={int(result['counts'].max())}")
    print(f"nonzero_channels={int((result['counts'] > 0).sum())}")
    for proj, values in trend_result["curves"].items():
        print(f"{proj}_mean_overlap_pct={float(np.mean(values)):.3f}")
    print(f"saved_png={output_prefix}.png")
    print(f"saved_pdf={output_prefix}.pdf")
    print(f"saved_csv={output_prefix}.csv")
    print(f"saved_layerwise_csv={output_prefix}_layerwise_overlap.csv")


if __name__ == "__main__":
    main()
