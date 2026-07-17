#!/usr/bin/env python3
import argparse
import csv
import os
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


def load_tensor_dict(path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in {path}, got {type(obj)}")
    return obj


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


def compute_overlap(act_scales, moe_act_p99s, layer, proj, topk, num_experts, shared_source):
    shared_scale = shared_scale_for_layer(act_scales, moe_act_p99s, layer, shared_source)
    scale_top = topk_indices(shared_scale, topk)
    counts, expert_topk = expert_frequency(act_scales, layer, proj, topk, num_experts)
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


def find_best_layer(act_scales, moe_act_p99s, layers, proj, topk, num_experts, shared_source):
    best = None
    for layer in layers:
        result = compute_overlap(act_scales, moe_act_p99s, layer, proj, topk, num_experts, shared_source)
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


def plot(result, output_prefix, fig_width=6.9, fig_height=3.45, segments=1):
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
            "axes.facecolor": "#FDFBF7",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    segments = max(1, int(segments))
    if segments == 1:
        fig, axes = plt.subplots(figsize=(fig_width, fig_height))
        axes = [axes]
    else:
        fig, axes = plt.subplots(
            segments,
            1,
            figsize=(fig_width, fig_height),
            sharey=True,
            gridspec_kw={"hspace": 0.10},
        )
        axes = list(np.ravel(axes))
    fig.patch.set_facecolor("white")

    y_max = max(int(counts.max()), topk)
    y_upper = y_max + max(4, y_max * 0.08)
    segment_edges = np.linspace(0, counts.size, segments + 1, dtype=int)

    for seg_idx, ax in enumerate(axes):
        start = int(segment_edges[seg_idx])
        end = int(segment_edges[seg_idx + 1])
        mask = (channels >= start) & (channels < end)
        ax.set_facecolor("#FDFBF7")

        ax.bar(
            channels[mask],
            counts[mask],
            width=1.0,
            color="#DDD6CB",
            edgecolor="#DDD6CB",
            linewidth=0.0,
            alpha=0.78,
            label="_nolegend_",
            zorder=1,
        )

        seg_scale = np.asarray([idx for idx in scale_top if start <= idx < end], dtype=int)
        if seg_scale.size:
            scale_heights = np.maximum(
                counts[seg_scale],
                max(2.0, float(counts.max()) * 0.08),
            )
            ax.vlines(
                seg_scale,
                0,
                scale_heights,
                color="#487DBA",
                linewidth=0.72,
                alpha=0.84,
                label=f"scale top-{topk}" if seg_idx == 0 else "_nolegend_",
                zorder=3,
            )

        seg_freq = np.asarray([idx for idx in freq_top if start <= idx < end], dtype=int)
        if seg_freq.size:
            ax.scatter(
                seg_freq,
                counts[seg_freq],
                s=14,
                color="#D56858",
                edgecolor="#FEE9E8",
                linewidth=0.55,
                label=f"frequency top-{topk}" if seg_idx == 0 else "_nolegend_",
                zorder=4,
            )

        ax.set_xlim(start - 2, end + 2)
        ax.set_ylim(0, y_upper)
        ax.grid(axis="y", linestyle="--", linewidth=0.42, color="#D6D9DC", alpha=0.55)
        for spine in ax.spines.values():
            spine.set_color("#B8B1A6")
            spine.set_linewidth(0.8)

        if segments > 1:
            ax.text(
                0.012,
                0.78,
                f"{start}-{end - 1}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=7.6,
                color="#6F767D",
                bbox=dict(facecolor="#FDFBF7", edgecolor="none", alpha=0.76, pad=0.5),
                zorder=5,
            )

        if seg_idx < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Channel index")

    axes[0].set_title(
        f"Layer {result['layer']} {result['proj']}: top-{topk} selected-channel frequency",
        pad=8,
    )

    axes[0].text(
        0.012,
        0.55 if segments > 1 else 0.90,
        f"Top-{topk} overlap = {overlap_pct:.1f}%",
        transform=axes[0].transAxes,
        ha="left",
        va="center",
        fontsize=8.4,
        color="#4B514F",
        bbox=dict(facecolor="#FDFBF7", edgecolor="none", alpha=0.78, pad=1.0),
        zorder=5,
    )

    if segments == 1:
        axes[0].set_ylabel(f"Selected count across experts (n={result['num_experts']})")
    else:
        fig.text(
            0.014,
            0.52,
            f"Selected count across experts (n={result['num_experts']})",
            rotation=90,
            va="center",
            ha="center",
            fontsize=9.2,
        )

    axes[0].legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CFC6B8",
        borderpad=0.35,
        handlelength=1.8,
    )
    if segments == 1:
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.12, hspace=0.10)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_prefix) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_prefix) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot overlap between layer-wise activation-scale top-K channels and "
            "per-expert activation top-K channel frequency."
        )
    )
    parser.add_argument("--act_scales", default=DEFAULT_ACT_SCALES)
    parser.add_argument("--moe_act_p99s", default=DEFAULT_MOE_ACT_P99S)
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/scale_topk_overlap")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--auto_best", action="store_true")
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=15)
    parser.add_argument("--proj", default="up_proj")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument(
        "--shared_source",
        default="act_scale_gate",
        choices=["act_scale_gate", "moe_p99_gate"],
        help=(
            "act_scale_gate uses the cached activation max scale for model.layers.L.mlp.gate; "
            "moe_p99_gate uses the p99 scale used by fc1_scale_merge=act_p99."
        ),
    )
    parser.add_argument("--prefix", default="")
    parser.add_argument("--fig_width", type=float, default=6.9)
    parser.add_argument("--fig_height", type=float, default=3.45)
    parser.add_argument("--segments", type=int, default=1)
    args = parser.parse_args()

    act_scales = load_tensor_dict(args.act_scales)
    moe_act_p99s = load_tensor_dict(args.moe_act_p99s)

    if args.auto_best:
        layers = range(int(args.layer_start), int(args.layer_end) + 1)
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

    prefix = args.prefix
    if not prefix:
        prefix = (
            f"layer{result['layer']}_{args.proj}_top{args.topk}_"
            f"{args.shared_source}"
        )
    output_prefix = Path(args.output_dir) / prefix
    plot(
        result,
        output_prefix,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        segments=args.segments,
    )
    write_frequency_csv(str(output_prefix) + ".csv", result)

    print(f"layer={result['layer']}")
    print(f"proj={result['proj']}")
    print(f"topk={result['topk']}")
    print(f"shared_source={result['shared_source']}")
    print(f"overlap={result['overlap']:.6f}")
    print(f"max_count={int(result['counts'].max())}")
    print(f"nonzero_channels={int((result['counts'] > 0).sum())}")
    print(f"saved_png={output_prefix}.png")
    print(f"saved_pdf={output_prefix}.pdf")
    print(f"saved_csv={output_prefix}.csv")


if __name__ == "__main__":
    main()
