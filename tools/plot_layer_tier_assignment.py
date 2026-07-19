#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BIT_COLORS = {
    4: "#4C78A8",
    5: "#F58518",
    6: "#54A24B",
    8: "#B279A2",
    16: "#9D755D",
}


def load_plan(path):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "plan" in obj:
        records = obj["plan"]
        summary = obj.get("summary", {})
    elif isinstance(obj, list):
        records = obj
        summary = {}
    else:
        raise ValueError(f"unsupported plan format in {path}")
    return records, summary


def mode(values):
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def collect_layer(records, layer):
    grouped = defaultdict(list)
    for rec in records:
        if int(rec.get("layer_index", -1)) != int(layer):
            continue
        expert = int(rec["expert_index"])
        grouped[expert].append(rec)

    if not grouped:
        layers = sorted({int(rec.get("layer_index", -1)) for rec in records})
        raise ValueError(f"layer {layer} not found; available layers: {layers[:10]} ... {layers[-5:]}")

    rows = []
    warnings = []
    for expert in sorted(grouped):
        recs = grouped[expert]
        abits_values = [int(r["abits"]) for r in recs]
        wbits_values = [int(r["wbits"]) for r in recs]
        quant_values = [str(r.get("quantization", f"w{r['wbits']}a{r['abits']}")) for r in recs]
        abits = mode(abits_values)
        wbits = mode(wbits_values)
        if len(set(abits_values)) > 1 or len(set(wbits_values)) > 1:
            warnings.append(
                f"expert {expert}: mixed projection bits "
                f"A={abits_values}, W={wbits_values}; using mode A{abits}/W{wbits}"
            )
        score_values = [r.get("selection_score") for r in recs if r.get("selection_score") is not None]
        route_values = [r.get("route_mass_fraction") for r in recs if r.get("route_mass_fraction") is not None]
        rows.append(
            {
                "expert": expert,
                "abits": abits,
                "wbits": wbits,
                "quantization": mode(quant_values),
                "selection_score": float(score_values[0]) if score_values else None,
                "route_mass_fraction": float(route_values[0]) if route_values else None,
                "num_projection_records": len(recs),
            }
        )
    return rows, warnings


def bit_color(bit):
    return BIT_COLORS.get(int(bit), "#BAB0AC")


def setup_common_axis(ax, experts, tick_every):
    ax.set_xlim(min(experts) - 0.6, max(experts) + 0.6)
    ticks = [e for e in experts if e % tick_every == 0] if tick_every > 1 else experts
    ax.set_xticks(ticks)
    ax.set_xlabel("Expert ID")


def parse_figsize(value):
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in value)
    parts = str(value).lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must be WIDTH,HEIGHT or WIDTHxHEIGHT")
    return float(parts[0]), float(parts[1])


def plot_diverging_bars(
    rows,
    output_base,
    title,
    tick_every,
    figsize,
    show_title,
    legend_y,
    bar_top,
    bar_bottom,
    y_pad,
):
    experts = [r["expert"] for r in rows]
    abits = np.array([r["abits"] for r in rows], dtype=float)
    wbits = np.array([r["wbits"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(experts, abits, width=0.82, color=[bit_color(v) for v in abits], edgecolor="white", linewidth=0.35)
    ax.bar(experts, -wbits, width=0.82, color=[bit_color(v) for v in wbits], edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#222222", linewidth=0.9)
    setup_common_axis(ax, experts, tick_every)

    max_abits = int(max(abits))
    max_wbits = int(max(wbits))
    yticks = sorted(set([int(v) for v in abits] + [0] + [-int(v) for v in wbits]))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"A{y}" if y > 0 else ("0" if y == 0 else f"W{-y}") for y in yticks])
    ax.set_ylim(-(max_wbits + y_pad), max_abits + y_pad)
    ax.set_ylabel("Precision tier")
    if show_title:
        ax.set_title(title)

    legend_bits = sorted(set(int(v) for v in np.concatenate([abits, wbits])))
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=5,
            markerfacecolor=bit_color(bit),
            markeredgecolor=bit_color(bit),
            label=f"{bit}-bit",
        )
        for bit in legend_bits
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        handlelength=0.7,
        handletextpad=0.25,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(top=bar_top, bottom=bar_bottom, left=0.06, right=0.99)
    fig.savefig(output_base + "_bars.png", dpi=300)
    fig.savefig(output_base + "_bars.pdf")
    plt.close(fig)


def plot_two_band_heatmap(rows, output_base, title, tick_every, figsize, show_title):
    experts = [r["expert"] for r in rows]
    bit_values = sorted(set([int(r["abits"]) for r in rows] + [int(r["wbits"]) for r in rows]))
    bit_to_idx = {bit: idx for idx, bit in enumerate(bit_values)}
    data = np.array(
        [
            [bit_to_idx[int(r["abits"])] for r in rows],
            [bit_to_idx[int(r["wbits"])] for r in rows],
        ],
        dtype=float,
    )
    colors = [bit_color(bit) for bit in bit_values]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Activation bits", "Weight bits"])
    if show_title:
        ax.set_title(title)
    ax.set_xlabel("Expert ID")

    tick_positions = [i for i, e in enumerate(experts) if e % tick_every == 0] if tick_every > 1 else list(range(len(experts)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([experts[i] for i in tick_positions])
    ax.set_xticks(np.arange(-0.5, len(experts), 1), minor=True)
    ax.grid(which="minor", axis="x", color="white", linewidth=0.25, alpha=0.7)
    ax.tick_params(which="minor", bottom=False)

    for row_idx, label in enumerate(["A", "W"]):
        for col_idx, r in enumerate(rows):
            bit = r["abits"] if row_idx == 0 else r["wbits"]
            ax.text(col_idx, row_idx, f"{label}{bit}", ha="center", va="center", fontsize=7, color="white")

    handles = [Patch(facecolor=bit_color(bit), label=f"{bit}-bit") for bit in bit_values]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=len(handles), frameon=False)

    fig.tight_layout()
    fig.savefig(output_base + "_bands.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_base + "_bands.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(rows, path):
    fields = [
        "expert",
        "abits",
        "wbits",
        "quantization",
        "selection_score",
        "route_mass_fraction",
        "num_projection_records",
    ]
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in fields) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot per-layer expert tier assignments from a MoE quantization plan.")
    parser.add_argument("--plan", required=True, help="Path to moe_quant_plan_*.json")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--show_title", action="store_true", help="Show the title at the top of the figure")
    parser.add_argument("--tick_every", type=int, default=4)
    parser.add_argument("--bar_figsize", type=parse_figsize, default=(13, 3.2), help="Bar figure size, e.g. 13,3.2 or 13x3.2")
    parser.add_argument("--band_figsize", type=parse_figsize, default=(13, 2.5), help="Band figure size, e.g. 13,2.5 or 13x2.5")
    parser.add_argument("--legend_y", type=float, default=0.9, help="Figure-level legend y position for the bar plot")
    parser.add_argument("--bar_top", type=float, default=0.84, help="Top margin for the bar plot axes; larger moves plot closer to legend")
    parser.add_argument("--bar_bottom", type=float, default=0.16, help="Bottom margin for the bar plot axes")
    parser.add_argument("--y_pad", type=float, default=0.35, help="Y-axis padding around max A bits and max W bits")
    args = parser.parse_args()

    records, summary = load_plan(args.plan)
    rows, warnings = collect_layer(records, args.layer)

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.plan), "figures")
    os.makedirs(output_dir, exist_ok=True)
    metric = summary.get("metric") or (records[0].get("selection_metric") if records else "metric")
    prefix = args.prefix or f"layer{args.layer}_{metric}_tier_assignment"
    output_base = os.path.join(output_dir, prefix)
    title = args.title or f"Layer {args.layer} Expert Tier Assignment ({metric})"

    for warning in warnings:
        print(f"[plot-layer-tier] WARNING: {warning}")
    print(f"[plot-layer-tier] experts: {len(rows)}")
    print(f"[plot-layer-tier] A bits: {Counter(r['abits'] for r in rows)}")
    print(f"[plot-layer-tier] W bits: {Counter(r['wbits'] for r in rows)}")

    plot_diverging_bars(
        rows,
        output_base,
        title,
        args.tick_every,
        args.bar_figsize,
        args.show_title,
        args.legend_y,
        args.bar_top,
        args.bar_bottom,
        args.y_pad,
    )
    plot_two_band_heatmap(rows, output_base, title, args.tick_every, args.band_figsize, args.show_title)
    write_csv(rows, output_base + ".csv")

    print(f"[plot-layer-tier] bars: {output_base}_bars.png")
    print(f"[plot-layer-tier] bands: {output_base}_bands.png")
    print(f"[plot-layer-tier] csv: {output_base}.csv")


if __name__ == "__main__":
    main()
