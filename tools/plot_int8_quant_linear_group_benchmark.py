#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_key(row):
    return (
        row["scope"],
        row["shape_name"],
        row["M"],
        row["num_active_experts"],
        row["tokens_per_expert"],
        row["act_scale_mode"],
    )


def label_for_group(value):
    value = int(value)
    return "per-tensor" if value <= 0 else str(value)


def plot_case(rows, output_dir, key):
    case_rows = sorted(rows, key=lambda r: int(r["group_size"]))
    if not case_rows:
        return
    scope, shape, m, experts, tokens, act_mode = key
    groups = [label_for_group(r["group_size"]) for r in case_rows]
    speedups = [float(r["speedup_over_group1"]) for r in case_rows]
    latencies = [float(r["latency_ms"]) for r in case_rows]

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(groups, speedups, color="#9eb7d8", edgecolor="#6f8fb7", label="Speedup vs group=1")
    ax1.set_ylabel("Speedup (x)")
    ax1.set_xlabel("Output-channel group size")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(groups, latencies, marker="o", color="#9a5b38", linewidth=2, label="Latency")
    ax2.set_ylabel("Latency (ms)")

    title_bits = [scope, shape, f"M={m}", f"act={act_mode}"]
    if int(experts) > 0:
        title_bits.extend([f"E={experts}", f"T={tokens}"])
    ax1.set_title(" | ".join(title_bits))
    fig.tight_layout()

    safe = "_".join(title_bits).replace("=", "").replace("|", "_").replace(" ", "")
    path = os.path.join(output_dir, f"{safe}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot INT8 quantized Linear group-size benchmark.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scope_filter", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_rows(args.input_csv)
    if args.scope_filter:
        allowed = {item.strip() for item in args.scope_filter.split(",") if item.strip()}
        rows = [row for row in rows if row["scope"] in allowed]

    grouped = defaultdict(list)
    for row in rows:
        if not row.get("speedup_over_group1"):
            continue
        grouped[group_key(row)].append(row)

    for key, case_rows in grouped.items():
        plot_case(case_rows, args.output_dir, key)

    print(f"[plot-int8-group-bench] plots: {args.output_dir}")


if __name__ == "__main__":
    main()
