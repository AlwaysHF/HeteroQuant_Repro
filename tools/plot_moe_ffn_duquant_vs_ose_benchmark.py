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
        row["preset"],
        row["active_experts"],
        row["tokens_per_expert"],
    )


def is_full_variant(row):
    return not row["variant"].startswith("component_")


def plot_case(rows, output_dir, key):
    rows = [r for r in rows if is_full_variant(r)]
    order = ["duquant_once", "ours_ose", "plain_no_ose", "duquant_twice"]
    rows_by_variant = {r["variant"]: r for r in rows}
    variants = [v for v in order if v in rows_by_variant]
    if not variants:
        return

    labels = {
        "duquant_once": "DuQuant once",
        "duquant_twice": "DuQuant twice",
        "plain_no_ose": "No transform",
        "ours_ose": "Ours + OSE",
    }
    speedups = [float(rows_by_variant[v]["speedup_over_duquant_once"]) for v in variants]
    latencies = [float(rows_by_variant[v]["latency_ms"]) for v in variants]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.bar([labels[v] for v in variants], speedups, color="#9eb7d8", edgecolor="#6f8fb7")
    ax1.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax1.set_ylabel("Speedup vs DuQuant once")
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot([labels[v] for v in variants], latencies, color="#9a5b38", marker="o", linewidth=2)
    ax2.set_ylabel("Latency (ms)")

    scope, preset, active_experts, tokens_per_expert = key
    ax1.set_title(f"{scope} | {preset} | E={active_experts} | T/expert={tokens_per_expert}")
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()

    safe = f"{scope}_{preset}_E{active_experts}_T{tokens_per_expert}".replace("/", "_")
    fig.savefig(os.path.join(output_dir, f"{safe}.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, f"{safe}.pdf"))
    plt.close(fig)


def plot_components(rows, output_dir):
    component_rows = [r for r in rows if r["variant"].startswith("component_")]
    if not component_rows:
        return
    grouped = defaultdict(list)
    for row in component_rows:
        grouped[group_key(row)].append(row)

    for key, case_rows in grouped.items():
        case_rows = sorted(case_rows, key=lambda r: r["variant"])
        labels = [r["variant"].replace("component_", "") for r in case_rows]
        latencies = [float(r["latency_ms"]) for r in case_rows]

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(labels, latencies, color="#d7c7a6", edgecolor="#9a8052")
        ax.set_ylabel("Latency (ms)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        scope, preset, active_experts, tokens_per_expert = key
        ax.set_title(f"Components | {scope} | {preset} | E={active_experts} | T={tokens_per_expert}")
        fig.tight_layout()
        safe = f"components_{scope}_{preset}_E{active_experts}_T{tokens_per_expert}".replace("/", "_")
        fig.savefig(os.path.join(output_dir, f"{safe}.png"), dpi=200)
        fig.savefig(os.path.join(output_dir, f"{safe}.pdf"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot MoE FFN DuQuant-vs-OSE benchmark.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        if row.get("speedup_over_duquant_once"):
            grouped[group_key(row)].append(row)
    for key, case_rows in grouped.items():
        plot_case(case_rows, args.output_dir, key)
    plot_components(rows, args.output_dir)
    print(f"[plot-moe-ffn] plots: {args.output_dir}")


if __name__ == "__main__":
    main()
