#!/usr/bin/env python3
"""Add theoretical compute and metadata metrics to post-quant FFN benchmark CSV.

The numbers are analytic counts for the runtime path represented by
triton_post_quant_moe_ffn_duquant_vs_ose_benchmark.py.  They are useful for
paper tables because they do not depend on GPU scheduling noise.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Iterable, List, Sequence


def read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transform_kernel_count(stages: int, include_perm: bool) -> int:
    return stages + max(0, stages - 1 if include_perm else 0)


def qparam_groups_per_expert(hidden: int, intermediate: int, group_size: int) -> int:
    channels = hidden if group_size <= 0 else group_size
    gate_up = math.ceil(intermediate / min(channels, intermediate))
    down = math.ceil(hidden / min(channels, hidden))
    return 2 * gate_up + down


def add_metrics(row: dict, qparam_bytes_per_group: int) -> dict:
    out = dict(row)
    variant = row["variant"]
    e = int(row["active_experts"])
    t = int(row["tokens_per_expert"])
    routed = e * t
    hidden = int(row["hidden_size"])
    intermediate = int(row["intermediate_size"])
    model_experts = int(row["num_experts_model"])
    topk = int(row["ose_topk"])
    block = int(row["duquant_block_size"])
    stages = int(row["duquant_rotation_stages"])
    include_perm = bool(int(row["include_duquant_permutation"]))
    include_down_du = bool(int(row["include_down_duquant_transform"]))
    group_size = int(row["main_weight_group_size"])

    main_gemm_macs = 3 * routed * hidden * intermediate
    gateup_transform_macs = stages * routed * hidden * block
    down_transform_macs = stages * routed * intermediate * block if include_down_du else 0
    ose_branch_macs = 2 * routed * topk * intermediate

    total_macs = 0
    if variant == "duquant_once":
        total_macs = main_gemm_macs + gateup_transform_macs + down_transform_macs
    elif variant == "plain_no_ose":
        total_macs = main_gemm_macs + down_transform_macs
    elif variant == "ours_ose_w8a8":
        # Current benchmark keeps the full main GEMMs and adds the OSE branch.
        total_macs = main_gemm_macs + ose_branch_macs + down_transform_macs
    elif variant == "component_gateup_duquant_transform":
        total_macs = gateup_transform_macs
    elif variant == "component_ose_w8a8_branch":
        total_macs = ose_branch_macs

    # A compact deployed OSE layout can avoid storing/multiplying the split
    # columns twice.  The current benchmark intentionally keeps the main GEMM
    # identical and adds OSE separately, so both views are reported.
    compact_ours_total_macs = main_gemm_macs + down_transform_macs
    duplicate_ose_extra_macs = ose_branch_macs if variant == "ours_ose_w8a8" else 0

    transform_kernels = transform_kernel_count(stages, include_perm)
    down_transform_kernels = transform_kernels if include_down_du else 0
    if variant == "duquant_once":
        kernel_launches = transform_kernels + 1 + 2 + 1 + down_transform_kernels + 1 + 1
    elif variant == "plain_no_ose":
        kernel_launches = 1 + 2 + 1 + down_transform_kernels + 1 + 1
    elif variant == "ours_ose_w8a8":
        kernel_launches = 1 + 2 + 1 + 1 + down_transform_kernels + 1 + 1
    elif variant == "component_gateup_duquant_transform":
        kernel_launches = transform_kernels
    elif variant == "component_ose_w8a8_branch":
        kernel_launches = 1
    else:
        kernel_launches = ""

    q_groups_expert = qparam_groups_per_expert(hidden, intermediate, group_size)
    q_groups_model = q_groups_expert * model_experts
    q_groups_active = q_groups_expert * e
    q_groups_pc_expert = qparam_groups_per_expert(hidden, intermediate, 1)

    main_weight_bytes_per_expert_int8 = 3 * hidden * intermediate
    ose_duplicate_weight_bytes_per_expert_int8 = 2 * topk * intermediate

    latency_ms = float(row["latency_ms"]) if row.get("latency_ms") else 0.0
    effective_tops = 0.0
    if latency_ms > 0 and total_macs > 0:
        effective_tops = (2.0 * total_macs) / (latency_ms * 1.0e-3) / 1.0e12

    out.update(
        {
            "routed_token_experts": routed,
            "main_gemm_macs": main_gemm_macs,
            "gateup_duquant_transform_macs": gateup_transform_macs,
            "common_down_duquant_transform_macs": down_transform_macs,
            "ose_branch_macs": ose_branch_macs,
            "total_online_macs_current_layout": total_macs,
            "total_online_ops_mac2_current_layout": 2 * total_macs,
            "effective_tops_mac2_current_layout": f"{effective_tops:.6f}",
            "compact_ours_total_macs_if_split_main_columns": compact_ours_total_macs,
            "duplicate_ose_extra_macs_current_layout": duplicate_ose_extra_macs,
            "duquant_unique_overhead_macs_per_routed_token": stages * hidden * block,
            "ose_unique_overhead_macs_per_routed_token": 2 * topk * intermediate,
            "unique_overhead_reduction_duquant_vs_ose": f"{(stages * hidden * block) / max(2 * topk * intermediate, 1):.6f}",
            "estimated_kernel_launches": kernel_launches,
            "main_qparam_groups_per_expert": q_groups_expert,
            "main_qparam_groups_active_experts": q_groups_active,
            "main_qparam_groups_all_model_experts": q_groups_model,
            "main_qparam_bytes_active_experts": q_groups_active * qparam_bytes_per_group,
            "main_qparam_bytes_all_model_experts": q_groups_model * qparam_bytes_per_group,
            "main_qparam_reduction_vs_per_channel": f"{q_groups_pc_expert / q_groups_expert:.6f}",
            "main_int8_weight_bytes_per_expert": main_weight_bytes_per_expert_int8,
            "ose_duplicate_int8_weight_bytes_per_expert_current_layout": ose_duplicate_weight_bytes_per_expert_int8,
            "ose_duplicate_weight_overhead_ratio_current_layout": f"{ose_duplicate_weight_bytes_per_expert_int8 / main_weight_bytes_per_expert_int8:.6f}",
        }
    )
    return out


def build_summary(rows: List[dict]) -> List[dict]:
    summary = []
    keys = sorted(
        {
            (
                int(r["main_weight_group_size"]),
                r["variant"],
            )
            for r in rows
        }
    )
    for group_size, variant in keys:
        items = [r for r in rows if int(r["main_weight_group_size"]) == group_size and r["variant"] == variant]
        if not items:
            continue
        summary.append(
            {
                "main_weight_group_size": group_size,
                "variant": variant,
                "num_cases": len(items),
                "macs_per_routed_token_main_gemm": int(items[0]["main_gemm_macs"]) // max(int(items[0]["routed_token_experts"]), 1),
                "macs_per_routed_token_gateup_duquant_transform": int(items[0]["gateup_duquant_transform_macs"]) // max(int(items[0]["routed_token_experts"]), 1),
                "macs_per_routed_token_ose_branch": int(items[0]["ose_branch_macs"]) // max(int(items[0]["routed_token_experts"]), 1),
                "unique_overhead_reduction_duquant_vs_ose": items[0]["unique_overhead_reduction_duquant_vs_ose"],
                "estimated_kernel_launches": items[0]["estimated_kernel_launches"],
                "main_qparam_groups_per_expert": items[0]["main_qparam_groups_per_expert"],
                "main_qparam_bytes_all_model_experts": items[0]["main_qparam_bytes_all_model_experts"],
                "main_qparam_reduction_vs_per_channel": items[0]["main_qparam_reduction_vs_per_channel"],
                "ose_duplicate_weight_overhead_ratio_current_layout": items[0]["ose_duplicate_weight_overhead_ratio_current_layout"],
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute theoretical metrics for post-quant FFN benchmark CSV.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_csv")
    parser.add_argument("--qparam_bytes_per_group", type=int, default=4)
    args = parser.parse_args()

    rows = [add_metrics(row, args.qparam_bytes_per_group) for row in read_csv(args.input_csv)]
    fields = list(rows[0].keys()) if rows else []
    write_csv(args.output_csv, rows, fields)
    print(f"[post-quant-metrics] detail: {args.output_csv}")

    if args.summary_csv:
        summary = build_summary(rows)
        summary_fields = list(summary[0].keys()) if summary else []
        write_csv(args.summary_csv, summary, summary_fields)
        print(f"[post-quant-metrics] summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
