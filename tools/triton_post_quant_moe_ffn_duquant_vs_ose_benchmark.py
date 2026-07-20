#!/usr/bin/env python3
"""Post-quant MoE FFN benchmark: DuQuant gate/up transform vs OSE branch.

This benchmark simulates the runtime path after calibration, channel selection,
weight transformation, quantization, and packing have already happened.

It uses a shared W8A8 Triton backend for both methods:

  DuQuant-like:
    x_fp16 -> gate/up DuQuant activation transform once
    -> dynamic activation quant -> int8 gate/up GEMMs
    -> silu(gate) * up
    -> optional common down DuQuant transform
    -> dynamic activation quant -> int8 down GEMM

  Ours/OSE:
    x_fp16 -> dynamic activation quant -> int8 gate/up main GEMMs
    -> W8A8 top-k input-channel OSE branch added to gate/up
    -> silu(gate) * up
    -> optional common down DuQuant transform
    -> dynamic activation quant -> int8 down GEMM

The main comparison is controlled: same input/output dtype, same random
pre-quantized int8 weights, same scale layout, same GEMM kernels, same shapes.
The benchmark does not claim packed INT4 latency because this project does not
currently provide a packed INT4 GEMM deployment kernel.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MoEPreset:
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int


PRESETS: Dict[str, MoEPreset] = {
    "olmoe": MoEPreset(hidden=2048, intermediate=1024, num_experts=64, top_k=8),
    "qwen": MoEPreset(hidden=2048, intermediate=1408, num_experts=60, top_k=4),
}


@triton.jit
def _dynamic_quant_sym_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    vals = tl.load(
        x_ptr + pid_e * M * K + pid_m * K + offs_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.abs(vals), axis=0)
    scale = tl.maximum(absmax / 127.0, 1.0e-8)
    qf = vals / scale
    qf = tl.where(qf >= 0.0, qf + 0.5, qf - 0.5)
    qf = tl.minimum(tl.maximum(qf, -127.0), 127.0)
    tl.store(q_ptr + pid_e * M * K + pid_m * K + offs_k, qf.to(tl.int8), mask=mask)
    tl.store(scale_ptr + pid_e * M + pid_m, scale)


@triton.jit
def _grouped_int8_gemm_output_scale_kernel(
    x_ptr,
    w_ptr,
    act_scale_ptr,
    weight_scale_ptr,
    y_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_WEIGHT_SCALES: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    WEIGHT_PER_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

    x_base = pid_e * M * K
    w_base = pid_e * N * K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + x_base + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            w_ptr + w_base + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, w, out_dtype=tl.int32)

    sx = tl.load(
        act_scale_ptr + pid_e * M + offs_m,
        mask=offs_m < M,
        other=1.0,
    ).to(tl.float32)
    scale_base = pid_e * NUM_WEIGHT_SCALES
    if WEIGHT_PER_TENSOR:
        sw = tl.load(weight_scale_ptr + scale_base).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        sw = tl.load(
            weight_scale_ptr + scale_base + scale_ids,
            mask=offs_n < N,
            other=1.0,
        ).to(tl.float32)

    out = acc.to(tl.float32) * sx[:, None] * sw[None, :]
    tl.store(
        y_ptr + pid_e * M * N + offs_m[:, None] * N + offs_n[None, :],
        out.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _duquant_block_rotate_kernel(
    x_ptr,
    r_ptr,
    y_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = tl.arange(0, BLOCK_H)
    offs_o = tl.arange(0, BLOCK_H)
    h0 = pid_b * BLOCK_H

    x = tl.load(
        x_ptr + pid_e * M * H + offs_m[:, None] * H + h0 + offs_i[None, :],
        mask=(offs_m[:, None] < M) & (h0 + offs_i[None, :] < H),
        other=0.0,
    )
    r = tl.load(
        r_ptr + pid_e * BLOCK_H * BLOCK_H + offs_i[:, None] * BLOCK_H + offs_o[None, :],
        mask=(offs_i[:, None] < BLOCK_H) & (offs_o[None, :] < BLOCK_H),
        other=0.0,
    )
    out = tl.dot(x, r, out_dtype=tl.float32)
    tl.store(
        y_ptr + pid_e * M * H + offs_m[:, None] * H + h0 + offs_o[None, :],
        out.to(tl.float16),
        mask=(offs_m[:, None] < M) & (h0 + offs_o[None, :] < H),
    )


@triton.jit
def _permute_hidden_kernel(
    x_ptr,
    perm_ptr,
    y_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = E * M * H
    mask = offs < total
    h = offs % H
    mt = offs // H
    e = mt // M
    m = mt - e * M
    src_h = tl.load(perm_ptr + e * H + h, mask=mask, other=0)
    vals = tl.load(x_ptr + e * M * H + m * H + src_h, mask=mask, other=0.0)
    tl.store(y_ptr + offs, vals, mask=mask)


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (gate * tl.sigmoid(gate) * up).to(tl.float16), mask=mask)


@triton.jit
def _ose_w8a8_two_proj_add_kernel(
    x_ptr,
    gate_idx_ptr,
    up_idx_ptr,
    gate_w_ptr,
    up_w_ptr,
    gate_scale_ptr,
    up_scale_ptr,
    gate_ptr,
    up_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    GATE_NUM_WEIGHT_SCALES: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    WEIGHT_PER_TENSOR: tl.constexpr,
    ADD_TO_MAIN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < TOPK

    gate_idx = tl.load(gate_idx_ptr + pid_e * TOPK + offs_k, mask=k_mask, other=0)
    up_idx = tl.load(up_idx_ptr + pid_e * TOPK + offs_k, mask=k_mask, other=0)
    x_gate = tl.load(
        x_ptr + pid_e * M * H + offs_m[:, None] * H + gate_idx[None, :],
        mask=(offs_m[:, None] < M) & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x_up = tl.load(
        x_ptr + pid_e * M * H + offs_m[:, None] * H + up_idx[None, :],
        mask=(offs_m[:, None] < M) & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    gate_abs = tl.max(tl.abs(x_gate), axis=1)
    up_abs = tl.max(tl.abs(x_up), axis=1)
    gate_sx = tl.maximum(gate_abs / 127.0, 1.0e-8)
    up_sx = tl.maximum(up_abs / 127.0, 1.0e-8)
    gate_qf = x_gate / gate_sx[:, None]
    up_qf = x_up / up_sx[:, None]
    gate_qf = tl.where(gate_qf >= 0.0, gate_qf + 0.5, gate_qf - 0.5)
    up_qf = tl.where(up_qf >= 0.0, up_qf + 0.5, up_qf - 0.5)
    gate_qf = tl.minimum(tl.maximum(gate_qf, -127.0), 127.0)
    up_qf = tl.minimum(tl.maximum(up_qf, -127.0), 127.0)

    wg = tl.load(
        gate_w_ptr + pid_e * I * TOPK + offs_n[None, :] * TOPK + offs_k[:, None],
        mask=(offs_n[None, :] < I) & k_mask[:, None],
        other=0,
    )
    wu = tl.load(
        up_w_ptr + pid_e * I * TOPK + offs_n[None, :] * TOPK + offs_k[:, None],
        mask=(offs_n[None, :] < I) & k_mask[:, None],
        other=0,
    )
    gate_acc = tl.dot(gate_qf.to(tl.int8), wg, out_dtype=tl.int32)
    up_acc = tl.dot(up_qf.to(tl.int8), wu, out_dtype=tl.int32)

    scale_base = pid_e * GATE_NUM_WEIGHT_SCALES
    if WEIGHT_PER_TENSOR:
        gate_sw = tl.load(gate_scale_ptr + scale_base).to(tl.float32)
        up_sw = tl.load(up_scale_ptr + scale_base).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        gate_sw = tl.load(
            gate_scale_ptr + scale_base + scale_ids,
            mask=offs_n < I,
            other=1.0,
        ).to(tl.float32)
        up_sw = tl.load(
            up_scale_ptr + scale_base + scale_ids,
            mask=offs_n < I,
            other=1.0,
        ).to(tl.float32)

    gate_out = gate_acc.to(tl.float32) * gate_sx[:, None] * gate_sw[None, :]
    up_out = up_acc.to(tl.float32) * up_sx[:, None] * up_sw[None, :]
    out_offs = pid_e * M * I + offs_m[:, None] * I + offs_n[None, :]
    if ADD_TO_MAIN:
        gate_out += tl.load(
            gate_ptr + out_offs,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        up_out += tl.load(
            up_ptr + out_offs,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < I),
            other=0.0,
        ).to(tl.float32)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < I)
    tl.store(gate_ptr + out_offs, gate_out.to(tl.float16), mask=mask)
    tl.store(up_ptr + out_offs, up_out.to(tl.float16), mask=mask)


def parse_csv_ints(value: str) -> List[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def parse_csv_strings(value: str) -> List[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty list: {value!r}")
    return out


def ceil_pow2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def channels_and_scales(group_size: int, n: int) -> Tuple[int, int, bool]:
    channels = n if group_size <= 0 else min(int(group_size), n)
    num_scales = triton.cdiv(n, channels)
    return channels, num_scales, num_scales == 1


def make_scales(e: int, num_scales: int, device: str = "cuda") -> torch.Tensor:
    return (torch.rand((e, num_scales), device=device, dtype=torch.float32) * 0.01 + 0.001).contiguous()


def make_int8(shape: Sequence[int], device: str = "cuda") -> torch.Tensor:
    return torch.randint(-127, 128, tuple(shape), device=device, dtype=torch.int8).contiguous()


def make_fp16(shape: Sequence[int], device: str = "cuda") -> torch.Tensor:
    return (torch.randn(*shape, device=device, dtype=torch.float16) * 0.02).contiguous()


def make_rotation_bank(stages: int, active_experts: int, block_size: int, device: str = "cuda") -> torch.Tensor:
    # Random dense blocks are enough for timing the same block-rotation math.
    return (torch.randn((stages, active_experts, block_size, block_size), device=device, dtype=torch.float16) * 0.02).contiguous()


def make_permutation_bank(active_experts: int, hidden: int, device: str = "cuda") -> torch.Tensor:
    perms = [torch.randperm(hidden, device=device, dtype=torch.int64) for _ in range(active_experts)]
    return torch.stack(perms, dim=0).contiguous()


def make_ose_indices(active_experts: int, hidden: int, topk: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    gate = []
    up = []
    for _ in range(active_experts):
        gate.append(torch.randperm(hidden, device=device, dtype=torch.int64)[:topk])
        up.append(torch.randperm(hidden, device=device, dtype=torch.int64)[:topk])
    return torch.stack(gate, dim=0).contiguous(), torch.stack(up, dim=0).contiguous()


def zero_main_columns(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = weight.clone()
    for expert in range(out.shape[0]):
        out[expert, :, indices[expert]] = 0
    return out.contiguous()


def gather_ose_columns(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = []
    for expert in range(weight.shape[0]):
        rows.append(weight[expert].index_select(1, indices[expert]))
    return torch.stack(rows, dim=0).contiguous()


def launch_dynamic_quant(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, e: int, m: int, k: int) -> None:
    grid = (m, e)
    _dynamic_quant_sym_kernel[grid](
        x,
        q,
        scale,
        E=e,
        M=m,
        K=k,
        BLOCK_K=ceil_pow2(k),
        num_warps=8,
    )


def launch_int8_gemm(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    y: torch.Tensor,
    e: int,
    m: int,
    k: int,
    n: int,
    channels: int,
    num_scales: int,
    per_tensor: bool,
    args,
) -> None:
    grid = (triton.cdiv(m, args.block_m), triton.cdiv(n, args.block_n), e)
    _grouped_int8_gemm_output_scale_kernel[grid](
        x_q,
        w_q,
        x_scale,
        w_scale,
        y,
        E=e,
        M=m,
        K=k,
        N=n,
        NUM_WEIGHT_SCALES=num_scales,
        CHANNELS_PER_SCALE=channels,
        WEIGHT_PER_TENSOR=per_tensor,
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        BLOCK_K=args.block_k,
        num_warps=args.num_warps,
        num_stages=4,
    )


def launch_duquant_transform(
    x: torch.Tensor,
    tmp_a: torch.Tensor,
    tmp_b: torch.Tensor,
    rotations: torch.Tensor,
    perms: torch.Tensor,
    e: int,
    m: int,
    hidden: int,
    args,
) -> torch.Tensor:
    if hidden % args.duquant_block_size != 0:
        raise ValueError(f"hidden={hidden} must be divisible by block_size={args.duquant_block_size}")
    src = x
    dst = tmp_a
    scratch = tmp_b
    for stage in range(args.duquant_rotation_stages):
        grid = (triton.cdiv(m, args.block_m), hidden // args.duquant_block_size, e)
        _duquant_block_rotate_kernel[grid](
            src,
            rotations[stage],
            dst,
            E=e,
            M=m,
            H=hidden,
            BLOCK_M=args.block_m,
            BLOCK_H=args.duquant_block_size,
            num_warps=args.num_warps,
            num_stages=4,
        )
        src = dst
        dst = scratch if dst is tmp_a else tmp_a
        scratch = tmp_a if scratch is tmp_b else tmp_b
        if args.include_duquant_permutation and stage < args.duquant_rotation_stages - 1:
            grid_perm = (triton.cdiv(e * m * hidden, args.elem_block),)
            _permute_hidden_kernel[grid_perm](
                src,
                perms,
                dst,
                E=e,
                M=m,
                H=hidden,
                BLOCK=args.elem_block,
                num_warps=4,
            )
            src = dst
            dst = scratch if dst is tmp_a else tmp_a
            scratch = tmp_a if scratch is tmp_b else tmp_b
    return src


def launch_silu_mul(gate: torch.Tensor, up: torch.Tensor, inter: torch.Tensor, total: int, args) -> None:
    grid = (triton.cdiv(total, args.elem_block),)
    _silu_mul_kernel[grid](gate, up, inter, TOTAL=total, BLOCK=args.elem_block, num_warps=4)


def launch_ose_branch(
    x: torch.Tensor,
    gate_idx: torch.Tensor,
    up_idx: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    gate_scale: torch.Tensor,
    up_scale: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    e: int,
    m: int,
    hidden: int,
    intermediate: int,
    topk: int,
    channels: int,
    num_scales: int,
    per_tensor: bool,
    add_to_main: bool,
    args,
) -> None:
    grid = (triton.cdiv(m, args.block_m), triton.cdiv(intermediate, args.block_n), e)
    _ose_w8a8_two_proj_add_kernel[grid](
        x,
        gate_idx,
        up_idx,
        gate_w,
        up_w,
        gate_scale,
        up_scale,
        gate,
        up,
        E=e,
        M=m,
        H=hidden,
        I=intermediate,
        TOPK=topk,
        GATE_NUM_WEIGHT_SCALES=num_scales,
        CHANNELS_PER_SCALE=channels,
        WEIGHT_PER_TENSOR=per_tensor,
        ADD_TO_MAIN=add_to_main,
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        BLOCK_K=ceil_pow2(topk),
        num_warps=args.num_warps,
        num_stages=4,
    )


def measure_ms(fn: Callable[[], None], warmup: int, repeat: int, rounds: int) -> Tuple[float, float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / repeat)
    values = sorted(values)
    q20 = values[max(0, min(len(values) - 1, int(round(0.2 * (len(values) - 1)))))]
    q80 = values[max(0, min(len(values) - 1, int(round(0.8 * (len(values) - 1)))))]
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.median(values), q20, q80, stdev


def torch_round_to_int8(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x).clamp(-127, 127).to(torch.int8)


def torch_dynamic_quant_sym(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = x.float().abs().amax(dim=-1).clamp(min=1.0e-8) / 127.0
    q = torch_round_to_int8(x.float() / scale[..., None])
    return q.contiguous(), scale.contiguous()


def expand_weight_scale(scales: torch.Tensor, n: int, channels: int, per_tensor: bool) -> torch.Tensor:
    if per_tensor:
        return scales[:, :1].expand(scales.shape[0], n)
    idx = torch.arange(n, device=scales.device) // channels
    return scales.index_select(1, idx)


def torch_int8_gemm(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    channels: int,
    per_tensor: bool,
) -> torch.Tensor:
    acc = torch.bmm(x_q.float(), w_q.float().transpose(1, 2))
    sw = expand_weight_scale(w_scale, w_q.shape[1], channels, per_tensor)
    return (acc * x_scale[:, :, None] * sw[:, None, :]).to(torch.float16)


def torch_duquant_transform(x: torch.Tensor, rotations: torch.Tensor, perms: torch.Tensor, args) -> torch.Tensor:
    e, m, hidden = x.shape
    block = args.duquant_block_size
    y = x
    for stage in range(args.duquant_rotation_stages):
        r = rotations[stage].to(torch.float32)
        y = y.reshape(e, m, hidden // block, block).to(torch.float32)
        y = torch.einsum("embk,ekr->embr", y, r).reshape(e, m, hidden).to(torch.float16)
        if args.include_duquant_permutation and stage < args.duquant_rotation_stages - 1:
            y = y.gather(2, perms[:, None, :].expand(e, m, hidden))
    return y.contiguous()


def torch_ose_add(
    x: torch.Tensor,
    gate_idx: torch.Tensor,
    up_idx: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    gate_scale: torch.Tensor,
    up_scale: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    channels: int,
    per_tensor: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    e, m, _ = x.shape
    topk = gate_idx.shape[1]
    x_gate = x.gather(2, gate_idx[:, None, :].expand(e, m, topk))
    x_up = x.gather(2, up_idx[:, None, :].expand(e, m, topk))
    qg, sg = torch_dynamic_quant_sym(x_gate)
    qu, su = torch_dynamic_quant_sym(x_up)
    gate_ose = torch_int8_gemm(qg, gate_w, sg, gate_scale, channels, per_tensor)
    up_ose = torch_int8_gemm(qu, up_w, su, up_scale, channels, per_tensor)
    return (gate.float() + gate_ose.float()).to(torch.float16), (up.float() + up_ose.float()).to(torch.float16)


def torch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1.0e-6)
    return float(diff.max().item()), float(rel.max().item())


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_case_tensors(preset: MoEPreset, active_experts: int, tokens_per_expert: int, args) -> dict:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    topk = min(args.ose_topk, h)

    gate_channels, gate_scales, gate_per_tensor = channels_and_scales(args.main_weight_group_size, i)
    down_channels, down_scales, down_per_tensor = channels_and_scales(args.main_weight_group_size, h)
    ose_channels, ose_scales, ose_per_tensor = channels_and_scales(args.ose_weight_group_size, i)

    x = make_fp16((e, m, h))
    w_gate_full = make_int8((e, i, h))
    w_up_full = make_int8((e, i, h))
    w_down = make_int8((e, h, i))
    gate_idx, up_idx = make_ose_indices(e, h, topk)
    w_gate_ose = gather_ose_columns(w_gate_full, gate_idx)
    w_up_ose = gather_ose_columns(w_up_full, up_idx)
    w_gate_main = zero_main_columns(w_gate_full, gate_idx)
    w_up_main = zero_main_columns(w_up_full, up_idx)

    return {
        "x": x,
        "w_gate_full": w_gate_full,
        "w_up_full": w_up_full,
        "w_gate_main": w_gate_main,
        "w_up_main": w_up_main,
        "w_down": w_down,
        "w_gate_ose": w_gate_ose,
        "w_up_ose": w_up_ose,
        "gate_idx": gate_idx,
        "up_idx": up_idx,
        "gate_scale": make_scales(e, gate_scales),
        "up_scale": make_scales(e, gate_scales),
        "down_scale": make_scales(e, down_scales),
        "gate_ose_scale": make_scales(e, ose_scales),
        "up_ose_scale": make_scales(e, ose_scales),
        "rot_hidden": make_rotation_bank(args.duquant_rotation_stages, e, args.duquant_block_size),
        "perm_hidden": make_permutation_bank(e, h),
        "rot_inter": make_rotation_bank(args.duquant_rotation_stages, e, args.duquant_block_size),
        "perm_inter": make_permutation_bank(e, i),
        "gate_channels": gate_channels,
        "gate_scales": gate_scales,
        "gate_per_tensor": gate_per_tensor,
        "down_channels": down_channels,
        "down_scales": down_scales,
        "down_per_tensor": down_per_tensor,
        "ose_channels": ose_channels,
        "ose_scales": ose_scales,
        "ose_per_tensor": ose_per_tensor,
    }


def make_runtime_buffers(active_experts: int, tokens_per_expert: int, preset: MoEPreset, args) -> dict:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    return {
        "x_tmp_a": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "x_tmp_b": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "x_tmp_c": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "x_tmp_d": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "inter_tmp_a": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "inter_tmp_b": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "x_q": torch.empty((e, m, h), device="cuda", dtype=torch.int8),
        "x_scale": torch.empty((e, m), device="cuda", dtype=torch.float32),
        "inter_q": torch.empty((e, m, i), device="cuda", dtype=torch.int8),
        "inter_scale": torch.empty((e, m), device="cuda", dtype=torch.float32),
        "gate": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "up": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "inter": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "down": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
    }


def quantized_gate_up(
    x_for_main: torch.Tensor,
    tensors: dict,
    buffers: dict,
    preset: MoEPreset,
    active_experts: int,
    tokens_per_expert: int,
    gate_weight_name: str,
    up_weight_name: str,
    args,
) -> None:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    launch_dynamic_quant(x_for_main, buffers["x_q"], buffers["x_scale"], e, m, h)
    launch_int8_gemm(
        buffers["x_q"],
        tensors[gate_weight_name],
        buffers["x_scale"],
        tensors["gate_scale"],
        buffers["gate"],
        e,
        m,
        h,
        i,
        tensors["gate_channels"],
        tensors["gate_scales"],
        tensors["gate_per_tensor"],
        args,
    )
    launch_int8_gemm(
        buffers["x_q"],
        tensors[up_weight_name],
        buffers["x_scale"],
        tensors["up_scale"],
        buffers["up"],
        e,
        m,
        h,
        i,
        tensors["gate_channels"],
        tensors["gate_scales"],
        tensors["gate_per_tensor"],
        args,
    )


def quantized_one_proj(
    x_for_main: torch.Tensor,
    tensors: dict,
    buffers: dict,
    preset: MoEPreset,
    active_experts: int,
    tokens_per_expert: int,
    weight_name: str,
    scale_name: str,
    output_name: str,
    args,
) -> None:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    launch_dynamic_quant(x_for_main, buffers["x_q"], buffers["x_scale"], e, m, h)
    launch_int8_gemm(
        buffers["x_q"],
        tensors[weight_name],
        buffers["x_scale"],
        tensors[scale_name],
        buffers[output_name],
        e,
        m,
        h,
        i,
        tensors["gate_channels"],
        tensors["gate_scales"],
        tensors["gate_per_tensor"],
        args,
    )


def quantized_down(
    tensors: dict,
    buffers: dict,
    preset: MoEPreset,
    active_experts: int,
    tokens_per_expert: int,
    args,
) -> None:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    inter = buffers["inter"]
    if args.include_down_duquant_transform:
        inter = launch_duquant_transform(
            inter,
            buffers["inter_tmp_a"],
            buffers["inter_tmp_b"],
            tensors["rot_inter"],
            tensors["perm_inter"],
            e,
            m,
            i,
            args,
        )
    launch_dynamic_quant(inter, buffers["inter_q"], buffers["inter_scale"], e, m, i)
    launch_int8_gemm(
        buffers["inter_q"],
        tensors["w_down"],
        buffers["inter_scale"],
        tensors["down_scale"],
        buffers["down"],
        e,
        m,
        i,
        h,
        tensors["down_channels"],
        tensors["down_scales"],
        tensors["down_per_tensor"],
        args,
    )


def build_variant_fn(
    variant: str,
    tensors: dict,
    buffers: dict,
    preset: MoEPreset,
    active_experts: int,
    tokens_per_expert: int,
    args,
) -> Tuple[Callable[[], None], str]:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    topk = min(args.ose_topk, h)

    def finish():
        launch_silu_mul(buffers["gate"], buffers["up"], buffers["inter"], e * m * i, args)
        quantized_down(tensors, buffers, preset, e, m, args)

    if variant == "duquant_once":
        def fn():
            x_du = launch_duquant_transform(
                tensors["x"],
                buffers["x_tmp_a"],
                buffers["x_tmp_b"],
                tensors["rot_hidden"],
                tensors["perm_hidden"],
                e,
                m,
                h,
                args,
            )
            quantized_gate_up(x_du, tensors, buffers, preset, e, m, "w_gate_full", "w_up_full", args)
            finish()
        return fn, "gateup_duquant_once+dyn_quant+2x_i8_gemm+silu+common_down"

    if variant == "duquant_twice":
        def fn():
            x_gate = launch_duquant_transform(
                tensors["x"],
                buffers["x_tmp_a"],
                buffers["x_tmp_b"],
                tensors["rot_hidden"],
                tensors["perm_hidden"],
                e,
                m,
                h,
                args,
            )
            quantized_one_proj(x_gate, tensors, buffers, preset, e, m, "w_gate_full", "gate_scale", "gate", args)
            x_up = launch_duquant_transform(
                tensors["x"],
                buffers["x_tmp_c"],
                buffers["x_tmp_d"],
                tensors["rot_hidden"],
                tensors["perm_hidden"],
                e,
                m,
                h,
                args,
            )
            quantized_one_proj(x_up, tensors, buffers, preset, e, m, "w_up_full", "up_scale", "up", args)
            finish()
        return fn, "diagnostic_gateup_duquant_twice+2x_dyn_quant+2x_i8_gemm+silu+common_down"

    if variant == "plain_no_ose":
        def fn():
            quantized_gate_up(tensors["x"], tensors, buffers, preset, e, m, "w_gate_full", "w_up_full", args)
            finish()
        return fn, "dyn_quant+2x_i8_gemm+silu+common_down"

    if variant == "ours_ose_w8a8":
        def fn():
            quantized_gate_up(tensors["x"], tensors, buffers, preset, e, m, "w_gate_main", "w_up_main", args)
            launch_ose_branch(
                tensors["x"],
                tensors["gate_idx"],
                tensors["up_idx"],
                tensors["w_gate_ose"],
                tensors["w_up_ose"],
                tensors["gate_ose_scale"],
                tensors["up_ose_scale"],
                buffers["gate"],
                buffers["up"],
                e,
                m,
                h,
                i,
                topk,
                tensors["ose_channels"],
                tensors["ose_scales"],
                tensors["ose_per_tensor"],
                True,
                args,
            )
            finish()
        return fn, "dyn_quant+2x_i8_main_gemm+ose_w8a8_two_proj+silu+common_down"

    if variant == "component_gateup_duquant_transform":
        def fn():
            launch_duquant_transform(
                tensors["x"],
                buffers["x_tmp_a"],
                buffers["x_tmp_b"],
                tensors["rot_hidden"],
                tensors["perm_hidden"],
                e,
                m,
                h,
                args,
            )
        return fn, "gateup_duquant_transform_only"

    if variant == "component_ose_w8a8_branch":
        def fn():
            launch_ose_branch(
                tensors["x"],
                tensors["gate_idx"],
                tensors["up_idx"],
                tensors["w_gate_ose"],
                tensors["w_up_ose"],
                tensors["gate_ose_scale"],
                tensors["up_ose_scale"],
                buffers["gate"],
                buffers["up"],
                e,
                m,
                h,
                i,
                topk,
                tensors["ose_channels"],
                tensors["ose_scales"],
                tensors["ose_per_tensor"],
                False,
                args,
            )
        return fn, "ose_w8a8_two_proj_only"

    raise ValueError(f"unknown variant: {variant}")


def reference_for_variant(variant: str, tensors: dict, preset: MoEPreset, active_experts: int, tokens_per_expert: int, args) -> Optional[torch.Tensor]:
    if variant.startswith("component_"):
        return None

    x = tensors["x"]
    if variant in {"duquant_once", "duquant_twice"}:
        x_main = torch_duquant_transform(x, tensors["rot_hidden"], tensors["perm_hidden"], args)
        x_q, x_scale = torch_dynamic_quant_sym(x_main)
        gate = torch_int8_gemm(x_q, tensors["w_gate_full"], x_scale, tensors["gate_scale"], tensors["gate_channels"], tensors["gate_per_tensor"])
        up = torch_int8_gemm(x_q, tensors["w_up_full"], x_scale, tensors["up_scale"], tensors["gate_channels"], tensors["gate_per_tensor"])
    elif variant == "plain_no_ose":
        x_q, x_scale = torch_dynamic_quant_sym(x)
        gate = torch_int8_gemm(x_q, tensors["w_gate_full"], x_scale, tensors["gate_scale"], tensors["gate_channels"], tensors["gate_per_tensor"])
        up = torch_int8_gemm(x_q, tensors["w_up_full"], x_scale, tensors["up_scale"], tensors["gate_channels"], tensors["gate_per_tensor"])
    elif variant == "ours_ose_w8a8":
        x_q, x_scale = torch_dynamic_quant_sym(x)
        gate = torch_int8_gemm(x_q, tensors["w_gate_main"], x_scale, tensors["gate_scale"], tensors["gate_channels"], tensors["gate_per_tensor"])
        up = torch_int8_gemm(x_q, tensors["w_up_main"], x_scale, tensors["up_scale"], tensors["gate_channels"], tensors["gate_per_tensor"])
        gate, up = torch_ose_add(
            x,
            tensors["gate_idx"],
            tensors["up_idx"],
            tensors["w_gate_ose"],
            tensors["w_up_ose"],
            tensors["gate_ose_scale"],
            tensors["up_ose_scale"],
            gate,
            up,
            tensors["ose_channels"],
            tensors["ose_per_tensor"],
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    inter = torch_silu_mul(gate, up)
    if args.include_down_duquant_transform:
        inter = torch_duquant_transform(inter, tensors["rot_inter"], tensors["perm_inter"], args)
    inter_q, inter_scale = torch_dynamic_quant_sym(inter)
    return torch_int8_gemm(
        inter_q,
        tensors["w_down"],
        inter_scale,
        tensors["down_scale"],
        tensors["down_channels"],
        tensors["down_per_tensor"],
    )


def actual_for_variant(variant: str, buffers: dict) -> Optional[torch.Tensor]:
    if variant.startswith("component_"):
        return None
    return buffers["down"]


def run_case(preset_name: str, active_experts: int, tokens_per_expert: int, variant: str, args) -> dict:
    preset = PRESETS[preset_name]
    tensors = build_case_tensors(preset, active_experts, tokens_per_expert, args)
    buffers = make_runtime_buffers(active_experts, tokens_per_expert, preset, args)
    fn, kernel_sequence = build_variant_fn(variant, tensors, buffers, preset, active_experts, tokens_per_expert, args)

    fn()
    torch.cuda.synchronize()

    max_abs = 0.0
    max_rel = 0.0
    if args.validate and not variant.startswith("component_"):
        expected = reference_for_variant(variant, tensors, preset, active_experts, tokens_per_expert, args)
        actual = actual_for_variant(variant, buffers)
        max_abs, max_rel = error_stats(actual, expected)

    median, q20, q80, stdev = measure_ms(fn, args.warmup, args.repeat, args.rounds)
    return {
        "preset": preset_name,
        "variant": variant,
        "active_experts": active_experts,
        "tokens_per_expert": tokens_per_expert,
        "total_routed_tokens": active_experts * tokens_per_expert,
        "hidden_size": preset.hidden,
        "intermediate_size": preset.intermediate,
        "num_experts_model": preset.num_experts,
        "top_k_model": preset.top_k,
        "ose_topk": min(args.ose_topk, preset.hidden),
        "main_weight_group_size": args.main_weight_group_size,
        "ose_weight_group_size": args.ose_weight_group_size,
        "gate_channels_per_scale": tensors["gate_channels"],
        "down_channels_per_scale": tensors["down_channels"],
        "duquant_block_size": args.duquant_block_size,
        "duquant_rotation_stages": args.duquant_rotation_stages,
        "include_duquant_permutation": int(args.include_duquant_permutation),
        "include_down_duquant_transform": int(args.include_down_duquant_transform),
        "latency_ms": f"{median:.6f}",
        "latency_q20_ms": f"{q20:.6f}",
        "latency_q80_ms": f"{q80:.6f}",
        "latency_std_ms": f"{stdev:.6f}",
        "speedup_over_duquant_once": "",
        "speedup_over_plain_no_ose": "",
        "max_abs_error": f"{max_abs:.6e}",
        "max_rel_error": f"{max_rel:.6e}",
        "kernel_sequence": kernel_sequence,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_warps": args.num_warps,
        "dtype": "w8a8_int8_gemm_fp32_accum_fp16_output",
    }


FIELDS = [
    "preset",
    "variant",
    "active_experts",
    "tokens_per_expert",
    "total_routed_tokens",
    "hidden_size",
    "intermediate_size",
    "num_experts_model",
    "top_k_model",
    "ose_topk",
    "main_weight_group_size",
    "ose_weight_group_size",
    "gate_channels_per_scale",
    "down_channels_per_scale",
    "duquant_block_size",
    "duquant_rotation_stages",
    "include_duquant_permutation",
    "include_down_duquant_transform",
    "latency_ms",
    "latency_q20_ms",
    "latency_q80_ms",
    "latency_std_ms",
    "speedup_over_duquant_once",
    "speedup_over_plain_no_ose",
    "max_abs_error",
    "max_rel_error",
    "kernel_sequence",
    "block_m",
    "block_n",
    "block_k",
    "num_warps",
    "dtype",
]


def add_derived_metrics(rows: List[dict]) -> None:
    grouped: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for row in rows:
        key = (row["preset"], row["active_experts"], row["tokens_per_expert"])
        grouped.setdefault(key, {})[row["variant"]] = float(row["latency_ms"])
    for row in rows:
        key = (row["preset"], row["active_experts"], row["tokens_per_expert"])
        latency = float(row["latency_ms"])
        du = grouped.get(key, {}).get("duquant_once")
        plain = grouped.get(key, {}).get("plain_no_ose")
        if du:
            row["speedup_over_duquant_once"] = f"{du / latency:.6f}"
        if plain:
            row["speedup_over_plain_no_ose"] = f"{plain / latency:.6f}"


def write_report(output_dir: str, rows: List[dict]) -> None:
    report_path = os.path.join(output_dir, "report.md")
    full_rows = [row for row in rows if not row["variant"].startswith("component_")]
    row_map = {
        (row["preset"], row["active_experts"], row["tokens_per_expert"], row["variant"]): row
        for row in full_rows
    }
    keys = sorted({(r["preset"], r["active_experts"], r["tokens_per_expert"]) for r in full_rows})
    with open(report_path, "w") as f:
        f.write("# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark\n\n")
        f.write("This benchmark measures a synthetic post-quant W8A8 FFN runtime path. ")
        f.write("Calibration, DuQuant search, weight-side transforms, OSE channel selection, ")
        f.write("weight quantization, and packing are excluded.\n\n")
        f.write("`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. ")
        f.write("`ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. ")
        f.write("The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.\n\n")
        f.write("## Summary\n\n")
        f.write("| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for key in keys:
            du = row_map.get((*key, "duquant_once"))
            plain = row_map.get((*key, "plain_no_ose"))
            ours = row_map.get((*key, "ours_ose_w8a8"))
            if not du or not ours:
                continue
            plain_ms = float(plain["latency_ms"]) if plain else float("nan")
            f.write(
                f"| {key[0]} | {key[1]} | {key[2]} | "
                f"{float(du['latency_ms']):.6f} | {plain_ms:.6f} | "
                f"{float(ours['latency_ms']):.6f} | "
                f"{float(ours['speedup_over_duquant_once']):.4f} |\n"
            )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    presets = parse_csv_strings(args.presets)
    variants = parse_csv_strings(args.variants)
    tokens_values = parse_csv_ints(args.tokens_per_expert)
    active_values = parse_csv_ints(args.active_experts)

    for preset in presets:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; valid: {', '.join(sorted(PRESETS))}")
    if args.duquant_rotation_stages < 1:
        raise ValueError("--duquant_rotation_stages must be >= 1")
    if args.duquant_block_size <= 0:
        raise ValueError("--duquant_block_size must be positive")

    total_cases = 0
    for preset_name in presets:
        max_experts = PRESETS[preset_name].num_experts
        total_cases += len([e for e in active_values if e <= max_experts]) * len(tokens_values) * len(variants)

    rows: List[dict] = []
    done = 0
    for preset_name in presets:
        max_experts = PRESETS[preset_name].num_experts
        for active_experts in active_values:
            if active_experts > max_experts:
                print(
                    f"[post-quant-ffn] skip preset={preset_name} E={active_experts}; model has {max_experts} experts",
                    flush=True,
                )
                continue
            for tokens_per_expert in tokens_values:
                for variant in variants:
                    done += 1
                    print(
                        f"[post-quant-ffn] {done}/{total_cases} preset={preset_name} "
                        f"E={active_experts} T={tokens_per_expert} variant={variant}",
                        flush=True,
                    )
                    rows.append(run_case(preset_name, active_experts, tokens_per_expert, variant, args))

    add_derived_metrics(rows)
    detail_path = os.path.join(args.output_dir, "post_quant_moe_ffn_duquant_vs_ose_detail.csv")
    write_csv(detail_path, rows, FIELDS)
    write_report(args.output_dir, rows)
    print(f"[post-quant-ffn] detail: {detail_path}")
    print(f"[post-quant-ffn] report: {os.path.join(args.output_dir, 'report.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-quant W8A8 MoE FFN DuQuant-vs-OSE benchmark.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--presets", default="olmoe")
    parser.add_argument("--variants", default="duquant_once,plain_no_ose,ours_ose_w8a8,component_gateup_duquant_transform,component_ose_w8a8_branch")
    parser.add_argument("--tokens_per_expert", default="1,2,4,8,16,32")
    parser.add_argument("--active_experts", default="1,8,16,64")
    parser.add_argument("--ose_topk", type=int, default=64)
    parser.add_argument("--main_weight_group_size", type=int, default=2048)
    parser.add_argument("--ose_weight_group_size", type=int, default=1)
    parser.add_argument("--duquant_block_size", type=int, default=128)
    parser.add_argument("--duquant_rotation_stages", type=int, default=2)
    parser.add_argument("--include_duquant_permutation", action="store_true", default=True)
    parser.add_argument("--no_include_duquant_permutation", dest="include_duquant_permutation", action="store_false")
    parser.add_argument("--include_down_duquant_transform", action="store_true", default=True)
    parser.add_argument("--no_include_down_duquant_transform", dest="include_down_duquant_transform", action="store_false")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no_validate", dest="validate", action="store_false")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block_m", type=int, default=16)
    parser.add_argument("--block_n", type=int, default=64)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--num_warps", type=int, default=4)
    parser.add_argument("--elem_block", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
