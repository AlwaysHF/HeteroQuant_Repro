# MoE FFN DuQuant-vs-OSE Structure Benchmark

This is a synthetic-weights structure benchmark.  All full variants use the same Triton FP16 GEMM kernel, FP32 accumulation, and FP16 intermediate/output tensors.

## Variants

- `duquant_once`: optimized DuQuant-like baseline.  The gate/up activation transform is computed once and reused by both gate and up GEMMs.
- `duquant_twice`: less optimized path matching a separate transform before gate and up.
- `plain_no_ose`: no gate/up DuQuant transform and no OSE branch.
- `ours_ose`: no gate/up DuQuant transform; add top-k input-channel OSE gate/up branch before `silu(gate) * up`.
- `component_duquant_transform` and `component_ose_branch`: isolated component timings.

## Important Scope

This benchmark proves structural runtime differences under a shared GEMM backend.  It does not prove packed INT4/INT8 deployment latency.

## Compact Summary

| scope | preset | E | T/expert | duquant_once ms | ours_ose ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|
| moe_layer | olmoe | 8 | 1 | 0.610906 | 0.486022 | 1.2570 |
| moe_layer | olmoe | 8 | 2 | 0.615466 | 0.474995 | 1.2957 |
| moe_layer | olmoe | 8 | 4 | 0.613976 | 0.482827 | 1.2716 |
| moe_layer | olmoe | 8 | 8 | 0.604061 | 0.470582 | 1.2836 |
| moe_layer | olmoe | 8 | 16 | 0.622726 | 0.481287 | 1.2939 |
| moe_layer | olmoe | 8 | 32 | 0.601275 | 0.475437 | 1.2647 |
| moe_layer | olmoe | 16 | 1 | 0.630278 | 0.528704 | 1.1921 |
| moe_layer | olmoe | 16 | 2 | 0.644660 | 0.475820 | 1.3548 |
| moe_layer | olmoe | 16 | 4 | 0.608015 | 0.485113 | 1.2533 |
| moe_layer | olmoe | 16 | 8 | 0.634465 | 0.483845 | 1.3113 |
| moe_layer | olmoe | 16 | 16 | 0.627069 | 0.482218 | 1.3004 |
| moe_layer | olmoe | 16 | 32 | 0.601392 | 0.479352 | 1.2546 |
| moe_layer | olmoe | 64 | 1 | 1.495582 | 1.481369 | 1.0096 |
| moe_layer | olmoe | 64 | 2 | 1.500978 | 1.488028 | 1.0087 |
| moe_layer | olmoe | 64 | 4 | 1.516643 | 1.501914 | 1.0098 |
| moe_layer | olmoe | 64 | 8 | 1.543331 | 1.529082 | 1.0093 |
| moe_layer | olmoe | 64 | 16 | 1.599603 | 1.581729 | 1.0113 |
| moe_layer | olmoe | 64 | 32 | 1.728355 | 1.669807 | 1.0351 |
| single_expert | olmoe | 1 | 1 | 0.477246 | 0.352509 | 1.3539 |
| single_expert | olmoe | 1 | 2 | 0.475920 | 0.351152 | 1.3553 |
| single_expert | olmoe | 1 | 4 | 0.486569 | 0.352319 | 1.3810 |
| single_expert | olmoe | 1 | 8 | 0.482297 | 0.351380 | 1.3726 |
| single_expert | olmoe | 1 | 16 | 0.474777 | 0.353930 | 1.3414 |
| single_expert | olmoe | 1 | 32 | 0.475120 | 0.349003 | 1.3614 |
