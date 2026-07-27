# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 MoE runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| scope | preset | E | tokens/expert | total tokens | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_moe_layer | olmoe | 8 | 1 | 1 | 1.063092 | 0.848483 | 0.952933 | 1.1156 |
| routed_moe_layer | olmoe | 8 | 2 | 2 | 1.056944 | 0.836933 | 0.945268 | 1.1181 |
| routed_moe_layer | olmoe | 8 | 4 | 4 | 1.056333 | 0.841456 | 0.957468 | 1.1033 |
| routed_moe_layer | olmoe | 8 | 8 | 8 | 1.051827 | 0.834936 | 0.944651 | 1.1135 |
| routed_moe_layer | olmoe | 8 | 16 | 16 | 1.058474 | 0.838877 | 0.953118 | 1.1105 |
| routed_moe_layer | olmoe | 8 | 32 | 32 | 1.042901 | 0.842134 | 0.954239 | 1.0929 |
| routed_moe_layer | olmoe | 16 | 1 | 2 | 1.572667 | 0.845202 | 0.944986 | 1.6642 |
| routed_moe_layer | olmoe | 16 | 2 | 4 | 1.060483 | 0.847254 | 0.978139 | 1.0842 |
| routed_moe_layer | olmoe | 16 | 4 | 8 | 1.045140 | 0.827246 | 0.952094 | 1.0977 |
| routed_moe_layer | olmoe | 16 | 8 | 16 | 1.069899 | 0.845236 | 0.947787 | 1.1288 |
| routed_moe_layer | olmoe | 16 | 16 | 32 | 1.043825 | 0.838982 | 0.941121 | 1.1091 |
| routed_moe_layer | olmoe | 16 | 32 | 64 | 1.051447 | 0.836269 | 0.940459 | 1.1180 |
| routed_moe_layer | olmoe | 64 | 1 | 8 | 1.085734 | 1.029092 | 1.056939 | 1.0272 |
| routed_moe_layer | olmoe | 64 | 2 | 16 | 1.082804 | 1.041172 | 1.047904 | 1.0333 |
| routed_moe_layer | olmoe | 64 | 4 | 32 | 1.117678 | 1.053738 | 1.086194 | 1.0290 |
| routed_moe_layer | olmoe | 64 | 8 | 64 | 1.157623 | 1.089930 | 1.130455 | 1.0240 |
| routed_moe_layer | olmoe | 64 | 16 | 128 | 1.452582 | 1.146739 | 1.196216 | 1.2143 |
| routed_moe_layer | olmoe | 64 | 32 | 256 | 1.455393 | 1.264717 | 1.364638 | 1.0665 |
