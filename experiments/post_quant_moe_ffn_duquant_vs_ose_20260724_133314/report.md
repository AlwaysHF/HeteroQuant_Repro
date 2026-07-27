# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 MoE runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| scope | preset | E | tokens/expert | total tokens | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_moe_layer | olmoe | 8 | 1 | 1 | 1.050213 | 0.841198 | 0.941594 | 1.1154 |
| routed_moe_layer | olmoe | 8 | 2 | 2 | 1.042830 | 0.833678 | 0.940187 | 1.1092 |
| routed_moe_layer | olmoe | 8 | 4 | 4 | 1.048872 | 0.834444 | 0.947527 | 1.1070 |
| routed_moe_layer | olmoe | 8 | 8 | 8 | 1.028798 | 0.828908 | 0.935847 | 1.0993 |
| routed_moe_layer | olmoe | 8 | 16 | 16 | 1.041623 | 0.829275 | 0.943481 | 1.1040 |
| routed_moe_layer | olmoe | 8 | 32 | 32 | 1.038002 | 0.808102 | 0.939140 | 1.1053 |
| routed_moe_layer | olmoe | 16 | 1 | 2 | 1.045443 | 0.834831 | 0.929837 | 1.1243 |
| routed_moe_layer | olmoe | 16 | 2 | 4 | 1.040666 | 0.850192 | 0.942525 | 1.1041 |
| routed_moe_layer | olmoe | 16 | 4 | 8 | 1.066612 | 0.820217 | 0.950368 | 1.1223 |
| routed_moe_layer | olmoe | 16 | 8 | 16 | 1.031836 | 0.823459 | 0.944636 | 1.0923 |
| routed_moe_layer | olmoe | 16 | 16 | 32 | 1.054433 | 0.824979 | 0.953291 | 1.1061 |
| routed_moe_layer | olmoe | 16 | 32 | 64 | 1.040057 | 0.834995 | 1.186876 | 0.8763 |
| routed_moe_layer | olmoe | 64 | 1 | 8 | 1.285579 | 1.225669 | 1.243648 | 1.0337 |
| routed_moe_layer | olmoe | 64 | 2 | 16 | 1.285461 | 1.169662 | 1.266345 | 1.0151 |
| routed_moe_layer | olmoe | 64 | 4 | 32 | 1.257747 | 1.207657 | 1.379021 | 0.9121 |
| routed_moe_layer | olmoe | 64 | 8 | 64 | 1.305131 | 1.279754 | 1.341225 | 0.9731 |
| routed_moe_layer | olmoe | 64 | 16 | 128 | 1.194541 | 1.116459 | 1.406293 | 0.8494 |
| routed_moe_layer | olmoe | 64 | 32 | 256 | 1.669770 | 1.470013 | 1.584817 | 1.0536 |
