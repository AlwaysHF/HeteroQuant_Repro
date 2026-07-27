# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 MoE runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| scope | preset | E | tokens/expert | total tokens | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_moe_layer | olmoe | 1 | 1 | 1 | 1.057036 | 0.839135 | 0.952772 | 1.1094 |
| routed_moe_layer | olmoe | 1 | 2 | 1 | 1.062983 | 0.851939 | 0.954831 | 1.1133 |
| routed_moe_layer | olmoe | 1 | 4 | 1 | 1.160641 | 1.060629 | 0.951464 | 1.2198 |
| routed_moe_layer | olmoe | 1 | 8 | 1 | 1.052964 | 0.835952 | 0.939268 | 1.1210 |
| routed_moe_layer | olmoe | 1 | 16 | 2 | 1.046416 | 0.835714 | 0.931517 | 1.1233 |
| routed_moe_layer | olmoe | 1 | 32 | 4 | 1.038474 | 0.828173 | 0.942281 | 1.1021 |
| routed_moe_layer | olmoe | 8 | 1 | 1 | 1.043061 | 0.837230 | 0.942919 | 1.1062 |
| routed_moe_layer | olmoe | 8 | 2 | 2 | 1.065742 | 0.869260 | 0.942623 | 1.1306 |
| routed_moe_layer | olmoe | 8 | 4 | 4 | 1.049215 | 0.836837 | 0.933914 | 1.1235 |
| routed_moe_layer | olmoe | 8 | 8 | 8 | 1.038858 | 0.829567 | 0.935821 | 1.1101 |
| routed_moe_layer | olmoe | 8 | 16 | 16 | 1.051337 | 0.829125 | 0.939540 | 1.1190 |
| routed_moe_layer | olmoe | 8 | 32 | 32 | 1.061770 | 0.835405 | 0.939535 | 1.1301 |
| routed_moe_layer | olmoe | 16 | 1 | 2 | 1.132778 | 0.821443 | 0.954956 | 1.1862 |
| routed_moe_layer | olmoe | 16 | 2 | 4 | 1.052692 | 0.834964 | 1.294591 | 0.8131 |
| routed_moe_layer | olmoe | 16 | 4 | 8 | 1.045363 | 0.829711 | 0.948101 | 1.1026 |
| routed_moe_layer | olmoe | 16 | 8 | 16 | 1.051370 | 0.828231 | 0.935455 | 1.1239 |
| routed_moe_layer | olmoe | 16 | 16 | 32 | 1.053249 | 0.840849 | 0.932713 | 1.1292 |
| routed_moe_layer | olmoe | 16 | 32 | 64 | 1.045831 | 0.836117 | 0.945576 | 1.1060 |
| routed_moe_layer | olmoe | 64 | 1 | 8 | 1.075538 | 1.031389 | 1.153211 | 0.9326 |
| routed_moe_layer | olmoe | 64 | 2 | 16 | 1.091818 | 1.043922 | 1.225316 | 0.8911 |
| routed_moe_layer | olmoe | 64 | 4 | 32 | 1.115500 | 1.044288 | 1.062533 | 1.0498 |
| routed_moe_layer | olmoe | 64 | 8 | 64 | 1.153481 | 1.087541 | 1.250208 | 0.9226 |
| routed_moe_layer | olmoe | 64 | 16 | 128 | 1.529455 | 1.164662 | 1.171563 | 1.3055 |
| routed_moe_layer | olmoe | 64 | 32 | 256 | 1.443144 | 1.266955 | 1.470807 | 0.9812 |
