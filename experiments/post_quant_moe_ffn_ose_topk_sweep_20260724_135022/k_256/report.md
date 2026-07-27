# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 MoE runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| scope | preset | E | tokens/expert | total tokens | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_moe_layer | olmoe | 1 | 1 | 1 | 1.056103 | 0.839380 | 0.944824 | 1.1178 |
| routed_moe_layer | olmoe | 1 | 2 | 1 | 1.057565 | 0.842664 | 0.939263 | 1.1260 |
| routed_moe_layer | olmoe | 1 | 4 | 1 | 1.055824 | 0.843220 | 0.948750 | 1.1129 |
| routed_moe_layer | olmoe | 1 | 8 | 1 | 1.062424 | 0.840832 | 0.950538 | 1.1177 |
| routed_moe_layer | olmoe | 1 | 16 | 2 | 1.059133 | 0.835097 | 0.942317 | 1.1240 |
| routed_moe_layer | olmoe | 1 | 32 | 4 | 1.057061 | 0.854181 | 0.942570 | 1.1215 |
| routed_moe_layer | olmoe | 8 | 1 | 1 | 1.049850 | 0.841079 | 0.948487 | 1.1069 |
| routed_moe_layer | olmoe | 8 | 2 | 2 | 1.052451 | 0.836647 | 0.948258 | 1.1099 |
| routed_moe_layer | olmoe | 8 | 4 | 4 | 1.061767 | 0.844550 | 0.941272 | 1.1280 |
| routed_moe_layer | olmoe | 8 | 8 | 8 | 1.056343 | 0.854254 | 0.943515 | 1.1196 |
| routed_moe_layer | olmoe | 8 | 16 | 16 | 1.060062 | 0.841912 | 0.948359 | 1.1178 |
| routed_moe_layer | olmoe | 8 | 32 | 32 | 1.062090 | 0.836139 | 0.950773 | 1.1171 |
| routed_moe_layer | olmoe | 16 | 1 | 2 | 1.057057 | 0.843694 | 0.952812 | 1.1094 |
| routed_moe_layer | olmoe | 16 | 2 | 4 | 1.051121 | 0.836796 | 0.936345 | 1.1226 |
| routed_moe_layer | olmoe | 16 | 4 | 8 | 1.053317 | 0.841201 | 0.948820 | 1.1101 |
| routed_moe_layer | olmoe | 16 | 8 | 16 | 1.049706 | 0.835648 | 0.942809 | 1.1134 |
| routed_moe_layer | olmoe | 16 | 16 | 32 | 1.050422 | 0.839370 | 0.936274 | 1.1219 |
| routed_moe_layer | olmoe | 16 | 32 | 64 | 1.067865 | 0.840434 | 0.965663 | 1.1058 |
| routed_moe_layer | olmoe | 64 | 1 | 8 | 1.068678 | 1.114271 | 1.197760 | 0.8922 |
| routed_moe_layer | olmoe | 64 | 2 | 16 | 1.214954 | 1.137710 | 1.203142 | 1.0098 |
| routed_moe_layer | olmoe | 64 | 4 | 32 | 1.205021 | 1.130466 | 1.256407 | 0.9591 |
| routed_moe_layer | olmoe | 64 | 8 | 64 | 1.274804 | 1.173875 | 1.352315 | 0.9427 |
| routed_moe_layer | olmoe | 64 | 16 | 128 | 1.431713 | 1.256433 | 1.459148 | 0.9812 |
| routed_moe_layer | olmoe | 64 | 32 | 256 | 1.572815 | 1.408044 | 1.696353 | 0.9272 |
