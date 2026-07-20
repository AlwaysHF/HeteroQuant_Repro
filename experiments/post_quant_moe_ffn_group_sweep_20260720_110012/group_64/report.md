# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.850829 | 0.642048 | 0.753507 | 1.1292 |
| olmoe | 1 | 2 | 0.858204 | 0.667008 | 0.745512 | 1.1512 |
| olmoe | 1 | 4 | 0.848746 | 0.645259 | 0.743524 | 1.1415 |
| olmoe | 1 | 8 | 0.848776 | 0.641593 | 0.743191 | 1.1421 |
| olmoe | 1 | 16 | 0.847428 | 0.640757 | 0.739868 | 1.1454 |
| olmoe | 1 | 32 | 0.851160 | 0.640078 | 0.756622 | 1.1249 |
| olmoe | 8 | 1 | 0.849261 | 0.640146 | 0.743249 | 1.1426 |
| olmoe | 8 | 2 | 0.852602 | 0.645421 | 0.749577 | 1.1374 |
| olmoe | 8 | 4 | 0.847854 | 0.642396 | 0.746834 | 1.1353 |
| olmoe | 8 | 8 | 0.855893 | 0.644538 | 0.744266 | 1.1500 |
| olmoe | 8 | 16 | 0.848085 | 0.640945 | 0.748324 | 1.1333 |
| olmoe | 8 | 32 | 0.849203 | 0.641246 | 0.742233 | 1.1441 |
| olmoe | 16 | 1 | 0.850531 | 0.645924 | 0.753216 | 1.1292 |
| olmoe | 16 | 2 | 0.851760 | 0.642171 | 0.739652 | 1.1516 |
| olmoe | 16 | 4 | 0.857371 | 0.652661 | 0.743607 | 1.1530 |
| olmoe | 16 | 8 | 0.854166 | 0.644329 | 0.764079 | 1.1179 |
| olmoe | 16 | 16 | 0.844964 | 0.636174 | 0.740834 | 1.1406 |
| olmoe | 16 | 32 | 0.847793 | 0.639283 | 0.744336 | 1.1390 |
| olmoe | 64 | 1 | 0.894158 | 0.843417 | 0.868409 | 1.0297 |
| olmoe | 64 | 2 | 0.901487 | 0.850066 | 0.874394 | 1.0310 |
| olmoe | 64 | 4 | 0.916337 | 0.859875 | 0.886438 | 1.0337 |
| olmoe | 64 | 8 | 0.940692 | 0.879657 | 0.909611 | 1.0342 |
| olmoe | 64 | 16 | 0.988656 | 0.912516 | 0.959959 | 1.0299 |
| olmoe | 64 | 32 | 1.132491 | 0.984954 | 1.069870 | 1.0585 |
