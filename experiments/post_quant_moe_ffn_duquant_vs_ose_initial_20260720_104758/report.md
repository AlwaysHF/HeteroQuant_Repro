# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.858815 | 0.645631 | 0.746622 | 1.1503 |
| olmoe | 1 | 2 | 0.881007 | 0.643443 | 0.781430 | 1.1274 |
| olmoe | 1 | 4 | 0.935901 | 0.642701 | 0.767578 | 1.2193 |
| olmoe | 1 | 8 | 0.883012 | 0.647582 | 0.754427 | 1.1704 |
| olmoe | 1 | 16 | 0.887625 | 0.644501 | 0.751336 | 1.1814 |
| olmoe | 1 | 32 | 0.858832 | 0.646376 | 0.752590 | 1.1412 |
| olmoe | 8 | 1 | 0.860816 | 0.642122 | 0.778460 | 1.1058 |
| olmoe | 8 | 2 | 0.868750 | 0.648312 | 0.751526 | 1.1560 |
| olmoe | 8 | 4 | 0.941381 | 0.648987 | 0.749472 | 1.2561 |
| olmoe | 8 | 8 | 0.886301 | 0.646702 | 0.776678 | 1.1411 |
| olmoe | 8 | 16 | 0.880183 | 0.644207 | 0.751880 | 1.1706 |
| olmoe | 8 | 32 | 0.872561 | 0.652307 | 0.767157 | 1.1374 |
| olmoe | 16 | 1 | 0.893972 | 0.654824 | 0.756305 | 1.1820 |
| olmoe | 16 | 2 | 0.880158 | 0.642964 | 0.749685 | 1.1740 |
| olmoe | 16 | 4 | 0.891500 | 0.655909 | 0.747288 | 1.1930 |
| olmoe | 16 | 8 | 0.890108 | 0.653630 | 0.756917 | 1.1760 |
| olmoe | 16 | 16 | 0.881529 | 0.643571 | 0.747414 | 1.1794 |
| olmoe | 16 | 32 | 0.866489 | 0.651752 | 0.755386 | 1.1471 |
| olmoe | 64 | 1 | 0.894306 | 0.843060 | 0.868318 | 1.0299 |
| olmoe | 64 | 2 | 0.901838 | 0.849486 | 0.874268 | 1.0315 |
| olmoe | 64 | 4 | 0.916780 | 0.860065 | 0.886675 | 1.0340 |
| olmoe | 64 | 8 | 0.940938 | 0.879701 | 0.910094 | 1.0339 |
| olmoe | 64 | 16 | 0.990136 | 0.913518 | 0.960002 | 1.0314 |
| olmoe | 64 | 32 | 1.130368 | 0.983636 | 1.069245 | 1.0572 |
