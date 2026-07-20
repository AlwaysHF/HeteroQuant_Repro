# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.857322 | 0.641993 | 0.745756 | 1.1496 |
| olmoe | 1 | 2 | 0.856446 | 0.642010 | 0.744042 | 1.1511 |
| olmoe | 1 | 4 | 0.853789 | 0.638160 | 0.738952 | 1.1554 |
| olmoe | 1 | 8 | 0.852120 | 0.646626 | 0.744247 | 1.1449 |
| olmoe | 1 | 16 | 0.860310 | 0.648084 | 0.749929 | 1.1472 |
| olmoe | 1 | 32 | 0.846706 | 0.636916 | 0.748594 | 1.1311 |
| olmoe | 8 | 1 | 0.875890 | 0.641370 | 0.741288 | 1.1816 |
| olmoe | 8 | 2 | 0.851746 | 0.640166 | 0.749303 | 1.1367 |
| olmoe | 8 | 4 | 0.857209 | 0.646380 | 0.749219 | 1.1441 |
| olmoe | 8 | 8 | 1.020976 | 0.657783 | 0.751623 | 1.3584 |
| olmoe | 8 | 16 | 1.225693 | 0.934289 | 0.740478 | 1.6553 |
| olmoe | 8 | 32 | 0.854855 | 0.644204 | 0.744028 | 1.1490 |
| olmoe | 16 | 1 | 0.856702 | 0.647517 | 0.750288 | 1.1418 |
| olmoe | 16 | 2 | 0.839613 | 0.635102 | 0.749429 | 1.1203 |
| olmoe | 16 | 4 | 0.858019 | 0.646802 | 0.739660 | 1.1600 |
| olmoe | 16 | 8 | 0.862366 | 0.647972 | 0.752252 | 1.1464 |
| olmoe | 16 | 16 | 0.856959 | 0.641035 | 0.750299 | 1.1422 |
| olmoe | 16 | 32 | 0.867682 | 0.654597 | 0.752260 | 1.1534 |
| olmoe | 64 | 1 | 0.893621 | 0.842582 | 0.867791 | 1.0298 |
| olmoe | 64 | 2 | 0.900888 | 0.848612 | 0.873886 | 1.0309 |
| olmoe | 64 | 4 | 0.916405 | 0.859952 | 0.886524 | 1.0337 |
| olmoe | 64 | 8 | 0.940991 | 0.879858 | 0.910048 | 1.0340 |
| olmoe | 64 | 16 | 0.989214 | 0.912974 | 0.959598 | 1.0309 |
| olmoe | 64 | 32 | 1.133247 | 0.985024 | 1.071270 | 1.0579 |
