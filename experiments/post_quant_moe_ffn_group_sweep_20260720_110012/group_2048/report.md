# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.858710 | 0.643691 | 0.755901 | 1.1360 |
| olmoe | 1 | 2 | 0.857445 | 0.648326 | 0.750916 | 1.1419 |
| olmoe | 1 | 4 | 0.855647 | 0.645607 | 0.748448 | 1.1432 |
| olmoe | 1 | 8 | 0.859351 | 0.644973 | 0.751027 | 1.1442 |
| olmoe | 1 | 16 | 0.857066 | 0.646250 | 0.755399 | 1.1346 |
| olmoe | 1 | 32 | 0.859724 | 0.649376 | 0.754486 | 1.1395 |
| olmoe | 8 | 1 | 0.860399 | 0.646148 | 0.746161 | 1.1531 |
| olmoe | 8 | 2 | 0.862615 | 0.649738 | 0.752838 | 1.1458 |
| olmoe | 8 | 4 | 0.858666 | 0.645344 | 0.753098 | 1.1402 |
| olmoe | 8 | 8 | 0.861539 | 0.646256 | 0.755573 | 1.1402 |
| olmoe | 8 | 16 | 0.855536 | 0.642690 | 0.746798 | 1.1456 |
| olmoe | 8 | 32 | 0.863187 | 0.647200 | 0.745808 | 1.1574 |
| olmoe | 16 | 1 | 0.857553 | 0.644048 | 0.752845 | 1.1391 |
| olmoe | 16 | 2 | 0.851554 | 0.642929 | 0.751633 | 1.1329 |
| olmoe | 16 | 4 | 0.863546 | 0.649634 | 0.756304 | 1.1418 |
| olmoe | 16 | 8 | 0.864397 | 0.652986 | 0.755587 | 1.1440 |
| olmoe | 16 | 16 | 0.855964 | 0.644446 | 0.750390 | 1.1407 |
| olmoe | 16 | 32 | 0.866612 | 0.651751 | 0.759427 | 1.1411 |
| olmoe | 64 | 1 | 0.893780 | 0.842945 | 0.868154 | 1.0295 |
| olmoe | 64 | 2 | 0.901139 | 0.848688 | 0.874012 | 1.0310 |
| olmoe | 64 | 4 | 0.916714 | 0.859974 | 0.886577 | 1.0340 |
| olmoe | 64 | 8 | 0.940804 | 0.879898 | 0.909699 | 1.0342 |
| olmoe | 64 | 16 | 0.989592 | 0.913161 | 0.959922 | 1.0309 |
| olmoe | 64 | 32 | 1.133120 | 0.985324 | 1.071208 | 1.0578 |
