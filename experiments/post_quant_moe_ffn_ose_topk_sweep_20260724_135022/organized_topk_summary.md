# Organized OSE Top-k Sweep Results

Root: `/home/lwk/HeteroQuant_Repro/experiments/post_quant_moe_ffn_ose_topk_sweep_20260724_135022`

Scope: `routed_moe_layer`. Speedup is `duquant_once_latency / ours_ose_w8a8_latency`; larger is better for ours.

`k` is OSE outlier input channels (`OSE_TOPK`), not the MoE router top-k. Router top-k for OlmoE remains 8.

## Main Summary

| OSE k | all median | all fast/cases | E1 med | E8 med | E16 med | E64 med | DuQuant/OSE MAC ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.1146 | 22/24 | 1.1242 | 1.1160 | 1.1155 | 1.0428 | 16.0x |
| 32 | 1.1117 | 19/24 | 1.1172 | 1.1212 | 1.1150 | 0.9569 | 8.0x |
| 64 | 1.1108 | 24/24 | 1.1148 | 1.1170 | 1.1095 | 1.0327 | 4.0x |
| 128 | 1.1055 | 19/24 | 1.1055 | 1.1234 | 1.1112 | 0.9928 | 2.0x |
| 256 | 1.1131 | 19/24 | 1.1196 | 1.1174 | 1.1118 | 0.9509 | 1.0x |

## Stable-shape Summary

| OSE k | E1/E8/E16 median | fast/cases | E8/E16 median | fast/cases | E64 median |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.1191 | 17/18 | 1.1160 | 11/12 | 1.0428 |
| 32 | 1.1200 | 17/18 | 1.1212 | 11/12 | 0.9569 |
| 64 | 1.1146 | 18/18 | 1.1142 | 12/12 | 1.0327 |
| 128 | 1.1092 | 17/18 | 1.1175 | 11/12 | 0.9928 |
| 256 | 1.1177 | 18/18 | 1.1152 | 12/12 | 0.9509 |

## Slow Cases

| OSE k | E | T/expert | total tokens | DuQuant ms | Ours ms | speedup | ours std ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 4 | 8 | 1.050186 | 1.303924 | 0.8054 | 0.281829 |
| 16 | 64 | 16 | 128 | 1.289164 | 1.294509 | 0.9959 | 0.074857 |
| 32 | 16 | 2 | 4 | 1.052692 | 1.294591 | 0.8131 | 0.228110 |
| 32 | 64 | 1 | 8 | 1.075538 | 1.153211 | 0.9326 | 0.160702 |
| 32 | 64 | 2 | 16 | 1.091818 | 1.225316 | 0.8911 | 0.257603 |
| 32 | 64 | 8 | 64 | 1.153481 | 1.250208 | 0.9226 | 0.099402 |
| 32 | 64 | 32 | 256 | 1.443144 | 1.470807 | 0.9812 | 0.084299 |
| 128 | 16 | 1 | 2 | 1.095410 | 1.130897 | 0.9686 | 0.119243 |
| 128 | 64 | 1 | 8 | 1.109123 | 1.162056 | 0.9544 | 0.045139 |
| 128 | 64 | 2 | 16 | 1.160275 | 1.169907 | 0.9918 | 0.069197 |
| 128 | 64 | 4 | 32 | 1.200355 | 1.249296 | 0.9608 | 0.125797 |
| 128 | 64 | 16 | 128 | 1.295419 | 1.303430 | 0.9939 | 0.059615 |
| 256 | 64 | 1 | 8 | 1.068678 | 1.197760 | 0.8922 | 0.015981 |
| 256 | 64 | 4 | 32 | 1.205021 | 1.256407 | 0.9591 | 0.056538 |
| 256 | 64 | 8 | 64 | 1.274804 | 1.352315 | 0.9427 | 0.135826 |
| 256 | 64 | 16 | 128 | 1.431713 | 1.459148 | 0.9812 | 0.049306 |
| 256 | 64 | 32 | 256 | 1.572815 | 1.696353 | 0.9272 | 0.089940 |

## Reading

- `k=64` is the most stable setting in this run: all 24 cases are faster than DuQuant, and median speedup is about 1.11x.
- `k=16` has the lowest OSE MACs, but includes two slow/high-noise cases; its all-case median is similar to k=64.
- `k=128/256` keeps E1/E8/E16 speedup, but E64 becomes near break-even or slower.
- `ACTIVE_EXPERTS=1` is a synthetic single-active-expert case for OlmoE; normal single-token routing is closer to E=8,T=1.
