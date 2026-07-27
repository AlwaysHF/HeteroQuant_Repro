# Organized Routed MoE Layer Results

Root: `/home/lwk/HeteroQuant_Repro/experiments/post_quant_moe_ffn_duquant_vs_ose_20260724_133314`

Scope: full routed MoE layer after routing result is known: dispatch -> expert FFN -> weighted scatter/add. Router linear/top-k is not included because it is common to both methods.

## Overall

- Cases: 18
- Ours vs DuQuant speedup median: 1.1017x
- Ours vs DuQuant speedup mean: 1.0501x
- Range: 0.8494x to 1.1243x
- Median excluding high-variance rows: 1.1057x

## By Active Experts

| E | cases | fast | slow | min | median | mean | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 6 | 0 | 1.0993 | 1.1061 | 1.1067 | 1.1154 |
| 16 | 6 | 5 | 1 | 0.8763 | 1.1051 | 1.0709 | 1.1243 |
| 64 | 6 | 3 | 3 | 0.8494 | 0.9941 | 0.9728 | 1.0536 |

## Case Table

| E | T/expert | total tokens | DuQuant ms | Ours ms | speedup | note |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 1 | 1 | 1.050213 | 0.941594 | 1.1154 | stable_speedup |
| 8 | 2 | 2 | 1.042830 | 0.940187 | 1.1092 | stable_speedup |
| 8 | 4 | 4 | 1.048872 | 0.947527 | 1.1070 | stable_speedup |
| 8 | 8 | 8 | 1.028798 | 0.935847 | 1.0993 | stable_speedup |
| 8 | 16 | 16 | 1.041623 | 0.943481 | 1.1040 | stable_speedup |
| 8 | 32 | 32 | 1.038002 | 0.939140 | 1.1053 | stable_speedup |
| 16 | 1 | 2 | 1.045443 | 0.929837 | 1.1243 | stable_speedup |
| 16 | 2 | 4 | 1.040666 | 0.942525 | 1.1041 | stable_speedup |
| 16 | 4 | 8 | 1.066612 | 0.950368 | 1.1223 | stable_speedup |
| 16 | 8 | 16 | 1.031836 | 0.944636 | 1.0923 | stable_speedup |
| 16 | 16 | 32 | 1.054433 | 0.953291 | 1.1061 | stable_speedup |
| 16 | 32 | 64 | 1.040057 | 1.186876 | 0.8763 | ours_slower; high_variance |
| 64 | 1 | 8 | 1.285579 | 1.243648 | 1.0337 | near_tie; high_variance |
| 64 | 2 | 16 | 1.285461 | 1.266345 | 1.0151 | near_tie; high_variance |
| 64 | 4 | 32 | 1.257747 | 1.379021 | 0.9121 | ours_slower |
| 64 | 8 | 64 | 1.305131 | 1.341225 | 0.9731 | near_tie; high_variance |
| 64 | 16 | 128 | 1.194541 | 1.406293 | 0.8494 | ours_slower; high_variance |
| 64 | 32 | 256 | 1.669770 | 1.584817 | 1.0536 | stable_speedup; high_variance |

## Reading

- E=8 is stable: every case is about 1.10x-1.12x faster.
- E=16 is mostly faster, but T=32 has high variance and should be rerun before citing.
- E=64 is near break-even or slower in several cases; this all-expert-active stress shape should not be used as the primary speedup claim.
