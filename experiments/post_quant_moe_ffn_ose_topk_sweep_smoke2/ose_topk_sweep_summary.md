# OSE Top-k Sweep Summary

Root: `/home/lwk/HeteroQuant_Repro/experiments/post_quant_moe_ffn_ose_topk_sweep_smoke2`

Speedup is `duquant_once_latency / ours_ose_w8a8_latency`; larger is better for ours.

| ose_topk | speedup_median | speedup_mean | E1_speedup_median | E8_speedup_median | E16_speedup_median | E64_speedup_median | duquant_over_ose_macs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.276820 | 1.276820 | 1.671401 | 0.882240 |  |  | 16.000000 |
| 64 | 1.061719 | 1.061719 | 1.290090 | 0.833349 |  |  | 4.000000 |
