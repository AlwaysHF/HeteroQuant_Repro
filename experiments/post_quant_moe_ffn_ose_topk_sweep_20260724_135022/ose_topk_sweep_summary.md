# OSE Top-k Sweep Summary

Root: `/home/lwk/HeteroQuant_Repro/experiments/post_quant_moe_ffn_ose_topk_sweep_20260724_135022`

Speedup is `duquant_once_latency / ours_ose_w8a8_latency`; larger is better for ours.

| ose_topk | speedup_median | speedup_mean | E1_speedup_median | E8_speedup_median | E16_speedup_median | E64_speedup_median | duquant_over_ose_macs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.114602 | 1.087183 | 1.124199 | 1.115989 | 1.115515 | 1.042844 | 16.000000 |
| 32 | 1.111686 | 1.085519 | 1.117157 | 1.121225 | 1.114969 | 0.956919 | 8.000000 |
| 64 | 1.110762 | 1.113626 | 1.114841 | 1.116999 | 1.109495 | 1.032717 | 4.000000 |
| 128 | 1.105533 | 1.074864 | 1.105533 | 1.123448 | 1.111206 | 0.992811 | 2.000000 |
| 256 | 1.113119 | 1.075600 | 1.119622 | 1.117433 | 1.111757 | 0.950892 | 1.000000 |
