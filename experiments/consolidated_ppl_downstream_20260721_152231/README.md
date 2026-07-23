# Consolidated PPL/Downstream Results

Generated at 2026-07-21 15:22:33. Same config duplicates are collapsed to the latest timestamp within each experiment context. Full raw/latest CSVs are in this directory.

## Final Downstream And Baselines

| model | label | metric | aw_bits | weight_group | wikitext2 | c4 | acc_avg | avg_no_obqa | arc_challenge | arc_easy | boolq | openbookqa | piqa | winogrande | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olmoe | olmoe_final_routed_tail_group2048 | routed_tail |  | 2048 | 7.2137 | 11.4510 | 0.6457 | 0.6844 | 0.4829 | 0.7681 | 0.6859 | 0.4520 | 0.7873 | 0.6977 | 2026-07-18 03:41:10 |
| olmoe | aw11 | routed_tail | 11 | 2048 | 7.3937 | 11.5702 | 0.6381 | 0.6753 | 0.4770 | 0.7597 | 0.6661 | 0.4520 | 0.7911 | 0.6827 | 2026-07-18 15:22:46 |
| olmoe | output | routed_tail | 12 | 2048 | 7.1766 | 11.4514 | 0.6480 | 0.6864 | 0.4915 | 0.7698 | 0.6911 | 0.4560 | 0.7943 | 0.6851 | 2026-07-18 14:13:49 |
| olmoe | reproduce_olmoe_691_route_range | route_range | 12 | 1 | 6.9065 | 11.1342 | 0.6432 | 0.6803 | 0.4821 | 0.7555 | 0.6936 | 0.4580 | 0.7867 | 0.6835 | 2026-07-21 02:15:28 |
| olmoe | reproduce_olmoe_691_route_range | route_range | 12 | 2048 | 7.2361 | 11.5041 | 0.6453 | 0.6840 | 0.4915 | 0.7677 | 0.6853 | 0.4520 | 0.7894 | 0.6859 | 2026-07-13 23:39:01 |
| olmoe | aw13 | routed_tail | 13 | 2048 | 7.1766 | 11.4514 | 0.6480 | 0.6864 | 0.4915 | 0.7698 | 0.6911 | 0.4560 | 0.7943 | 0.6851 | 2026-07-18 15:22:46 |
| olmoe | fp16_olmoe |  | 32 | None | 6.6490 | 10.8632 | 0.6490 | 0.6896 | 0.4966 | 0.7706 | 0.6954 | 0.4460 | 0.7982 | 0.6875 | 2026-07-14 00:13:23 |
| qwen | qwen_final_routed_tail_group2048 | routed_tail |  | 2048 | 8.2490 | 10.7045 | 0.6392 | 0.6763 | 0.4283 | 0.7197 | 0.7780 | 0.4540 | 0.7916 | 0.6638 | 2026-07-18 03:41:10 |
| qwen | aw11 | routed_tail | 11 | 2048 | 8.3033 | 10.7660 | 0.6375 | 0.6738 | 0.4189 | 0.7184 | 0.7667 | 0.4560 | 0.7949 | 0.6701 | 2026-07-19 01:01:09 |
| qwen | aw13 | routed_tail | 13 | 2048 | 8.2422 | 10.6614 | 0.6407 | 0.6804 | 0.4275 | 0.7222 | 0.7768 | 0.4420 | 0.8009 | 0.6748 | 2026-07-19 01:01:09 |
| qwen2_moe | output | routed_tail | 12 | 2048 | 8.2422 | 10.6614 | 0.6407 | 0.6804 | 0.4275 | 0.7222 | 0.7768 | 0.4420 | 0.8009 | 0.6748 | 2026-07-19 01:01:07 |
| qwen2_moe | qwen_group2048_route_range | route_range | 12 | 2048 | 8.2650 | 10.7136 | 0.6370 | 0.6764 | 0.4300 | 0.7184 | 0.7709 | 0.4400 | 0.7965 | 0.6661 | 2026-07-14 04:25:47 |
| qwen2_moe | fp16_qwen |  | 32 | None | 7.2174 | 9.3660 | 0.6511 | 0.6925 | 0.4428 | 0.7319 | 0.7936 | 0.4440 | 0.8052 | 0.6890 | 2026-07-14 00:50:09 |

## Allocation Metrics Group2048

| label | metric | aw_bits | weight_group | ose_topk | ose_quant | ose_score | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kurtosis | kurtosis | 12.0 | 2048 |  |  |  | 7.2681 | 11.5500 | 2026-07-16 13:38:34 |
| range | routed_range | 12.0 | 2048 |  |  |  | 7.3035 | 11.5497 | 2026-07-16 13:38:34 |
| residual_qerr | residual_qerr | 12.0 | 2048 |  |  |  | 7.4368 | 11.6590 | 2026-07-16 13:38:34 |
| residual_qerr_gain | residual_qerr_gain | 12.0 | 2048 |  |  |  | 7.4386 | 11.6572 | 2026-07-16 13:38:34 |
| route | route | 12.0 | 2048 |  |  |  | 7.2151 | 11.4456 | 2026-07-16 13:38:34 |
| route_range | routed_range | 12.0 | 2048 |  |  |  | 7.2277 | 11.5132 | 2026-07-16 13:38:34 |
| routed_kurtosis | routed_kurtosis | 12.0 | 2048 |  |  |  | 7.2276 | 11.5218 | 2026-07-16 13:38:34 |
| routed_tail | routed_tail | 12.0 | 2048 |  |  |  | 7.2137 | 11.4510 | 2026-07-16 13:38:34 |
| tail | tail | 12.0 | 2048 |  |  |  | 7.2526 | 11.4808 | 2026-07-16 13:38:34 |

## Tier Routed Tail Group2048

| label | metric | aw_bits | weight_group | ose_topk | ose_quant | ose_score | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2tier_w4a4_w5a8 | routed_tail | 11.984375 | 2048 |  |  |  | 7.2270 | 11.4478 | 2026-07-16 16:52:36 |
| 2tier_w4a4_w6a8 | routed_tail | 12.03125 | 2048 |  |  |  | 7.2555 | 11.4989 | 2026-07-16 16:52:36 |
| 2tier_w4a4_w8a8 | routed_tail | 12.0 | 2048 |  |  |  | 7.3330 | 11.5642 | 2026-07-16 16:52:36 |
| 3tier_w4a4_w5a8_w6a8 | routed_tail | 12.0 | 2048 |  |  |  | 7.2137 | 11.4510 | 2026-07-16 16:52:36 |
| 4tier_add_w4a8 | routed_tail | 12.0 | 2048 |  |  |  | 7.2194 | 11.4477 | 2026-07-16 16:52:36 |
| 5tier_add_w8a8 | routed_tail | 11.953125 | 2048 |  |  |  | 7.2418 | 11.4804 | 2026-07-16 16:52:36 |

## OSE TopK Group2048

| label | metric | aw_bits | weight_group | ose_topk | ose_quant | ose_score | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topk_0 | routed_tail | 12.0 | 2048 | 0 | w8a8 | smooth_scale | 7.4730 | 11.7025 | 2026-07-16 19:30:59 |
| topk_128 | routed_tail | 12.0 | 2048 | 128 | w8a8 | smooth_scale | 7.2542 | 11.4810 | 2026-07-16 19:30:59 |
| topk_16 | routed_tail | 12.0 | 2048 | 16 | w8a8 | smooth_scale | 7.3290 | 11.5343 | 2026-07-16 19:30:59 |
| topk_256 | routed_tail | 12.0 | 2048 | 256 | w8a8 | smooth_scale | 7.2539 | 11.4520 | 2026-07-16 19:30:59 |
| topk_32 | routed_tail | 12.0 | 2048 | 32 | w8a8 | smooth_scale | 7.3076 | 11.5150 | 2026-07-16 19:30:59 |
| topk_64 | routed_tail | 12.0 | 2048 | 64 | w8a8 | smooth_scale | 7.2137 | 11.4510 | 2026-07-16 19:30:59 |

## Core Components Group2048

| label | metric | aw_bits | weight_group | ose_topk | ose_quant | ose_score | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| allocation_only | routed_tail | 12.0 | 2048 | 0 | same | smooth_scale | 7.4730 | 11.7025 | 2026-07-16 22:53:24 |
| full_method | routed_tail | 12.0 | 2048 | 64 | w8a8 | smooth_scale | 7.2137 | 11.4510 | 2026-07-16 22:53:24 |
| no_gateup_rotation | routed_tail | 12.0 | 2048 | 0 | same | smooth_scale | 8.3255 | 12.4453 | 2026-07-16 22:53:24 |
| ose_only | routed_tail | 12.0 | 2048 | 64 | w8a8 | smooth_scale | 7.3749 | 11.5576 | 2026-07-16 22:53:24 |
| ose_random_same_budget | random | 12.0 | 2048 | 64 | w8a8 | smooth_scale | 7.3329 | 11.5538 | 2026-07-16 22:53:24 |

## OSE Precision Group2048

| label | metric | aw_bits | weight_group | ose_topk | ose_quant | ose_score | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ose_fp16 | routed_tail | 12.0 | 2048 | 64 | fp16 | smooth_scale | 7.2250 | 11.4535 | 2026-07-17 00:37:31 |
| ose_same | routed_tail | 12.0 | 2048 | 64 | same | smooth_scale | 7.2784 | 11.5281 | 2026-07-17 00:37:31 |
| ose_w16a16 | routed_tail | 12.0 | 2048 | 64 | w16a16 | smooth_scale | 7.2250 | 11.4535 | 2026-07-17 00:37:31 |
| ose_w4a4 | routed_tail | 12.0 | 2048 | 64 | w4a4 | smooth_scale | 7.2626 | 11.5216 | 2026-07-17 00:37:31 |
| ose_w8a8 | routed_tail | 12.0 | 2048 | 64 | w8a8 | smooth_scale | 7.2137 | 11.4510 | 2026-07-17 00:37:31 |

## OSE Score Group2048

| label | metric | aw_bits | weight_group | ose_topk | ose_quant | ose_score | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_smooth_scale | routed_tail | 12.0 | 2048 | 64 | w8a8 | smooth_scale | 7.2137 | 11.4510 | 2026-07-17 01:47:12 |
| score_weight_error | routed_tail | 12.0 | 2048 | 64 | w8a8 | weight_error | 7.2386 | 11.4658 | 2026-07-17 01:47:12 |
| score_weight_max | routed_tail | 12.0 | 2048 | 64 | w8a8 | weight_max | 7.2492 | 11.4443 | 2026-07-17 01:47:12 |

## Group Size Scan Routed Tail

| label | metric | weight_group | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- |
| routed_tail | routed_tail | 1 | 6.9257 | 11.1835 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 1024 | 7.2003 | 11.4033 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 128 | 7.0053 | 11.2292 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 16 | 6.9161 | 11.1875 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 2048 | 7.2137 | 11.4510 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 256 | 7.0332 | 11.2800 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 32 | 6.9553 | 11.1814 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 512 | 7.1171 | 11.3164 | 2026-07-17 14:07:46 |
| routed_tail | routed_tail | 64 | 6.9978 | 11.2206 | 2026-07-17 14:07:46 |

## AW Budget Sweep OLMoE

| label | metric | aw_bits | weight_group | ose_topk | ose_quant | ose_score | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aw10 | routed_tail | 10 | 2048 | 64 | w8a8 | smooth_scale | 7.5303 | 11.6890 | 2026-07-18 15:22:46 |
| aw11 | routed_tail | 11 | 2048 | 64 | w8a8 | smooth_scale | 7.3937 | 11.5702 | 2026-07-18 15:22:46 |
| aw12 | routed_tail | 12 | 2048 | 64 | w8a8 | smooth_scale | 7.2137 | 11.4510 | 2026-07-18 15:22:46 |
| aw13 | routed_tail | 13 | 2048 | 64 | w8a8 | smooth_scale | 7.1766 | 11.4514 | 2026-07-18 15:22:46 |
| aw14 | routed_tail | 14 | 2048 | 64 | w8a8 | smooth_scale | 7.1532 | 11.4074 | 2026-07-18 15:22:46 |
| aw15 | routed_tail | 15 | 2048 | 64 | w8a8 | smooth_scale | 7.1409 | 11.4021 | 2026-07-18 15:22:46 |
| aw16 | routed_tail | 16 | 2048 | 64 | w8a8 | smooth_scale | 7.1428 | 11.4018 | 2026-07-18 15:22:46 |

## Older Group1 Allocation/Tier Ablations

| category | label | metric | weight_group | ose_topk | ose_quant | wikitext2 | c4 | timestamp_readable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| allocation_component_ablation | all_three |  | 1 |  |  | 6.9024 | 11.1318 | 2026-07-14 15:13:17 |
| allocation_component_ablation | no_down_range |  | 1 |  |  | 6.9091 | 11.1317 | 2026-07-14 15:13:17 |
| allocation_component_ablation | no_input_range |  | 1 |  |  | 6.8973 | 11.1254 | 2026-07-14 15:13:17 |
| allocation_component_ablation | no_weight_range |  | 1 |  |  | 6.9053 | 11.1321 | 2026-07-14 15:13:17 |
| allocation_formula_ablation | no_both_halves |  | 1 |  |  | 6.9083 | 11.1333 | 2026-07-14 12:07:41 |
| allocation_formula_ablation | no_log_no_halves |  | 1 |  |  | 6.9024 | 11.1318 | 2026-07-14 12:07:41 |
| allocation_formula_ablation | no_route_half |  | 1 |  |  | 6.9101 | 11.1345 | 2026-07-14 12:07:41 |
| allocation_formula_ablation | no_weight_half |  | 1 |  |  | 6.9106 | 11.1341 | 2026-07-14 12:07:41 |
| allocation_metric_ablation_group1 | kurtosis | kurtosis | 1 |  |  | 7.0188 | 11.2021 | 2026-07-15 01:38:36 |
| allocation_metric_ablation_group1 | residual_qerr | residual_qerr | 1 |  |  | 7.1445 | 11.3280 | 2026-07-15 01:38:36 |
| allocation_metric_ablation_group1 | residual_qerr_gain | residual_qerr_gain | 1 |  |  | 7.1449 | 11.3308 | 2026-07-15 01:38:36 |
| allocation_metric_ablation_group1 | route | route | 1 |  |  | 6.9220 | 11.1863 | 2026-07-15 01:38:36 |
| allocation_metric_ablation_group1 | tail | tail | 1 |  |  | 7.0082 | 11.2524 | 2026-07-15 01:38:36 |
| allocation_source_ablation | down_only |  | 1 |  |  | 6.9782 | 11.1919 | 2026-07-14 17:27:01 |
| allocation_source_ablation | weight_all_only |  | 1 |  |  | 6.9033 | 11.1324 | 2026-07-14 17:27:01 |
| allocation_source_ablation | weight_down_only |  | 1 |  |  | 6.9052 | 11.1312 | 2026-07-14 17:27:01 |
| allocation_source_ablation | weight_gate_only |  | 1 |  |  | 6.9872 | 11.1763 | 2026-07-14 17:27:01 |
| allocation_source_ablation | weight_up_only |  | 1 |  |  | 6.9400 | 11.1977 | 2026-07-14 17:27:01 |
| eaquant_act_per_tensor_group1 | eaquant_act_per_tensor |  | 1 | 0 |  | 7.0567 | 11.1925 | 2026-07-19 19:01:18 |
| gateup_duquant_vs_ose_repeat_group2048 | full_duquant |  | 1 | 0 | same | 7.3007 | 11.5620 | 2026-07-20 02:17:04 |
| gateup_duquant_vs_ose_repeat_group2048 | no_gateup_no_ose |  | 1 | 0 | same | 7.4748 | 11.7005 | 2026-07-20 02:17:04 |
| gateup_duquant_vs_ose_repeat_group2048 | ours_ose_fp16 |  | 1 | 64 | fp16 | 7.2250 | 11.4535 | 2026-07-20 02:17:04 |
| gateup_duquant_vs_ose_repeat_group2048 | ours_ose_w8a8 |  | 1 | 64 | w8a8 | 7.2137 | 11.4510 | 2026-07-20 02:17:04 |
| range_route_ablation_group1 | range | range | 1 |  |  | 6.9692 | 11.1595 | 2026-07-15 21:24:01 |
| range_route_ablation_group1 | route_range | route_range | 1 |  |  | 6.9024 | 11.1318 | 2026-07-15 21:24:01 |
| routed_metric_extra_group1 | routed_kurtosis | routed_kurtosis | 1 |  |  | 6.9778 | 11.1994 | 2026-07-15 22:14:29 |
| routed_metric_extra_group1 | routed_tail | routed_tail | 1 |  |  | 6.9257 | 11.1835 | 2026-07-15 22:14:29 |
| tier_ablation_route_range_group1 | 2tier_w4a4_w5a8 | route_range | 1 |  |  | 6.9146 | 11.1315 | 2026-07-16 00:37:33 |
| tier_ablation_route_range_group1 | 2tier_w4a4_w6a8 | route_range | 1 |  |  | 6.9813 | 11.2057 | 2026-07-16 00:37:33 |
| tier_ablation_route_range_group1 | 2tier_w4a4_w8a8 | route_range | 1 |  |  | 7.0491 | 11.2825 | 2026-07-16 00:37:33 |
| tier_ablation_route_range_group1 | 3tier_w4a4_w5a8_w6a8 | route_range | 1 |  |  | 6.9024 | 11.1318 | 2026-07-16 00:37:33 |
| tier_ablation_route_range_group1 | 4tier_add_w4a8 | route_range | 1 |  |  | 6.9095 | 11.1391 | 2026-07-16 00:37:33 |
| tier_ablation_route_range_group1 | 5tier_add_w8a8 | route_range | 1 |  |  | 6.9550 | 11.1783 | 2026-07-16 00:37:33 |