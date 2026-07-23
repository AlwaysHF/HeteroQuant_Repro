# OSE Channel Selection Metric Ablation

Fixed config: OLMoE, route_range allocation, 3-tier w4a4/w5a8/w6a8, group_size=1, OSE topK=64, OSE branch=w8a8. Lower PPL is better.

| selection metric | WikiText2 PPL | ΔWikiText2 vs shared_scale | C4 PPL | ΔC4 vs shared_scale |
|---|---:|---:|---:|---:|
| `expert_max` | 6.8946 | -0.0078 | 11.1238 | -0.0080 |
| `shared_scale` | 6.9024 | +0.0000 | 11.1318 | +0.0000 |
| `expert_p99_mean` | 6.9716 | +0.0693 | 11.2248 | +0.0930 |
| `expert_error` | 7.0242 | +0.1218 | 11.2761 | +0.1443 |

Best by both WikiText2 and C4: `expert_max`.
