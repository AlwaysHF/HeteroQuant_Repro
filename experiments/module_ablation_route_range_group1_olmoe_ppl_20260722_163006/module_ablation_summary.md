# Module Ablation: OLMoE route_range group1 PPL

Fixed quantization base: W4A8, group_size=1, router group_size=1, PPL only. Full method reference is the existing `route_range`, selected OSE topK=64, 3-tier allocation result and was not rerun by request.

| variant | migration | OSE | tier | WikiText2 PPL | ΔWiki vs full | C4 PPL | ΔC4 vs full |
|---|---:|---:|---|---:|---:|---:|---:|
| No migration, no OSE, no tier | 0 | 0 | none | 7.2718 | +0.3694 | 11.4414 | +0.3095 |
| Migration only | 1 | 0 | none | 7.7714 | +0.8691 | 11.8737 | +0.7419 |
| Migration + random OSE | 1 | 1 | none | 7.6980 | +0.7957 | 11.8248 | +0.6929 |
| Migration + selected OSE | 1 | 1 | none | 7.0656 | +0.1632 | 11.2318 | +0.0999 |
| Migration + selected OSE + random tier | 1 | 1 | random | 7.0086 | +0.1063 | 11.2023 | +0.0705 |
| Full method reference | 1 | 1 | route_range | 6.9024 | +0.0000 | 11.1318 | +0.0000 |

## Observations

- Migration alone worsens PPL under uniform W4A8, so it needs OSE to protect migrated outlier channels.
- Random OSE gives only a small recovery over migration alone; selected OSE is the main contributor.
- Random same-budget tier is better than no tier, but still worse than the full route_range tier assignment.
