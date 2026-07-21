# H024 Replica Sign Check

Primary metric: deterministic chunked text8 TEST bits/char. Lower is better.

| Seed | Budget | COLD | CURRICULUM | CURRICULUM minus COLD |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 42,000 | 1.357318 | 1.362414 | +0.005096 |
| 11 | 20,000 | 1.410154 | 1.419999 | +0.009845 |
| 19 | 20,000 | 1.410779 | 1.418571 | +0.007791 |

- Seed-7 registered primary read: **E-null**.
- Replica signs agree with seed 7: **True**.
- Seed-7 marginal-margin trigger: **False**.
- Full-budget replica escalation required: **False**.
- H024 conclusion: **seed-robust in sign; no full-budget replica escalation required**.

Figure: `fig3_h024_replica_sign_check.png`.
