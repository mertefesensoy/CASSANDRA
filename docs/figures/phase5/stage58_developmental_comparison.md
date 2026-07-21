# Stage 58 Developmental Comparison

Primary metric: deterministic chunked text8 TEST bits/char. Lower is better.

| Arm | Global step | text8 TEST bits/char | TEST NLL | Characters evaluated |
| --- | ---: | ---: | ---: | ---: |
| COLD | 42,000 | 1.357318 | 0.940821 | 4,999,936 |
| CURRICULUM | 42,000 | 1.362414 | 0.944354 | 4,999,936 |
| MIXTURE | 42,000 | 1.385295 | 0.960213 | 4,999,936 |

- Primary delta, CURRICULUM minus COLD: `+0.005096` bits/char.
- Seed-7 primary read: **E-null**.
- Secondary delta, MIXTURE minus COLD: `+0.027977` bits/char.
- Secondary delta, MIXTURE minus CURRICULUM: `+0.022881` bits/char.

This generator reports the seed-7 read only. The reduced-budget seed-11 and seed-19 sign replicas are generated separately by `make_h024_replica_figure.py` into `h024_replica_sign_check.json`; the combined three-seed verdict is recorded in ADR 0016.

## Retention Series

| Arm | Global step | TinyStories val bits/char | NLL | Characters evaluated |
| --- | ---: | ---: | ---: | ---: |
| COLD | 5,000 | 3.654046 | 2.532791 | 1,499,904 |
| COLD | 10,000 | 3.734058 | 2.588252 | 1,499,904 |
| COLD | 15,000 | 3.829376 | 2.654321 | 1,499,904 |
| COLD | 20,000 | 3.645778 | 2.527060 | 1,499,904 |
| COLD | 25,000 | 3.672875 | 2.545843 | 1,499,904 |
| COLD | 30,000 | 3.606769 | 2.500022 | 1,499,904 |
| COLD | 35,000 | 3.561688 | 2.468774 | 1,499,904 |
| COLD | 40,000 | 3.571101 | 2.475299 | 1,499,904 |
| COLD | 42,000 | 3.556502 | 2.465180 | 1,499,904 |
| CURRICULUM | 5,000 | 0.993415 | 0.688583 | 1,499,904 |
| CURRICULUM | 10,000 | 0.903474 | 0.626240 | 1,499,904 |
| CURRICULUM | 12,500 | 0.877782 | 0.608432 | 1,499,904 |
| CURRICULUM | 15,000 | 3.188760 | 2.210280 | 1,499,904 |
| CURRICULUM | 20,000 | 3.428489 | 2.376448 | 1,499,904 |
| CURRICULUM | 25,000 | 3.548431 | 2.459585 | 1,499,904 |
| CURRICULUM | 30,000 | 3.553305 | 2.462963 | 1,499,904 |
| CURRICULUM | 35,000 | 3.553964 | 2.463420 | 1,499,904 |
| CURRICULUM | 40,000 | 3.507700 | 2.431352 | 1,499,904 |
| CURRICULUM | 42,000 | 3.529069 | 2.446164 | 1,499,904 |
| MIXTURE | 5,000 | 1.170210 | 0.811128 | 1,499,904 |
| MIXTURE | 10,000 | 1.061833 | 0.736006 | 1,499,904 |
| MIXTURE | 15,000 | 1.004973 | 0.696594 | 1,499,904 |
| MIXTURE | 20,000 | 0.962061 | 0.666850 | 1,499,904 |
| MIXTURE | 25,000 | 0.920961 | 0.638361 | 1,499,904 |
| MIXTURE | 30,000 | 0.882576 | 0.611755 | 1,499,904 |
| MIXTURE | 35,000 | 0.851025 | 0.589886 | 1,499,904 |
| MIXTURE | 40,000 | 0.831327 | 0.576232 | 1,499,904 |
| MIXTURE | 42,000 | 0.826285 | 0.572737 | 1,499,904 |

Figures: `fig1_stage58_text8_primary.png`, `fig2_stage58_tinystories_retention.png`, and `fig4_phase5_arc_and_recipe.png`.
