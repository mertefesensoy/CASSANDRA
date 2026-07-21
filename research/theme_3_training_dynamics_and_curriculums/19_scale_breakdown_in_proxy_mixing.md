# Scale Breakdown in Proxy Data Mixing

**Date Checked**: 2026-07-21
**Source 1**: "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance" (Ye et al., 2024)
**Source 2**: Recent literature on scaling laws and proxy capacity competition.

## What the Source Claims
While Data Mixing Laws (like DoReMi and predictive mixture models) can mathematically optimize training ratios, the assumption of **scale invariance**—that the optimal mixture for a small proxy model is the optimal mixture for a massive target model—often breaks down. 
As models increase in parameter count, their capacity to absorb and partition diverse, competing knowledge domains shifts non-monotonically. A very small proxy model experiences intense "capacity competition"; it might penalize complex domains too heavily because it lacks the representation space to learn them alongside simpler data. A larger target model has enough capacity to isolate those domains without interference. Therefore, predictive extrapolation from an undersized proxy will drastically miscalculate the loss topography of the larger target.

## Direct Relevance to Cassandra
In Stage 59/60, the attempt to use a frozen 3.2M proxy to predict the data mixing toll for an 85M model failed entirely. The 3.2M proxy over-shot the measured 85M cost by 2.09x, landing on the wrong side of the lab's KILL line. This perfectly mirrors the literature: proxy-predicted data decisions are invalid when the proxy crosses a capacity-competition threshold relative to the target.

## Roadmap & Experiment Change
**Hypothesis formulation for Phase 7 (Intake decision)**: Proxy-predicted data mixture decisions are out of the toolkit at this scale. Data mixtures for the 85M and 200M models must either rely on heuristic static mixtures (like the 25% TinyStories / 75% text8 split) or require active, continuous dynamic routing on the target model itself, as offline 3M proxy sweeps are definitively proven to be a structurally broken proxy.
