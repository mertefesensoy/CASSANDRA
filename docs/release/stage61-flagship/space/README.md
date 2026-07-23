---
title: Cassandra 200M text8
emoji: 🔤
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: apache-2.0
models:
- mertefesensoy/cassandra-200m-text8
---

# Cassandra 200M text8 demo

An interactive demo of the
[cassandra-200m-text8](https://huggingface.co/mertefesensoy/cassandra-200m-text8)
character language model: 201M parameters, trained from scratch on text8
(Wikipedia) on a single laptop GPU.

Recommended settings are temperature 0.8 and top-p 0.9. The model writes fluent
Wikipedia-style prose locally but drifts off topic across long generations,
because its context window is only 256 characters. It is a research and
education artifact, not a factual assistant.

Free CPU Spaces generate character-by-character slowly, so outputs are kept
short. Upgrade the Space to GPU hardware for faster, longer generations.
