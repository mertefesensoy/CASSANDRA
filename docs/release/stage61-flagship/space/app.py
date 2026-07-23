"""Gradio demo for the Cassandra 200M text8 character flagship.

Runs as a free CPU Hugging Face Space. Downloads the model from the model repo
on startup and generates with the recommended top-p 0.9 default. Generation is
character-by-character and slow on CPU, so outputs are kept short; upgrade the
Space to GPU hardware for snappier, longer generations.
"""

import random

import gradio as gr
from huggingface_hub import hf_hub_download

from flagship_eval_lib import load_model_from_safetensors, sample_text

MODEL_REPO = "mertefesensoy/cassandra-200m-text8"
DEVICE = "cpu"

weights = hf_hub_download(MODEL_REPO, "stage61_pure_broad_200m_text8_fp16.safetensors")
config = hf_hub_download(MODEL_REPO, "config.json")
codec_path = hf_hub_download(MODEL_REPO, "codec.json")
model, codec, cfg, meta = load_model_from_safetensors(
    weights, config, codec_path, device=DEVICE
)


def generate(prompt, temperature, top_p, max_new_chars, seed):
    used_seed = int(seed) if seed is not None else random.randint(0, 1_000_000)
    return sample_text(
        model,
        codec,
        prompt=prompt or "the ",
        max_new_chars=int(max_new_chars),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=used_seed,
        device=DEVICE,
    )


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(value="the history of ", label="Prompt (lowercase a-z and spaces; other characters are folded to space)"),
        gr.Slider(0.1, 1.2, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p (nucleus): 0.9 removes garbage"),
        gr.Slider(20, 400, value=120, step=20, label="Characters to generate (CPU is slow; keep short)"),
        gr.Number(value=7, label="Seed", precision=0),
    ],
    outputs=gr.Textbox(label="Generation"),
    title="Cassandra 200M text8 (character-level)",
    description=(
        "A 201M-parameter character language model trained from scratch on text8 "
        "(cleaned Wikipedia) on a single laptop GPU. Recommended settings: "
        "temperature 0.8, top-p 0.9. It writes fluent Wikipedia-style prose "
        "locally but drifts off topic across long generations, because its "
        "context window is only 256 characters. Research and education artifact, "
        "not a factual assistant."
    ),
    examples=[
        ["the history of ", 0.8, 0.9, 120, 7],
        ["the united states ", 0.8, 0.9, 120, 11],
        ["the city of ", 0.7, 0.95, 120, 19],
    ],
)

if __name__ == "__main__":
    demo.launch()
