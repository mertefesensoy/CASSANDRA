"""Cassandra model playground: a local web UI for trying the Stage 55 models.

Two tabs:
  Generate  - pick a model, type a prompt, tune temperature / top-k / length.
  Blind A/B - same prompt goes to two randomly picked models shown unlabeled;
              vote which reads better; identities are revealed after the vote
              and every vote is appended to runs/human_ab_votes.jsonl. This
              turns the ADR 0013 required human review into structured,
              bias-resistant evidence.

Run from the repo root (opens a browser tab on localhost):
  python .\\experiments\\tiny_language_lab\\playground.py
  python .\\experiments\\tiny_language_lab\\playground.py --device cpu   # keep GPU free

Nothing leaves the machine; gradio runs purely locally with a loopback bind.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import torch

from flagship_eval_lib import (
    FINAL_CHECKPOINTS,
    LAB_DIR,
    load_model,
    sample_text,
)

VOTES_PATH = LAB_DIR / "runs" / "human_ab_votes.jsonl"

# Historical seed prompts spanning three eras of LM sampling, for prompt
# variety in review rounds. All are normalized into the 33-char alphabet
# on use. Sources: Sutskever, Martens, Hinton 2011 (char-RNN demo seed);
# OpenAI GPT-2 announcement samples, 2019; Eldan and Li, TinyStories, 2023
# (their eval completes 50 GPT-4-written story beginnings).
SUGGESTED_PROMPTS = {
    "TinyStories opener (house default)": "once upon a time ",
    "TinyStories eval seed, Eldan and Li 2023": "once upon a time, in an ancient house, ",
    "TinyStories register, friends": "tom and jane are friends. one day, ",
    "TinyStories register, discovery": "one day, a little girl found ",
    "char-RNN era seed, Sutskever et al. 2011": "the meaning of life is ",
    "GPT-2 unicorn prompt, OpenAI 2019": (
        "in a shocking finding, scientist discovered a herd of unicorns "
        "living in a remote, previously unexplored valley, in the andes "
        "mountains. even more surprising to the researchers was the fact "
        "that the unicorns spoke perfect english. "
    ),
    "wikipedia register (H022 probe prompt)": "the history of ",
}

_cache: dict[str, tuple] = {}
_device = "cuda"


def get_model(name: str):
    if name not in _cache:
        # Keep at most two models resident so the 8 GB card never fights for room.
        while len(_cache) >= 2:
            evicted = next(iter(_cache))
            model, _ = _cache.pop(evicted)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model, codec, _, _ = load_model(FINAL_CHECKPOINTS[name], device=_device)
        _cache[name] = (model, codec)
    return _cache[name]


def generate(name: str, prompt: str, temperature: float, top_k: int, max_new: int, seed: int) -> str:
    model, codec = get_model(name)
    return sample_text(
        model,
        codec,
        prompt or "once upon a time ",
        max_new_chars=int(max_new),
        temperature=float(temperature),
        top_k=int(top_k),
        seed=int(seed),
        device=_device,
    )


PAIR_RANDOM = "random (default)"


def pair_choices() -> list[str]:
    names = sorted(FINAL_CHECKPOINTS.keys())
    pairs = [PAIR_RANDOM]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append(f"{names[i]} vs {names[j]}")
    return pairs


def ab_generate(prompt: str, temperature: float, max_new: int, pairing: str = PAIR_RANDOM):
    if pairing and pairing != PAIR_RANDOM and " vs " in pairing:
        names = pairing.split(" vs ")
    else:
        pairing = PAIR_RANDOM
        names = random.sample(sorted(FINAL_CHECKPOINTS.keys()), 2)
    names = list(names)
    # Position stays random even for a pinned pair, or the reviewer would
    # know which side is which and the blindness is lost.
    random.shuffle(names)
    seed = random.randint(1, 10_000_000)
    out_a = generate(names[0], prompt, temperature, 0, max_new, seed)
    out_b = generate(names[1], prompt, temperature, 0, max_new, seed + 1)
    state = {"model_a": names[0], "model_b": names[1], "prompt": prompt,
             "temperature": temperature, "seed": seed, "pairing": pairing}
    return out_a, out_b, state, "Two anonymous models answered. Vote below."


def ab_vote(vote: str, state: dict):
    if not state:
        return "Generate a pair first."
    record = dict(state)
    record["vote"] = vote
    record["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
    VOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VOTES_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return (
        f"Vote recorded: {vote}. Model A was {state['model_a']}, "
        f"Model B was {state['model_b']}. Logged to {VOTES_PATH.name}."
    )


def main() -> None:
    global _device
    parser = argparse.ArgumentParser(description="Cassandra model playground")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    _device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    import gradio as gr

    model_names = sorted(FINAL_CHECKPOINTS.keys())
    with gr.Blocks(title="Cassandra Tiny Language Lab Playground") as demo:
        gr.Markdown(
            "# Cassandra playground\n"
            "Character-level TinyStories models from Stage 51 and Stage 55. "
            "Prompts are lowercased and mapped into the 33-char alphabet "
            "(letters, space, `.,!?'` and newline)."
        )
        with gr.Tab("Generate"):
            with gr.Row():
                model_dd = gr.Dropdown(model_names, value="flagship_200m_50k_seed7", label="Model")
                seed_nb = gr.Number(value=7, precision=0, label="Seed")
            gen_sugg = gr.Dropdown(
                list(SUGGESTED_PROMPTS.keys()),
                value=None,
                label="Suggested prompts (historical seeds; picking one fills the box)",
            )
            prompt_tb = gr.Textbox(value="once upon a time ", label="Prompt", lines=2)
            gen_sugg.change(
                lambda k: SUGGESTED_PROMPTS.get(k, "once upon a time "),
                inputs=gen_sugg,
                outputs=prompt_tb,
            )
            with gr.Row():
                temp_sl = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                topk_sl = gr.Slider(0, 33, value=0, step=1, label="Top-k (0 = off)")
                len_sl = gr.Slider(50, 1500, value=400, step=50, label="New characters")
            gen_btn = gr.Button("Generate", variant="primary")
            out_tb = gr.Textbox(label="Continuation", lines=12)
            gen_btn.click(
                generate,
                inputs=[model_dd, prompt_tb, temp_sl, topk_sl, len_sl, seed_nb],
                outputs=out_tb,
            )
        with gr.Tab("Blind A/B review"):
            gr.Markdown(
                "Same prompt, two randomly chosen models, unlabeled. Vote which "
                "continuation reads better; identities are revealed after your "
                "vote and appended to `runs/human_ab_votes.jsonl`."
            )
            ab_sugg = gr.Dropdown(
                list(SUGGESTED_PROMPTS.keys()),
                value=None,
                label="Suggested prompts (vary these across votes; the review protocol wants 5 or more distinct prompts)",
            )
            ab_prompt = gr.Textbox(value="once upon a time ", label="Prompt", lines=2)
            ab_sugg.change(
                lambda k: SUGGESTED_PROMPTS.get(k, "once upon a time "),
                inputs=ab_sugg,
                outputs=ab_prompt,
            )
            with gr.Row():
                ab_temp = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                ab_len = gr.Slider(100, 1000, value=400, step=50, label="New characters")
            ab_pair = gr.Dropdown(
                pair_choices(),
                value=PAIR_RANDOM,
                label="Pairing (random is the honest default; pin a pair only for a targeted round)",
            )
            ab_btn = gr.Button("Generate pair", variant="primary")
            with gr.Row():
                ab_out_a = gr.Textbox(label="Model A", lines=10)
                ab_out_b = gr.Textbox(label="Model B", lines=10)
            ab_state = gr.State()
            ab_status = gr.Markdown()
            with gr.Row():
                vote_a = gr.Button("A reads better")
                vote_tie = gr.Button("Tie / both weak")
                vote_b = gr.Button("B reads better")
            ab_btn.click(
                ab_generate,
                inputs=[ab_prompt, ab_temp, ab_len, ab_pair],
                outputs=[ab_out_a, ab_out_b, ab_state, ab_status],
            )
            vote_a.click(lambda s: ab_vote("A", s), inputs=ab_state, outputs=ab_status)
            vote_tie.click(lambda s: ab_vote("tie", s), inputs=ab_state, outputs=ab_status)
            vote_b.click(lambda s: ab_vote("B", s), inputs=ab_state, outputs=ab_status)

    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True, share=False)


if __name__ == "__main__":
    main()
