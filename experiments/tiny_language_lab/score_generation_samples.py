from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_RUN = Path(__file__).with_name("runs") / "phase2_tinystories_modern_b500.jsonl"
DEFAULT_CORPUS = Path(__file__).with_name("corpus") / "tinystories_char_seed.txt"
DEFAULT_OUT = Path(__file__).with_name("runs") / "phase2_tinystories_modern_b500_generation_quality.md"

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
STORY_CUE_WORDS = {
    "once",
    "time",
    "there",
    "was",
    "little",
    "girl",
    "boy",
    "mom",
    "dad",
    "friend",
    "friends",
    "said",
    "asked",
    "went",
    "wanted",
    "found",
    "saw",
    "happy",
    "sad",
    "play",
    "day",
    "home",
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def corpus_vocabulary(path: Path, min_count: int) -> set[str]:
    if not path.exists():
        return set()
    words = WORD_RE.findall(path.read_text(encoding="utf-8").lower())
    counts = Counter(words)
    return {word for word, count in counts.items() if count >= min_count}


def repeated_char_fraction(text: str) -> float:
    if not text:
        return 0.0
    repeated = 0
    run_char = ""
    run_len = 0
    for char in text:
        if char == run_char:
            run_len += 1
        else:
            run_char = char
            run_len = 1
        if run_len >= 4:
            repeated += 1
    return repeated / len(text)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return statistics.mean(values)


def score_sample(row: dict[str, object], known_words: set[str]) -> dict[str, object]:
    sample = str(row.get("sample", ""))
    prompt = str(row.get("used_prompt", ""))
    prompt_ok = bool(prompt) and sample.startswith(prompt)
    completion = sample[len(prompt) :] if prompt_ok else sample
    completion = completion.strip()
    words = WORD_RE.findall(completion.lower())
    word_count = len(words)
    known_count = sum(1 for word in words if word in known_words) if known_words else 0
    known_ratio = known_count / word_count if word_count else 0.0
    story_cues = sum(1 for word in words if word in STORY_CUE_WORDS)
    avg_word_len = mean(len(word) for word in words)
    sentence_marks = sum(1 for char in completion if char in ".?!")
    unique_ratio = len(set(words)) / word_count if word_count else 0.0
    repeated_fraction = repeated_char_fraction(completion)
    weird_words = sum(1 for word in words if len(word) > 16 or re.search(r"(.)\1\1\1", word))
    weird_ratio = weird_words / word_count if word_count else 0.0
    lower_completion = completion.lower()
    bad_marker = "endoftext" in lower_completion
    repeated_prompt = bool(prompt) and prompt.strip().lower() in lower_completion

    relevance = 0
    if prompt_ok and word_count >= 8:
        relevance = 1
    if prompt_ok and story_cues >= 3 and word_count >= 20 and not bad_marker:
        relevance = 2

    grammar = 0
    if word_count >= 8 and known_ratio >= 0.70 and weird_ratio <= 0.25 and not bad_marker:
        grammar = 1
    if (
        word_count >= 20
        and known_ratio >= 0.93
        and weird_ratio <= 0.12
        and repeated_fraction <= 0.02
        and 2.0 <= avg_word_len <= 8.5
        and sentence_marks >= 2
        and not bad_marker
    ):
        grammar = 2

    coherence = 0
    if word_count >= 15 and known_ratio >= 0.70 and unique_ratio >= 0.25 and not bad_marker:
        coherence = 1
    if (
        word_count >= 40
        and known_ratio >= 0.90
        and sentence_marks >= 3
        and story_cues >= 6
        and 0.25 <= unique_ratio <= 0.95
        and weird_ratio <= 0.12
        and not repeated_prompt
        and not bad_marker
    ):
        coherence = 2

    return {
        "comparison_name": row.get("comparison_name"),
        "seed": row.get("seed"),
        "val_nll": row.get("val_nll"),
        "val_bits_per_char": row.get("val_bits_per_char"),
        "prompt_ok": prompt_ok,
        "completion_chars": len(completion),
        "word_count": word_count,
        "known_word_ratio": round(known_ratio, 6),
        "story_cues": story_cues,
        "sentence_marks": sentence_marks,
        "unique_word_ratio": round(unique_ratio, 6),
        "repeated_char_fraction": round(repeated_fraction, 6),
        "weird_word_ratio": round(weird_ratio, 6),
        "bad_marker": bad_marker,
        "repeated_prompt": repeated_prompt,
        "coherence": coherence,
        "grammar": grammar,
        "relevance": relevance,
        "total": coherence + grammar + relevance,
        "sample_excerpt": sample[:260].replace("\n", "\\n"),
    }


def markdown_table(rows: list[dict[str, object]], title: str, run_paths: list[Path], corpus: Path) -> str:
    lines = [
        f"# {title}",
        "",
        "Scores are deterministic proxy scores, not a human-quality claim. Each",
        "dimension is `0` to `2`; total is `0` to `6`. The rubric checks prompt",
        "prefix adherence, story-cue presence, corpus-word ratio, sentence",
        "punctuation, repetition, and simple gibberish indicators.",
        "",
        f"Runs: {', '.join(str(path) for path in run_paths)}",
        f"Vocabulary source: `{corpus}`",
        "",
        "## Mean Scores",
        "",
        "| Config | Rows | Mean total | Mean coherence | Mean grammar | Mean relevance | Mean val NLL | Mean bits/char |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["comparison_name"]), []).append(row)
    for name, group in sorted(groups.items()):
        lines.append(
            "| "
            f"{name} | {len(group)} | "
            f"{mean(float(row['total']) for row in group):.3f} | "
            f"{mean(float(row['coherence']) for row in group):.3f} | "
            f"{mean(float(row['grammar']) for row in group):.3f} | "
            f"{mean(float(row['relevance']) for row in group):.3f} | "
            f"{mean(float(row['val_nll']) for row in group):.6f} | "
            f"{mean(float(row['val_bits_per_char']) for row in group):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Raw Scores",
            "",
            "| Config | Seed | Total | Coherence | Grammar | Relevance | Known words | Story cues | Sentences | Bad marker | Prompt repeated | Prompt OK | Excerpt |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['comparison_name']} | {row['seed']} | {row['total']} | "
            f"{row['coherence']} | {row['grammar']} | {row['relevance']} | "
            f"{row['known_word_ratio']:.6f} | {row['story_cues']} | "
            f"{row['sentence_marks']} | {row['bad_marker']} | "
            f"{row['repeated_prompt']} | {row['prompt_ok']} | "
            f"`{row['sample_excerpt']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved generation samples from Cassandra JSONL runs")
    parser.add_argument("--runs", type=Path, nargs="+", default=[DEFAULT_RUN])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--min-word-count", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", type=str, default="Phase 2 TinyStories Generation Quality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_word_count < 1:
        raise ValueError("--min-word-count must be positive")
    known_words = corpus_vocabulary(args.corpus, args.min_word_count)
    rows: list[dict[str, object]] = []
    for run_path in args.runs:
        for row in read_jsonl(run_path):
            rows.append(score_sample(row, known_words))
    if not rows:
        raise ValueError("No generation rows found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown_table(rows, args.title, args.runs, args.corpus), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
