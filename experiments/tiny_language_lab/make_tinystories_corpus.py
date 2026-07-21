from __future__ import annotations

import argparse
import io
import json
import unicodedata
from collections import Counter
from pathlib import Path

from make_natural_text_corpus import DEFAULT_ALPHABET, normalize_text, split_at_index


DEFAULT_SOURCE_DIR = Path(__file__).with_name("corpus") / "tinystories_raw"
DEFAULT_OUT = Path(__file__).with_name("corpus") / "tinystories_char_seed.txt"
SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".json"}
IGNORED_DISCOVERY_SUFFIXES = (".download.json",)


class StreamingNormalizer:
    def __init__(self, alphabet: str) -> None:
        self.allowed = set(alphabet)
        self.writer = io.StringIO()
        self.started = False
        self.last = ""
        self.newline_run = 0
        self.pending_space = False
        self.replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": " ",
            "\u201d": " ",
            "\t": " ",
        }

    def feed(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFKD", text).lower()
        for char in text:
            if unicodedata.combining(char):
                continue
            char = self.replacements.get(char, char)
            if char not in self.allowed:
                char = " " if " " in self.allowed else ""
            if not char:
                continue
            self._emit(char)

    def _emit(self, char: str) -> None:
        if char == " ":
            if self.started and self.last != "\n":
                self.pending_space = True
            return
        if char == "\n" and "\n" in self.allowed:
            self.pending_space = False
            if self.started and self.newline_run < 2:
                self.writer.write("\n")
                self.last = "\n"
                self.newline_run += 1
            return
        if self.pending_space and self.started and self.last != "\n":
            self.writer.write(" ")
        self.pending_space = False
        self.writer.write(char)
        self.started = True
        self.last = char
        self.newline_run = 0

    def finish(self) -> str:
        normalized = self.writer.getvalue().strip()
        return normalized + "\n"


def collect_source_files(sources: list[Path], source_dir: Path) -> list[Path]:
    def discoverable(path: Path) -> bool:
        name = path.name.lower()
        return path.suffix.lower() in SUPPORTED_SUFFIXES and not any(
            name.endswith(suffix) for suffix in IGNORED_DISCOVERY_SUFFIXES
        )

    candidates: list[Path] = []
    for source in sources:
        if source.is_dir():
            candidates.extend(path for path in source.rglob("*") if discoverable(path))
        else:
            candidates.append(source)

    if not candidates and source_dir.exists():
        candidates.extend(path for path in source_dir.rglob("*") if discoverable(path))

    unique: dict[Path, None] = {}
    for path in candidates:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported source suffix for {path}; use .txt, .jsonl, or .json")
        if not path.exists():
            raise FileNotFoundError(path)
        unique[path.resolve()] = None
    return sorted(unique)


def read_json_text(value: object, text_key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        text = value.get(text_key)
        return [text] if isinstance(text, str) else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(read_json_text(item, text_key))
        return texts
    return []


def read_source(path: Path, text_key: str) -> tuple[list[str], int]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        return [text], 1
    if suffix == ".jsonl":
        texts: list[str] = []
        records = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            extracted = read_json_text(value, text_key)
            texts.extend(extracted)
            records += len(extracted)
        return texts, records
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        texts = read_json_text(value, text_key)
        return texts, len(texts)
    raise ValueError(f"Unsupported source suffix for {path}")


def write_shards(text: str, split_at: int, shard_dir: Path, shard_chars: int) -> list[dict[str, object]]:
    if shard_chars <= 0:
        raise ValueError("--shard-chars must be positive when --shard-dir is set")

    shard_dir.mkdir(parents=True, exist_ok=True)
    train_text = text[:split_at]
    val_text = text[split_at:]
    shards: list[dict[str, object]] = []
    for index, start in enumerate(range(0, len(train_text), shard_chars)):
        shard_text = train_text[start : start + shard_chars]
        shard_path = shard_dir / f"train_{index:05d}.txt"
        shard_path.write_text(shard_text, encoding="utf-8")
        shards.append({"path": str(shard_path), "split": "train", "chars": len(shard_text)})
    val_path = shard_dir / "val.txt"
    val_path.write_text(val_text, encoding="utf-8")
    shards.append({"path": str(val_path), "split": "val", "chars": len(val_text)})
    return shards


def build_metadata(
    source_files: list[Path],
    raw_chars: int,
    raw_records: int,
    normalized_text: str,
    alphabet: str,
    val_fraction: float,
    block_size: int,
    shards: list[dict[str, object]],
) -> dict[str, object]:
    counts = Counter(normalized_text)
    split_at = split_at_index(len(normalized_text), val_fraction, block_size)
    return {
        "generator": "make_tinystories_corpus.py",
        "source_files": [str(path) for path in source_files],
        "input_chars": raw_chars,
        "input_records": raw_records,
        "normalized_chars": len(normalized_text),
        "normalization": [
            "unicode NFKD",
            "lowercase",
            "unsupported characters mapped to space",
            "space runs collapsed",
            "blank-line runs capped at two newlines",
        ],
        "alphabet": alphabet,
        "observed_vocab": sorted(counts),
        "observed_vocab_size": len(counts),
        "observed_counts": {char: counts[char] for char in sorted(counts)},
        "val_fraction": val_fraction,
        "block_size_for_split_record": block_size,
        "split_protocol": "same deterministic prefix train and suffix validation split as cassandra_tiny_transformer.py",
        "train_chars_at_recorded_block_size": split_at,
        "val_chars_at_recorded_block_size": len(normalized_text) - split_at,
        "shards": shards,
        "phase2_note": "character-level TinyStories prep; no pretrained checkpoint is loaded",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a character-level TinyStories corpus for Cassandra Phase 2")
    parser.add_argument("--source", type=Path, nargs="*", default=[])
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--json-text-key", type=str, default="text")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)
    parser.add_argument("--min-chars", type=int, default=1_000_000)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--story-separator", type=str, default="\\n\\n")
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--shard-chars", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_chars < 1:
        raise ValueError("--min-chars must be positive")
    if args.max_chars < 0:
        raise ValueError("--max-chars must be non-negative")
    if not 0.05 <= args.val_fraction <= 0.5:
        raise ValueError("--val-fraction must be between 0.05 and 0.5")
    if args.block_size < 8:
        raise ValueError("--block-size must be at least 8")

    alphabet = args.alphabet.replace("\\n", "\n")
    separator = args.story_separator.replace("\\n", "\n")
    source_files = collect_source_files(args.source, args.source_dir)
    if not source_files:
        raise FileNotFoundError(
            f"No TinyStories source files found. Put .txt, .jsonl, or .json files in {args.source_dir} "
            "or pass --source explicitly."
        )

    normalizer = StreamingNormalizer(alphabet)
    first_piece = True
    raw_chars = 0
    raw_records = 0
    for path in source_files:
        if path.suffix.lower() == ".txt":
            if not first_piece:
                normalizer.feed(separator)
            with path.open("r", encoding="utf-8") as handle:
                while True:
                    chunk = handle.read(1_048_576)
                    if not chunk:
                        break
                    raw_chars += len(chunk)
                    normalizer.feed(chunk)
            raw_records += 1
            first_piece = False
            continue

        texts, records = read_source(path, args.json_text_key)
        for text in texts:
            if not first_piece:
                normalizer.feed(separator)
            raw_chars += len(text)
            normalizer.feed(text)
            first_piece = False
        raw_records += records

    if raw_records == 0:
        raise ValueError("No text records were extracted from the source files")

    normalized_text = normalizer.finish()
    if args.max_chars > 0:
        normalized_text = normalized_text[: args.max_chars].rstrip() + "\n"
    if len(normalized_text) < args.min_chars:
        raise ValueError(
            f"Normalized corpus is too short: {len(normalized_text)} chars "
            f"after normalization, wanted at least {args.min_chars}"
        )

    split_at = split_at_index(len(normalized_text), args.val_fraction, args.block_size)
    shards: list[dict[str, object]] = []
    if args.shard_dir is not None:
        shards = write_shards(normalized_text, split_at, args.shard_dir, args.shard_chars)

    metadata = build_metadata(
        source_files,
        raw_chars,
        raw_records,
        normalized_text,
        alphabet,
        args.val_fraction,
        args.block_size,
        shards,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(normalized_text, encoding="utf-8")
    metadata_out = args.metadata_out or args.out.with_suffix(".meta.json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"wrote {args.out} ({len(normalized_text)} chars, "
        f"train={split_at}, val={len(normalized_text) - split_at}, "
        f"vocab={metadata['observed_vocab_size']})"
    )
    print(f"wrote {metadata_out}")
    if shards:
        print(f"wrote {len(shards)} shards to {args.shard_dir}")


if __name__ == "__main__":
    main()
