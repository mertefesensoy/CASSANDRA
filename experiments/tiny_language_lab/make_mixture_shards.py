from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_TINY_DIR = LAB_DIR / "corpus" / "tinystories_char_shards_500mb"
DEFAULT_BROAD_DIR = LAB_DIR / "corpus" / "text8_char_shards"
DEFAULT_OUT_DIR = LAB_DIR / "corpus" / "mixture_char_shards"
DEFAULT_META = LAB_DIR / "corpus" / "mixture_char_shards.meta.json"


class CircularShardReader:
    def __init__(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("CircularShardReader needs at least one source shard")
        self.paths = paths
        self.file_index = 0
        self.text = ""
        self.offset = 0
        self.total_read = 0
        self.wraps = 0
        self._load_current()

    def _load_current(self) -> None:
        self.text = self.paths[self.file_index].read_text(encoding="utf-8")
        self.offset = 0
        if not self.text:
            raise ValueError(f"Empty shard: {self.paths[self.file_index]}")

    def read(self, count: int) -> str:
        if count <= 0:
            return ""
        pieces: list[str] = []
        remaining = count
        while remaining > 0:
            available = len(self.text) - self.offset
            take = min(available, remaining)
            pieces.append(self.text[self.offset : self.offset + take])
            self.offset += take
            self.total_read += take
            remaining -= take
            if self.offset >= len(self.text):
                self.file_index += 1
                if self.file_index >= len(self.paths):
                    self.file_index = 0
                    self.wraps += 1
                self._load_current()
        return "".join(pieces)


class ShardWriter:
    def __init__(self, out_dir: Path, shard_chars: int) -> None:
        if shard_chars <= 0:
            raise ValueError("--shard-chars must be positive")
        self.out_dir = out_dir
        self.shard_chars = shard_chars
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for old in self.out_dir.glob("train_*.txt"):
            old.unlink()
        self.index = 0
        self.buffer: list[str] = []
        self.buffer_chars = 0
        self.total_chars = 0
        self.hash = hashlib.sha256()
        self.shards: list[dict[str, object]] = []

    def write(self, text: str) -> None:
        if not text:
            return
        start = 0
        while start < len(text):
            remaining = self.shard_chars - self.buffer_chars
            piece = text[start : start + remaining]
            self.buffer.append(piece)
            self.buffer_chars += len(piece)
            self.total_chars += len(piece)
            self.hash.update(piece.encode("utf-8"))
            start += len(piece)
            if self.buffer_chars == self.shard_chars:
                self.flush()

    def flush(self) -> None:
        if self.buffer_chars == 0:
            return
        path = self.out_dir / f"train_{self.index:05d}.txt"
        path.write_text("".join(self.buffer), encoding="utf-8")
        self.shards.append({"path": str(path), "chars": self.buffer_chars, "split": "train"})
        self.index += 1
        self.buffer = []
        self.buffer_chars = 0


def train_shards(shard_dir: Path) -> list[Path]:
    paths = sorted(shard_dir.glob("train_*.txt"))
    if not paths:
        paths = sorted(path for path in shard_dir.glob("*.txt") if path.name.lower() != "val.txt")
    if not paths:
        raise FileNotFoundError(f"No train shards found in {shard_dir}")
    return paths


def shard_summary(paths: list[Path]) -> dict[str, object]:
    return {
        "dir": str(paths[0].parent) if paths else "",
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "chars": sum(len(path.read_text(encoding="utf-8")) for path in paths),
        "names": [path.name for path in paths],
    }


def build_mixture(
    tiny_dir: Path,
    broad_dir: Path,
    out_dir: Path,
    metadata_out: Path,
    tiny_weight: int,
    broad_weight: int,
    unit_chars: int,
    total_chars: int,
    shard_chars: int,
) -> dict[str, object]:
    if tiny_weight <= 0 or broad_weight <= 0:
        raise ValueError("weights must be positive")
    if unit_chars <= 0:
        raise ValueError("--unit-chars must be positive")
    if total_chars <= 0:
        raise ValueError("--total-chars must be positive")

    tiny_paths = train_shards(tiny_dir)
    broad_paths = train_shards(broad_dir)
    tiny = CircularShardReader(tiny_paths)
    broad = CircularShardReader(broad_paths)
    writer = ShardWriter(out_dir, shard_chars)

    cycle_chars = unit_chars * (tiny_weight + broad_weight)
    target_tiny = total_chars * tiny_weight // (tiny_weight + broad_weight)
    target_broad = total_chars - target_tiny
    written_tiny = 0
    written_broad = 0
    cycles = 0

    while written_tiny + written_broad < total_chars:
        tiny_take = min(unit_chars * tiny_weight, target_tiny - written_tiny)
        broad_take = min(unit_chars * broad_weight, target_broad - written_broad)
        if tiny_take > 0:
            writer.write(tiny.read(tiny_take))
            written_tiny += tiny_take
        if broad_take > 0:
            writer.write(broad.read(broad_take))
            written_broad += broad_take
        cycles += 1
        if tiny_take <= 0 and broad_take <= 0:
            break

    writer.flush()
    meta = {
        "generator": "make_mixture_shards.py",
        "tiny_source": shard_summary(tiny_paths),
        "broad_source": shard_summary(broad_paths),
        "out_dir": str(out_dir),
        "metadata_out": str(metadata_out),
        "tiny_weight": tiny_weight,
        "broad_weight": broad_weight,
        "ratio": f"{tiny_weight}:{broad_weight}",
        "unit_chars": unit_chars,
        "cycle_chars": cycle_chars,
        "requested_total_chars": total_chars,
        "written_chars": writer.total_chars,
        "written_tiny_chars": written_tiny,
        "written_broad_chars": written_broad,
        "tiny_fraction": written_tiny / writer.total_chars if writer.total_chars else None,
        "broad_fraction": written_broad / writer.total_chars if writer.total_chars else None,
        "tiny_reader_wraps": tiny.wraps,
        "broad_reader_wraps": broad.wraps,
        "cycles": cycles,
        "sha256": writer.hash.hexdigest(),
        "shards": writer.shards,
        "note": "No normalization; deterministic token-ratio interleaving of source train shards.",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic TinyStories/text8 mixture train shards")
    parser.add_argument("--tiny-dir", type=Path, default=DEFAULT_TINY_DIR)
    parser.add_argument("--broad-dir", type=Path, default=DEFAULT_BROAD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_META)
    parser.add_argument("--tiny-weight", type=int, default=1)
    parser.add_argument("--broad-weight", type=int, default=3)
    parser.add_argument("--unit-chars", type=int, default=8192)
    parser.add_argument("--total-chars", type=int, default=204_800_000)
    parser.add_argument("--shard-chars", type=int, default=10_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = build_mixture(
        args.tiny_dir,
        args.broad_dir,
        args.out_dir,
        args.metadata_out,
        args.tiny_weight,
        args.broad_weight,
        args.unit_chars,
        args.total_chars,
        args.shard_chars,
    )
    print(
        f"wrote {meta['written_chars']:,} chars to {args.out_dir} "
        f"({meta['written_tiny_chars']:,} tiny, {meta['written_broad_chars']:,} broad, "
        f"sha256={meta['sha256']})"
    )
    print(f"wrote {args.metadata_out}")


if __name__ == "__main__":
    main()
