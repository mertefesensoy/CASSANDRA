from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
DEFAULT_OUT = Path(__file__).with_name("corpus") / "tinystories_raw" / "TinyStories-train.head.txt"


def download_with_retries(
    url: str,
    out: Path,
    max_bytes: int,
    attempts: int,
    delay_seconds: float,
    chunk_bytes: int,
) -> dict[str, object]:
    if attempts < 1:
        raise ValueError("--attempts must be positive")
    if max_bytes < 0:
        raise ValueError("--max-bytes must be non-negative")
    if chunk_bytes <= 0:
        raise ValueError("--chunk-bytes must be positive")

    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    target_bytes = max_bytes if max_bytes > 0 else None
    started = time.perf_counter()
    last_error = ""

    for attempt in range(1, attempts + 1):
        existing = part.stat().st_size if part.exists() else 0
        if target_bytes is not None and existing >= target_bytes:
            break

        headers = {}
        if existing > 0:
            if target_bytes is None:
                headers["Range"] = f"bytes={existing}-"
            else:
                headers["Range"] = f"bytes={existing}-{target_bytes - 1}"
        elif target_bytes is not None:
            headers["Range"] = f"bytes=0-{target_bytes - 1}"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                mode = "ab" if existing > 0 and response.status == 206 else "wb"
                if mode == "wb":
                    existing = 0
                with part.open(mode) as handle:
                    while True:
                        if target_bytes is not None:
                            remaining = target_bytes - handle.tell()
                            if remaining <= 0:
                                break
                            chunk = response.read(min(chunk_bytes, remaining))
                        else:
                            chunk = response.read(chunk_bytes)
                        if not chunk:
                            break
                        handle.write(chunk)
            current = part.stat().st_size if part.exists() else 0
            print(f"attempt={attempt} downloaded={current} bytes", flush=True)
            if target_bytes is None or current >= target_bytes:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            print(f"attempt={attempt} failed: {last_error}", flush=True)
            if attempt < attempts:
                time.sleep(delay_seconds)

    if not part.exists():
        raise RuntimeError(f"download failed, no partial file was written: {last_error}")
    final_bytes = part.stat().st_size
    if target_bytes is not None and final_bytes < target_bytes:
        raise RuntimeError(
            f"download incomplete after {attempts} attempts: got {final_bytes}, wanted {target_bytes}; {last_error}"
        )

    if out.exists():
        out.unlink()
    part.replace(out)
    return {
        "url": url,
        "out": str(out),
        "bytes": final_bytes,
        "max_bytes": max_bytes,
        "attempts": attempts,
        "seconds": round(time.perf_counter() - started, 4),
        "partial_or_full": "partial" if max_bytes > 0 else "full",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TinyStories text with retries and optional byte cap")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--max-bytes", type=int, default=50_000_000)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--chunk-bytes", type=int, default=1_048_576)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = download_with_retries(
        args.url,
        args.out,
        args.max_bytes,
        args.attempts,
        args.delay_seconds,
        args.chunk_bytes,
    )
    metadata_out = args.metadata_out or args.out.with_suffix(args.out.suffix + ".download.json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({report['bytes']} bytes)")
    print(f"wrote {metadata_out}")


if __name__ == "__main__":
    main()
