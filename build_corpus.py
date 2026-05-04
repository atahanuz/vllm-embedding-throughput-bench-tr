"""
Build a length-stratified Turkish corpus from Alaeddin/wikipedia-turkish for
embedding-throughput benchmarking.

The dataset is *paragraph-level* (one row = one paragraph, with a `title`
column). Rows are streamed in article order, so we accumulate paragraphs while
the title stays the same and flush a full article when the title changes.

Each finished article is tokenized with tiktoken cl100k_base as a length proxy
and assigned to a bucket. For each bucket we collect up to `--per-bucket`
samples, slicing each one to roughly the bucket's center length so requests
within a bucket are homogeneous in size.

Output: corpus.jsonl with one JSON record per line:
  {"bucket": 256, "approx_tokens": 251, "title": "...", "text": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken
from datasets import load_dataset


# Bucket centers in tokens. A sample falls into the bucket whose [low, high)
# range contains its token count; we then truncate to roughly the center.
# Top end goes to 16k so we can stress long-context embedding throughput,
# matching the typical max_model_len you'd configure on the vLLM side.
DEFAULT_BUCKETS = [64, 256, 1024, 4096, 8192, 16384]


def bucket_ranges(buckets: list[int]) -> list[tuple[int, int, int]]:
    """Return (center, low, high) for each bucket. Bands are geometric-ish.

    For the top bucket we accept arbitrarily long articles and let the slicer
    truncate them to `center` — Turkish Wikipedia has very few articles above
    ~32k tokens, so a strict upper band would starve the largest bucket.
    """
    ranges = []
    for i, c in enumerate(buckets):
        low = c // 2 if i > 0 else 1
        if i + 1 < len(buckets):
            high = (c + buckets[i + 1]) // 2
        else:
            high = 10**9
        ranges.append((c, low, high))
    return ranges


def assign_to_bucket(
    text: str,
    title: str,
    enc: "tiktoken.Encoding",
    ranges: list[tuple[int, int, int]],
    collected: dict[int, list[dict]],
    per_bucket: int,
) -> None:
    toks = enc.encode(text)
    n = len(toks)
    for center, low, high in ranges:
        if low <= n < high and len(collected[center]) < per_bucket:
            sliced = enc.decode(toks[:center])
            collected[center].append({
                "bucket": center,
                "approx_tokens": min(n, center),
                "title": title,
                "text": sliced,
            })
            return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="Alaeddin/wikipedia-turkish")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="paragraph",
                   help="Name of the paragraph-text column in the dataset")
    p.add_argument("--title-field", default="title",
                   help="Column used to group paragraphs into articles")
    p.add_argument("--buckets", type=int, nargs="+", default=DEFAULT_BUCKETS,
                   help="Bucket center lengths in tokens")
    p.add_argument("--per-bucket", type=int, default=500,
                   help="How many samples to keep per bucket")
    p.add_argument("--max-scan", type=int, default=2_000_000,
                   help="Cap on dataset rows (paragraphs) to scan before giving up. "
                        "Long-context buckets (8k/16k) are rare, default is generous.")
    p.add_argument("--out", type=Path, default=Path("corpus.jsonl"))
    args = p.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    ranges = bucket_ranges(sorted(args.buckets))
    collected: dict[int, list[dict]] = {c: [] for c, _, _ in ranges}

    print(f"streaming {args.dataset}:{args.split}", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    cur_title: str | None = None
    cur_paragraphs: list[str] = []
    scanned = 0
    articles = 0

    def flush() -> None:
        nonlocal articles
        if not cur_paragraphs:
            return
        articles += 1
        text = "\n\n".join(cur_paragraphs)
        assign_to_bucket(text, cur_title or "", enc, ranges, collected, args.per_bucket)

    def all_full() -> bool:
        return all(len(collected[c]) >= args.per_bucket for c, _, _ in ranges)

    for row in ds:
        scanned += 1
        if scanned > args.max_scan:
            break
        title = row.get(args.title_field)
        para = row.get(args.text_field)
        if not isinstance(para, str) or not para:
            continue

        if title != cur_title:
            flush()
            cur_title = title
            cur_paragraphs = []
        cur_paragraphs.append(para)

        if scanned % 25_000 == 0:
            counts = " ".join(f"{c}:{len(collected[c])}" for c, _, _ in ranges)
            print(f"  scanned={scanned} articles={articles}  {counts}", file=sys.stderr)

        if all_full():
            break

    flush()  # last article

    with args.out.open("w") as f:
        for c, _, _ in ranges:
            for rec in collected[c]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\ndone.", file=sys.stderr)
    print(f"  scanned {scanned} paragraphs across {articles} articles", file=sys.stderr)
    for c, _, _ in ranges:
        print(f"  bucket {c:>5}: {len(collected[c])} samples", file=sys.stderr)
    print(f"  wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
