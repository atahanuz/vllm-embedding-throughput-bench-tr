"""
Throughput / latency benchmark for a vLLM embedding server.

Sweeps a list of concurrency levels and, for each one:
  1. Sends a warmup burst (results discarded).
  2. Runs a steady-state phase of N requests with at most `concurrency` in flight.
  3. Records per-request latency, HTTP status, and token count.
  4. Aggregates throughput (req/s, tokens/s) and latency percentiles.

Outputs:
  results/raw_<timestamp>.jsonl     per-request records
  results/summary_<timestamp>.json  per-concurrency aggregates
  results/summary_<timestamp>.csv   same, flat for spreadsheets
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import tiktoken


# ---------- workload generation ----------

# Synthetic fallback uses tiktoken cl100k_base purely as a *length proxy* to
# build inputs of a target token count. The server-side tokenizer is the
# model's own; small drift is fine because we report what the server returns
# in `usage.prompt_tokens`.
_ENC = tiktoken.get_encoding("cl100k_base")
_VOCAB = [_ENC.decode([i]) for i in range(1000, 5000)]  # readable-ish slice


def synthetic_text(target_tokens: int, rng: random.Random) -> str:
    pieces = [rng.choice(_VOCAB) for _ in range(target_tokens)]
    return " ".join(pieces)


def load_corpus(path: Path, bucket: int | None) -> list[str]:
    """Load texts from a corpus.jsonl built by build_corpus.py.

    If bucket is given, return only that bucket's samples; otherwise all.
    """
    out: list[str] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if bucket is None or rec["bucket"] == bucket:
                out.append(rec["text"])
    if not out:
        raise RuntimeError(f"no samples in {path} for bucket={bucket}")
    return out


def make_payload(
    model: str,
    target_tokens: int,
    rng: random.Random,
    corpus: list[str] | None,
) -> dict:
    if corpus is not None:
        text = corpus[rng.randrange(len(corpus))]
    else:
        text = synthetic_text(target_tokens, rng)
    return {"model": model, "input": text}


# ---------- per-request execution ----------

@dataclass
class RequestResult:
    concurrency: int
    ok: bool
    status: int
    latency_s: float
    prompt_tokens: int
    error: str | None


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    concurrency: int,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return RequestResult(concurrency, False, r.status_code, dt, 0, r.text[:200])
        data = r.json()
        toks = int(data.get("usage", {}).get("prompt_tokens", 0))
        return RequestResult(concurrency, True, 200, dt, toks, None)
    except Exception as e:
        dt = time.perf_counter() - t0
        return RequestResult(concurrency, False, 0, dt, 0, repr(e)[:200])


async def run_phase(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    n_requests: int,
    concurrency: int,
    target_tokens: int,
    seed: int,
    corpus: Optional[list[str]] = None,
) -> tuple[list[RequestResult], float]:
    """Fire n_requests with at most `concurrency` in flight. Returns (results, wall_time)."""
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)

    async def bound(i: int) -> RequestResult:
        async with sem:
            payload = make_payload(model, target_tokens, rng, corpus)
            return await one_request(client, url, payload, concurrency)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(bound(i) for i in range(n_requests)))
    wall = time.perf_counter() - t0
    return results, wall


# ---------- aggregation ----------

def summarize(results: list[RequestResult], wall_s: float, concurrency: int) -> dict:
    ok = [r for r in results if r.ok]
    lat_ms = np.array([r.latency_s * 1000 for r in ok]) if ok else np.array([0.0])
    tokens = sum(r.prompt_tokens for r in ok)
    n = len(results)
    n_ok = len(ok)
    return {
        "concurrency": concurrency,
        "requests_total": n,
        "requests_ok": n_ok,
        "error_rate": (n - n_ok) / n if n else 0.0,
        "wall_s": round(wall_s, 4),
        "throughput_rps": round(n_ok / wall_s, 2) if wall_s > 0 else 0.0,
        "throughput_input_tokens_per_s": round(tokens / wall_s, 2) if wall_s > 0 else 0.0,
        "latency_ms_p50": round(float(np.percentile(lat_ms, 50)), 2),
        "latency_ms_p90": round(float(np.percentile(lat_ms, 90)), 2),
        "latency_ms_p95": round(float(np.percentile(lat_ms, 95)), 2),
        "latency_ms_p99": round(float(np.percentile(lat_ms, 99)), 2),
        "latency_ms_mean": round(float(lat_ms.mean()), 2),
        "latency_ms_max": round(float(lat_ms.max()), 2),
    }


# ---------- driver ----------

async def main_async(args: argparse.Namespace) -> None:
    url = f"{args.base_url.rstrip('/')}/v1/embeddings"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"raw_{ts}.jsonl"
    summary_json = out_dir / f"summary_{ts}.json"
    summary_csv = out_dir / f"summary_{ts}.csv"

    # Single client across the whole sweep — connection pool is sized to the largest
    # concurrency we'll hit, otherwise httpx will queue at the pool layer and you'll
    # measure the pool, not the server.
    max_conc = max(args.concurrency)
    limits = httpx.Limits(
        max_connections=max_conc + 32,
        max_keepalive_connections=max_conc + 32,
    )
    timeout = httpx.Timeout(args.timeout, connect=10.0)

    corpus: Optional[list[str]] = None
    if args.corpus:
        corpus = load_corpus(Path(args.corpus), args.bucket)
        print(f"[corpus] {len(corpus)} samples loaded from {args.corpus}"
              + (f" (bucket={args.bucket})" if args.bucket else " (all buckets)"))

    def write_summary(rows: list[dict]) -> None:
        """Persist summary JSON + CSV. Overwrites — caller invokes after every phase
        so a Ctrl-C only loses the in-flight phase, never completed ones."""
        if not rows:
            return
        summary_json.write_text(json.dumps(rows, indent=2))
        keys = list(rows[0].keys())
        with summary_csv.open("w") as f:
            f.write(",".join(keys) + "\n")
            for s in rows:
                f.write(",".join(str(s[k]) for k in keys) + "\n")

    summaries: list[dict] = []
    interrupted = False
    with raw_path.open("w") as raw_f:
        async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
            try:
                # One global warmup so the first concurrency level isn't penalized.
                print(f"[warmup] {args.warmup} requests at concurrency 8")
                await run_phase(client, url, args.model, args.warmup, 8,
                                args.input_tokens, seed=0, corpus=corpus)

                for c in args.concurrency:
                    # Scale total requests with concurrency so the steady-state window
                    # is long enough to be statistically meaningful.
                    n = max(args.min_requests, c * args.requests_per_conc)
                    src = f"corpus={Path(args.corpus).name}" if corpus is not None else f"synthetic~{args.input_tokens}tok"
                    print(f"[run] concurrency={c}  requests={n}  {src}")
                    results, wall = await run_phase(
                        client, url, args.model, n, c, args.input_tokens, seed=c, corpus=corpus,
                    )
                    for r in results:
                        raw_f.write(json.dumps(asdict(r)) + "\n")
                    raw_f.flush()  # so a later Ctrl-C can't lose this phase's raw rows
                    s = summarize(results, wall, c)
                    print(
                        f"  -> {s['throughput_rps']} req/s | "
                        f"{s['throughput_input_tokens_per_s']} tok/s | "
                        f"p50 {s['latency_ms_p50']}ms p95 {s['latency_ms_p95']}ms | "
                        f"errors {s['error_rate']:.1%}"
                    )
                    summaries.append(s)
                    write_summary(summaries)  # checkpoint after every phase
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True
                print("\n[interrupted] saving partial results before exit...")
                write_summary(summaries)

    if interrupted:
        print(f"\npartial run — {len(summaries)} of {len(args.concurrency)} concurrency levels completed")
    print(f"\nraw:     {raw_path}")
    print(f"summary: {summary_json}")
    print(f"csv:     {summary_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="vLLM server base URL (default: %(default)s)")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B",
                   help="Model id as registered with vLLM")
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 10, 50, 100, 500, 1000],
                   help="Concurrency levels to sweep")
    p.add_argument("--requests-per-conc", type=int, default=20,
                   help="Steady-state requests per unit of concurrency (total = c * this)")
    p.add_argument("--min-requests", type=int, default=200,
                   help="Floor on requests per phase, so low-c phases still produce stable stats")
    p.add_argument("--warmup", type=int, default=32,
                   help="Warmup requests (discarded)")
    p.add_argument("--input-tokens", type=int, default=256,
                   help="Approx input length per request, in tokens (synthetic mode only)")
    p.add_argument("--corpus", default=None,
                   help="Path to corpus.jsonl (built by build_corpus.py). "
                        "If set, real Turkish text is used instead of synthetic.")
    p.add_argument("--bucket", type=int, default=None,
                   help="If --corpus is set, restrict to one bucket (e.g. 256). "
                        "Default: sample from all buckets in the file.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
