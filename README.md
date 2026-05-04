# vLLM Embedding Benchmark — Turkish

A throughput and latency benchmark for embedding models served by [vLLM](https://github.com/vllm-project/vllm), using **real Turkish Wikipedia text** as input.

Most public embedding benchmarks use English or synthetic data, neither of which reflects how a multilingual model like `Qwen/Qwen3-Embedding-8B` actually performs on Turkish in production. This harness:

- Streams real articles from [Alaeddin/wikipedia-turkish](https://huggingface.co/datasets/Alaeddin/wikipedia-turkish).
- Buckets them by length (**64 / 256 / 1024 / 4096 / 8192 / 16384** tokens) so you can measure throughput as a function of input length.
- Sweeps client concurrency (**1 → 1000+** in-flight requests) using async HTTP, so the bottleneck is the server, not the client.
- Reports request throughput, input-token throughput, and latency percentiles (p50/p90/p95/p99) per concurrency level.

---

## Files

| File | Purpose |
| --- | --- |
| `serve_vllm.sh` | Reference launch command for the vLLM embedding server. Run on the GPU host. |
| `build_corpus.py` | Streams Turkish Wikipedia, groups paragraphs into articles, buckets by length, writes `corpus.jsonl`. |
| `benchmark.py`   | Async load generator. Sweeps concurrency levels and writes per-request and aggregated results. |
| `plot.py`        | Throughput- and latency-vs-concurrency plots from a summary JSON. |
| `requirements.txt` | Client-side dependencies. The vLLM server has its own. |

---

## Quick start

### 1. Install client dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the Turkish corpus (one time, ~5 min)

```bash
python build_corpus.py --per-bucket 500 --out corpus.jsonl
```

This streams the dataset (no full download), accumulates paragraphs by article title, and writes 500 samples per length bucket. The 16k-token bucket is rare in Wikipedia, so this scans ~400k paragraphs to fill it.

Output:

```
bucket    64: 500 samples
bucket   256: 500 samples
bucket  1024: 500 samples
bucket  4096: 500 samples
bucket  8192: 500 samples
bucket 16384: 500 samples
```

### 3. Launch the vLLM server (on the GPU host)

```bash
./serve_vllm.sh
```

Defaults: `Qwen/Qwen3-Embedding-8B`, port 8000, `max_model_len=16384`, `max_num_seqs=256`, 90% GPU memory. Override with env vars, e.g.:

```bash
MODEL=Qwen/Qwen3-Embedding-8B MAX_NUM_SEQS=64 ./serve_vllm.sh
```

`Qwen3-Embedding-8B` is a decoder-style embedding model — vLLM needs `--task embed`, which the script already passes.

### 4. Run the benchmark

```bash
python benchmark.py \
    --base-url http://localhost:8000 \
    --corpus corpus.jsonl --bucket 256 \
    --concurrency 1 10 100 500 1000
```

The script will:

1. Send a warmup burst (discarded).
2. For each concurrency level, fire enough requests for a stable steady-state measurement.
3. Print a one-line summary per level and write three artifacts to `results/`:
   - `raw_<ts>.jsonl` — every request (latency, status, prompt tokens)
   - `summary_<ts>.json` — per-concurrency aggregates
   - `summary_<ts>.csv` — same, flat for spreadsheets

Example console output:

```
[run] concurrency=100  requests=2000  corpus=corpus.jsonl
  -> 412.7 req/s | 105651.2 tok/s | p50 218.4ms p95 412.0ms | errors 0.0%
```

### 5. Sweep all length buckets

To build a full throughput-vs-input-length picture:

```bash
for b in 64 256 1024 4096 8192 16384; do
  python benchmark.py \
      --base-url http://localhost:8000 \
      --corpus corpus.jsonl --bucket $b \
      --concurrency 1 10 100 500 1000 \
      --out-dir results/bucket_$b
done
```

### 6. Plot

```bash
python plot.py results/bucket_256/summary_*.json
```

Produces a side-by-side throughput/latency PNG next to the summary file.

---

## CLI reference

### `build_corpus.py`

| Flag | Default | Notes |
| --- | --- | --- |
| `--dataset` | `Alaeddin/wikipedia-turkish` | Any HF dataset with paragraph + title columns |
| `--text-field` | `paragraph` | Column with paragraph text |
| `--title-field` | `title` | Column used to group paragraphs into articles |
| `--buckets` | `64 256 1024 4096 8192 16384` | Bucket centers in tokens |
| `--per-bucket` | `500` | Samples per bucket |
| `--max-scan` | `2_000_000` | Paragraph cap before giving up |
| `--out` | `corpus.jsonl` | Output path |

### `benchmark.py`

| Flag | Default | Notes |
| --- | --- | --- |
| `--base-url` | `http://localhost:8000` | vLLM server base URL |
| `--model` | `Qwen/Qwen3-Embedding-8B` | Model id as registered with vLLM |
| `--concurrency` | `1 10 50 100 500 1000` | Concurrency levels to sweep |
| `--corpus` | *(none)* | Path to `corpus.jsonl`. If unset, uses synthetic random tokens. |
| `--bucket` | *(all)* | Restrict corpus sampling to one bucket length |
| `--input-tokens` | `256` | Synthetic-mode only |
| `--requests-per-conc` | `20` | Total requests per phase = `concurrency * this` |
| `--min-requests` | `200` | Floor on requests per phase |
| `--warmup` | `32` | Warmup requests (discarded) |
| `--timeout` | `120.0` | Per-request timeout in seconds |
| `--out-dir` | `results` | Output directory |

---

## Design notes

**Why steady-state, not burst.** Each phase warms up first, then measures wall-clock time for `concurrency × requests_per_conc` requests with a semaphore capping in-flight count. This gives "throughput at concurrency C," not a burst spike.

**Why one bucket per run.** Mixing 64- and 16384-token requests gives an averaged number that doesn't generalize to either. Running each bucket separately lets you build a throughput-vs-input-length curve, which is what tells you how the model behaves on your actual traffic mix.

**Why server-reported token counts.** Tokens are read from the response's `usage.prompt_tokens` field, not a local tokenizer. tiktoken is only used as a length proxy when bucketing. This keeps token-throughput numbers honest even though Qwen uses its own tokenizer.

**Connection-pool sizing.** The httpx pool is sized to `max_concurrency + 32` so the client doesn't queue at the pool layer — otherwise you'd be measuring the client, not the server.

---

## Caveats

- **Long-context buckets stress the KV cache.** At 16k tokens × 256 concurrent sequences, vLLM cache pressure goes up ~64× vs the 256-token case. You'll likely need to drop `MAX_NUM_SEQS` (e.g. 32–64) for the 16k sweep, or you'll see preemptions that look like throughput collapse but are really cache eviction.
- **OS file-descriptor limits.** For `--concurrency 1000+`, run `ulimit -n 8192` first.
- **Tokenizer drift.** tiktoken's cl100k_base is a length proxy — actual Qwen3 tokenization differs by a few percent. Bucket boundaries are approximate.
- **Dataset coverage.** Turkish Wikipedia has plenty of articles up to ~8k tokens but relatively few above 16k. The 16k bucket is filled by truncating longer articles.
