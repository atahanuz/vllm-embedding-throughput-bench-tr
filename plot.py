"""Plot throughput- and latency-vs-concurrency from a summary JSON file."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary_json", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    data = json.loads(args.summary_json.read_text())
    c = [d["concurrency"] for d in data]
    rps = [d["throughput_rps"] for d in data]
    tps = [d["throughput_input_tokens_per_s"] for d in data]
    p50 = [d["latency_ms_p50"] for d in data]
    p95 = [d["latency_ms_p95"] for d in data]
    p99 = [d["latency_ms_p99"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(c, rps, "o-", label="req/s")
    ax1b = ax1.twinx()
    ax1b.plot(c, tps, "s--", color="tab:orange", label="input tok/s")
    ax1.set_xscale("log")
    ax1.set_xlabel("concurrency")
    ax1.set_ylabel("throughput (req/s)")
    ax1b.set_ylabel("throughput (input tokens/s)")
    ax1.set_title("Throughput vs concurrency")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(c, p50, "o-", label="p50")
    ax2.plot(c, p95, "s-", label="p95")
    ax2.plot(c, p99, "^-", label="p99")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("concurrency")
    ax2.set_ylabel("latency (ms)")
    ax2.set_title("Latency vs concurrency")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = args.out or args.summary_json.with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
