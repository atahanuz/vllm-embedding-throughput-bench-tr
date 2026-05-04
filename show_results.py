"""
Pretty-print a benchmark summary JSON as a fixed-width table.

Usage:
    python show_results.py                                  # latest summary in results/
    python show_results.py results/summary_20260504_*.json  # specific file
    python show_results.py results/                         # latest under a dir
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


COLS = [
    ("concurrency",                  "concurrency",   "{:>11}"),
    ("requests",                     "requests",      "{:>10}"),
    ("throughput_rps",               "req/s",         "{:>9.2f}"),
    ("throughput_input_tokens_per_s","tok/s",         "{:>10.1f}"),
    ("latency_ms_p50",               "p50 ms",        "{:>9.1f}"),
    ("latency_ms_p95",               "p95 ms",        "{:>9.1f}"),
    ("latency_ms_p99",               "p99 ms",        "{:>9.1f}"),
    ("error_rate",                   "errors",        "{:>7.1%}"),
]


def find_latest(path: Path) -> Path:
    """Resolve `path` to a concrete summary file (auto-pick latest under a dir)."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.rglob("summary_*.json"))
        if not candidates:
            sys.exit(f"no summary_*.json under {path}")
        return candidates[-1]
    # treat as glob
    matches = sorted(glob.glob(str(path)))
    if not matches:
        sys.exit(f"no match for {path}")
    return Path(matches[-1])


def render(rows: list[dict], src: Path) -> None:
    headers = [h for _, h, _ in COLS]
    # header
    header_fmts = [fmt.replace("d", "s").replace("f", "s").replace(".1%", "s")
                   .replace(".2", "").replace(".1", "") for _, _, fmt in COLS]
    # easier: build widths from format specs
    sep = "  "
    widths = [int(fmt.split(":>")[1].split("}")[0].split(".")[0]) for _, _, fmt in COLS]
    head = sep.join(f"{h:>{w}}" for h, w in zip(headers, widths))
    rule = sep.join("-" * w for w in widths)
    print()
    print(f"  source: {src}")
    print(f"  rows:   {len(rows)}")
    print()
    print(head)
    print(rule)

    best_rps_idx = max(range(len(rows)), key=lambda i: rows[i]["throughput_rps"])
    for i, r in enumerate(rows):
        cells = []
        for key, _, fmt in COLS:
            if key == "requests":
                cells.append(f"{r['requests_ok']}/{r['requests_total']}".rjust(10))
                continue
            v = r.get(key, 0)
            cells.append(fmt.format(v))
        line = sep.join(cells)
        if i == best_rps_idx:
            line += "   <- peak req/s"
        print(line)
    print()

    # callouts
    peak = rows[best_rps_idx]
    print(f"  peak throughput: {peak['throughput_rps']:.1f} req/s "
          f"({peak['throughput_input_tokens_per_s']:.0f} tok/s) "
          f"at concurrency {peak['concurrency']}")
    print(f"  best latency:    p50 {min(r['latency_ms_p50'] for r in rows):.1f} ms "
          f"(at concurrency {min(rows, key=lambda r: r['latency_ms_p50'])['concurrency']})")

    # saturation hint: did throughput peak before max concurrency?
    if best_rps_idx < len(rows) - 1:
        next_r = rows[best_rps_idx + 1]
        drop = (peak["throughput_rps"] - next_r["throughput_rps"]) / peak["throughput_rps"]
        if drop > 0.05:
            print(f"  saturation:      throughput drops {drop:.0%} from concurrency "
                  f"{peak['concurrency']} -> {next_r['concurrency']} — server is past its "
                  f"sweet spot; latency grows without throughput gain")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="results",
                   help="summary JSON file, glob, or directory (default: results/)")
    args = p.parse_args()

    src = find_latest(Path(args.path))
    rows = json.loads(src.read_text())
    if not isinstance(rows, list) or not rows:
        sys.exit(f"{src} is not a non-empty list of summaries")
    render(rows, src)


if __name__ == "__main__":
    main()
