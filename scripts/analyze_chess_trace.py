"""Print an op-level device-time breakdown of a jax.profiler trace, straight
from the trace file -- no TensorBoard, no reading a multi-MB JSON by hand.

`jax.profiler` writes a gzipped Chrome Trace Format JSON under
`<trace-dir>/plugins/profile/<run>/*.trace.json.gz`; this script loads the
newest one, keeps the `ph:"X"` duration events on the device "XLA Ops"
threads (resolving `pid`/`tid` to device/thread names via the `ph:"M"`
metadata events), and sums `dur` per op name, reported as a top-N
individual-op list.

Typical use: `just prof-scalar-analyze` (traces ONE vmapped `Chess.step`
call, then runs this), or standalone on any existing trace:

    uv run python scripts/analyze_chess_trace.py /tmp/pgx1/scalar --steps 20
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os

# Substring -> label for named pipeline parts (empty now that the Pallas
# kernels are gone). Anything unmatched lands in "other XLA ops".
BUCKETS = []
OTHER = "other XLA ops"


def find_trace(trace_dir: str) -> str:
    paths = glob.glob(os.path.join(trace_dir, "plugins", "profile", "*", "*.trace.json.gz"))
    if not paths:
        raise SystemExit(
            f"no *.trace.json.gz under {trace_dir}/plugins/profile/ -- "
            "run `just prof-scalar` first"
        )
    return max(paths, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", nargs="?", default="/tmp/pgx1/scalar")
    parser.add_argument(
        "--steps", type=int, default=1,
        help="how many steps the trace covers; totals are divided by this to report ms/step",
    )
    parser.add_argument("--top", type=int, default=20, help="how many individual ops to list")
    args = parser.parse_args()

    path = find_trace(args.trace_dir)
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]

    proc: dict = {}
    thread: dict = {}
    for e in events:
        if e.get("ph") == "M":
            if e["name"] == "process_name":
                proc[e["pid"]] = e["args"]["name"]
            elif e["name"] == "thread_name":
                thread[(e["pid"], e.get("tid"))] = e["args"]["name"]

    per_device: dict = collections.defaultdict(collections.Counter)
    for e in events:
        if e.get("ph") != "X":
            continue
        pname = proc.get(e.get("pid"), "")
        if pname.startswith("/device:") and thread.get((e.get("pid"), e.get("tid"))) == "XLA Ops":
            per_device[pname][e["name"]] += e.get("dur", 0)

    if not per_device:
        raise SystemExit(f"no device 'XLA Ops' duration events found in {path}")

    # Multi-device traces mirror the same timeline onto every device when the
    # computation isn't sharded; report just the busiest one.
    device = max(per_device, key=lambda d: sum(per_device[d].values()))
    ops = per_device[device]
    total = sum(ops.values())

    print(f"trace:  {path}")
    print(f"device: {device} (busiest of {sorted(per_device)})")
    print(f"total device time: {total / 1e3:.3f} ms over {args.steps} step(s)"
          f" = {total / 1e3 / args.steps:.3f} ms/step")

    buckets = collections.Counter()
    for name, dur in ops.items():
        for substr, label in BUCKETS:
            if substr in name:
                buckets[label] += dur
                break
        else:
            buckets[OTHER] += dur

    width = max(len(label) for label in buckets)
    print("\nmajor parts:")
    for label, dur in buckets.most_common():
        print(f"  {label:<{width}}  {dur / 1e3 / args.steps:9.3f} ms/step  {100 * dur / total:5.1f}%")

    print(f"\ntop {args.top} individual ops:")
    for name, dur in ops.most_common(args.top):
        print(f"  {dur / 1e3 / args.steps:9.3f} ms/step  {100 * dur / total:5.1f}%  {name}")


if __name__ == "__main__":
    main()
