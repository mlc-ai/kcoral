"""Print the README tables from out/mrpw<M>_n<N>.json."""

import glob
import json
import os
import re
import statistics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def cell(s, fmt):
    return f"{s['median']:{fmt}} [{s['min']:{fmt}}, {s['max']:{fmt}}]"


def size(n):
    b = n * 4
    return f"{b >> 20} MiB" if b >= 1 << 20 else f"{b >> 10} KiB"


runs = []
for path in glob.glob(os.path.join(OUT, "mrpw*_n*.json")):
    mrpw, n = map(int, re.findall(r"\d+", os.path.basename(path)))
    with open(path) as f:
        runs.append((-mrpw, n, json.load(f)))
runs.sort(key=lambda r: r[:2])

print("End-to-end wall time per job, ms: median [min, max]\n")
print(
    "| workers | tensor | local | local, new process "
    "| KCoral, cached inputs | KCoral, new inputs | first request |"
)
print("|---|---|---|---|---|---|---|")
for mrpw, n, r in runs:
    e = r["e2e_ms"]
    print(
        f"| {'fresh' if mrpw else 'reused'} | {size(n)} | {cell(e['local'], '.2f')} "
        f"| {cell(e['local_proc'], '.0f')} | {cell(e['kcoral_cached'], '.1f')} "
        f"| {cell(e['kcoral_new'], '.1f')} | {r['first_request']['wall_ms']:.0f} |"
    )

print("\nKCoral request breakdown with cached inputs, ms (medians)\n")
print("| workers | tensor | client build | server elapsed | lease held | rest (HTTP, decode) |")
print("|---|---|---|---|---|---|")
for mrpw, n, r in runs:
    raw = r["raw"]["breakdown_ms"]["kcoral_cached"]
    rest = [
        w - s["client_build_ms"] - s["elapsed_ms"]
        for w, s in zip(r["raw"]["e2e_ms"]["kcoral_cached"], raw)
    ]
    b = r["breakdown_ms"]["kcoral_cached"]
    print(
        f"| {'fresh' if mrpw else 'reused'} | {size(n)} | {b['client_build_ms']['median']:.1f} "
        f"| {b['elapsed_ms']['median']:.1f} | {b['lease_held_ms']['median']:.1f} "
        f"| {statistics.median(rest):.1f} |"
    )

print("\nKernel latency (per-run CUPTI median), us: median [min, max]\n")
print("| workers | tensor | local | KCoral |")
print("|---|---|---|---|")
for mrpw, n, r in runs:
    k = r["kernel_us"]
    print(
        f"| {'fresh' if mrpw else 'reused'} | {size(n)} | {cell(k['local'], '.2f')} "
        f"| {cell(k['kcoral'], '.2f')} |"
    )
