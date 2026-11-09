"""Merge results files from latency.py and throughput.py into tables.md and
figures. Each PATH is a results file or a directory of them; a file's run is its
directory's name, so directories from several runs are shown side by side.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import PARTS

REQUESTS = ("no KCoral", "cache hit", "cache miss")

# Okabe-Ito colors, distinguishable with color-vision deficiencies. The part
# colors are light enough for black labels.
PART_COLOR = dict(zip((k for k, _ in PARTS), ("#56B4E9", "#E69F00", "#CC79A7", "#009E73")))
SERIES_COLOR = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")  # per tag set
RUN_STYLE = (("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"))  # (line, marker) per run


def load(paths):
    """Every results file under PATHS, with "run" set and "tags" as a sorted tuple."""
    files = []
    for path in paths:
        files += sorted(glob.glob(os.path.join(path, "*.json"))) if os.path.isdir(path) else [path]
    results = []
    for path in files:
        with open(path) as f:
            result = json.load(f)
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        results.append({**result, "run": run, "tags": tuple(sorted(result["tags"].items()))})
    return results


def ordered(values):
    """Distinct values in the order they first appear."""
    return list(dict.fromkeys(values))


def tag_label(tags):
    return ", ".join(f"{k}={v}" for k, v in tags)


def tag_keys(results):
    return sorted({k for r in results for k, _ in r["tags"]})


def tag_cells(result, keys):
    tags = dict(result["tags"])
    return [tags.get(k, "-") for k in keys]


def size_label(n):
    b = n * 4
    return f"{b >> 20} MiB" if b >= 1 << 20 else f"{b >> 10} KiB"


def cell(s, fmt):
    return f"{s['median']:{fmt}} [{s['p5']:{fmt}}, {s['p95']:{fmt}}]"


def table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def request(result):
    """The kind of latency request: one of REQUESTS."""
    params = result["params"]
    return f"cache {params['cache']}" if params["kcoral"] else "no KCoral"


def latency_records(results):
    """(result, record) for every latency record, ordered by run, tags, size, request."""
    latency = [r for r in results if r["benchmark"] == "latency"]
    runs = ordered(r["run"] for r in latency)
    pairs = [(r, rec) for r in latency for rec in r["records"]]
    return sorted(
        pairs,
        key=lambda p: (
            runs.index(p[0]["run"]),
            p[0]["tags"],
            p[1]["n"],
            REQUESTS.index(request(p[0])),
        ),
    )


def latency_figure(pairs, tags, path):
    """For the KCoral requests with TAGS: one panel per tensor size; in each, one
    stacked bar of median parts (ms) per (run, request)."""
    runs = ordered(r["run"] for r, _ in pairs)
    no_kcoral = {(r["run"], rec["n"]): rec for r, rec in pairs if not r["params"]["kcoral"]}
    by_size = {}  # n -> [(row label, record)]
    for result, record in sorted(pairs, key=lambda p: p[1]["n"]):
        if result["tags"] != tags or not result["params"]["kcoral"]:
            continue
        row = request(result)
        if len(runs) > 1:
            row = f"{result['run']}\n{row}"
        by_size.setdefault(record["n"], []).append((row, record))
    heights = [len(rows) for rows in by_size.values()]
    fig, axes = plt.subplots(
        len(by_size),
        1,
        figsize=(8, 0.55 * sum(heights) + 0.9 * len(by_size)),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    for ax, (n, rows) in zip(axes[:, 0], by_size.items()):
        widest = max(sum(r["timing_ms"][k]["median"] for k, _ in PARTS) for _, r in rows)
        for y, (_, record) in enumerate(rows):
            left = 0.0
            for key, part_label in PARTS:
                width = record["timing_ms"][key]["median"]
                ax.barh(y, width, left=left, height=0.6, color=PART_COLOR[key], label=part_label)
                if width > 0.08 * widest:  # label only segments wide enough to hold one
                    ax.text(
                        left + width / 2, y, f"{width:.3g}", ha="center", va="center", fontsize=8
                    )
                left += width
            ax.text(left, y, f"  {record['e2e_ms']['median']:.1f} ms", va="center")
        ax.set_xlim(0, widest * 1.18)
        ax.set_yticks(range(len(rows)), [row for row, _ in rows])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        # With several runs, the baselines without KCoral are in tables.md.
        base = no_kcoral.get((runs[0], n)) if len(runs) == 1 else None
        base = f", without KCoral {base['e2e_ms']['median']:.2f} ms" if base else ""
        ax.set_title(f"{size_label(n)} per tensor{base}", loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("ms (median of each part; own scale per panel)")
    title = "Where a KCoral request's time goes" + (f", {tag_label(tags)}" if tags else "")
    fig.suptitle(title, x=0.02, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[: len(PARTS)],
        labels[: len(PARTS)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(PARTS),
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def latency_table(pairs):
    keys = tag_keys([r for r, _ in pairs])
    rows = []
    for result, record in pairs:
        timing = record.get("timing_ms")  # None without KCoral
        rows.append(
            [
                result["run"],
                *tag_cells(result, keys),
                size_label(record["n"]),
                "yes" if result["params"]["kcoral"] else "no",
                result["params"]["cache"] or "-",
                cell(record["e2e_ms"], ".1f" if timing else ".2f"),
                *(f"{timing[k]['median']:.1f}" if timing else "-" for k, _ in PARTS),
            ]
        )
    return table(
        "Latency, ms: end-to-end median [p5, p95], and the median of each part",
        ["run", *keys, "tensor", "KCoral", "cache", "end-to-end", *(label for _, label in PARTS)],
        rows,
    )


def throughput_figure(results, path):
    runs = ordered(r["run"] for r in results)
    tag_sets = ordered(r["tags"] for r in results)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    concurrency = set()
    for result in results:
        line, marker = RUN_STYLE[runs.index(result["run"]) % len(RUN_STYLE)]
        color = SERIES_COLOR[tag_sets.index(result["tags"]) % len(SERIES_COLOR)]
        c = [lv["concurrency"] for lv in result["records"]]
        rps = [lv["requests_per_s"] for lv in result["records"]]
        concurrency.update(c)
        name = [result["run"]] if len(runs) > 1 else []
        name = ", ".join([*name, tag_label(result["tags"])] if result["tags"] else name)
        ax.plot(
            c,
            rps,
            color=color,
            linestyle=line,
            marker=marker,
            markersize=6,
            linewidth=2,
            label=name or None,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(concurrency), [str(v) for v in sorted(concurrency)])
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("requests per second")
    ax.set_title("Throughput, 4 KiB job, cache hit", loc="left")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def throughput_table(results):
    keys = tag_keys(results)
    rows = []
    for result in results:
        for lv in result["records"]:
            rows.append(
                [
                    result["run"],
                    *tag_cells(result, keys),
                    result["env"]["server_workers"],
                    lv["concurrency"],
                    f"{lv['requests_per_s']:.2f}",
                    cell(lv["wall_ms"], ".1f"),
                    f"{lv['queue_ms']['median']:.1f}",
                    f"{lv['lease_wait_ms']['median']:.1f}",
                    f"{lv['lease_held_ms']['median']:.1f}",
                ]
            )
    return table(
        "Throughput; ms are median [p5, p95] or medians",
        [
            "run",
            *keys,
            "server workers",
            "clients",
            "req/s",
            "wall ms",
            "queue ms",
            "GPU lease wait ms",
            "GPU lease held ms",
        ],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", metavar="PATH")
    parser.add_argument("-o", "--out", help="output directory (default: the only PATH)")
    args = parser.parse_args()
    if args.out is None and not (len(args.paths) == 1 and os.path.isdir(args.paths[0])):
        parser.error("-o is required unless PATH is a single directory")
    out = args.out or args.paths[0]
    os.makedirs(out, exist_ok=True)

    results = load(args.paths)
    written, tables = [], []
    pairs = latency_records(results)
    if pairs:
        for tags in ordered(r["tags"] for r, _ in pairs if r["params"]["kcoral"]):
            name = "_".join(["latency", *(f"{k}-{v}" for k, v in tags)]) + ".png"
            latency_figure(pairs, tags, os.path.join(out, name))
            written.append(name)
        tables.append(latency_table(pairs))
    throughput = [r for r in results if r["benchmark"] == "throughput"]
    if throughput:
        throughput_figure(throughput, os.path.join(out, "throughput.png"))
        written.append("throughput.png")
        tables.append(throughput_table(throughput))
    text = "\n\n".join(tables) + "\n"
    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {', '.join([*written, 'tables.md'])} to {out}")


if __name__ == "__main__":
    main()
