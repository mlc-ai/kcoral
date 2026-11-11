"""Build the REPORT.md figures and tables from the JSON results in out/.

One function per report section. Figures go to figures/ (committed with the
report); tables are printed and written to out/tables.md for copying into REPORT.md.
"""

import datetime
import glob
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGURES = os.path.join(HERE, "figures")
SIZES = (1024, 1048576, 16777216)  # float32 elements per tensor
MODES = ("reused", "fresh")

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"local": "#8a8984", "reused": "#2a78d6", "fresh": "#eb6834"}
LABEL = {"local": "local", "reused": "KCoral, reused workers", "fresh": "KCoral, fresh workers"}
# Breakdown parts, in request order; categorical slots 3-6 (validated adjacent set).
PARTS = (
    ("client_build", "client: build program + hash", "#1baf7a"),
    ("worker_handoff", "server: worker handoff", "#eda100"),
    ("lease_held", "server: GPU lease held", "#e87ba4"),
    ("rest", "HTTP + request handling", "#008300"),
)
DARK_PART = "#008300"  # the one part color dark enough for light text

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)
TABLES = []


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def overhead(mode, n):
    return load(f"overhead_{mode}_n{n}.json")


def size_label(n):
    b = n * 4
    return f"{b >> 20} MiB" if b >= 1 << 20 else f"{b >> 10} KiB"


def cell(s, fmt):
    return f"{s['median']:{fmt}} [{s['min']:{fmt}}, {s['max']:{fmt}}]"


def table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    TABLES.append("\n".join(lines))
    print(TABLES[-1] + "\n")


def save(fig, name):
    fig.savefig(os.path.join(FIGURES, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def breakdown(run, inputs="kcoral_cached"):
    """Median of each part of a request (cached or new inputs), from per-request samples."""
    raw = run["raw"]["breakdown_ms"][inputs]
    walls = run["raw"]["e2e_ms"][inputs]
    parts = {
        "client_build": [s["client_build_ms"] for s in raw],
        "worker_handoff": [s["elapsed_ms"] - s["lease_held_ms"] for s in raw],
        "lease_held": [s["lease_held_ms"] for s in raw],
        "rest": [w - s["client_build_ms"] - s["elapsed_ms"] for w, s in zip(walls, raw)],
    }
    return {k: statistics.median(v) for k, v in parts.items()}


def breakdown_figure(rows, title, xlabel, name, sharex=False):
    """One stacked bar per (label, run, inputs) row in the PARTS colors, each part
    labeled with its median ms where it is wide enough.

    Each row has its own scale unless sharex.
    """
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 1.4 * len(rows)), sharex=sharex)
    segments = []  # (ax, left, width, color), labeled once the scales are final
    for ax, (label, run, inputs) in zip(axes, rows):
        left = 0.0
        parts = breakdown(run, inputs)
        for key, part_label, color in PARTS:
            ax.barh(
                0,
                parts[key],
                left=left,
                height=0.6,
                color=color,
                label=part_label,
                edgecolor=SURFACE,
                linewidth=2,
            )
            segments.append((ax, left, parts[key], color))
            left += parts[key]
        total = run["e2e_ms"][inputs]["median"]
        ax.text(left * 1.01, 0, f"{total:.1f} ms", va="center", color=INK)
        if not sharex:
            ax.set_xlim(0, left * 1.15)
        ax.set_yticks([0], [label])
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
    axes[0].set_title(title, loc="left")
    axes[-1].set_xlabel(xlabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
    fig.tight_layout()
    renderer = fig.canvas.get_renderer()
    for ax, left, width, color in segments:
        # Largest font that fits inside the segment, or no label at all.
        to_px = ax.transData.transform
        room = to_px((left + width, 0))[0] - to_px((left, 0))[0] - 6
        for fontsize in (9, 8, 7):
            text = ax.text(
                left + width / 2,
                0,
                f"{width:.0f} ms" if width >= 10 else f"{width:.1f} ms",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=SURFACE if color == DARK_PART else INK,
            )
            if text.get_window_extent(renderer).width <= room:
                break
            text.remove()
    save(fig, name)


def q1_fixed_cost():
    n = SIZES[0]
    runs = {mode: overhead(mode, n) for mode in MODES}
    e2e = runs["fresh"]["e2e_ms"]
    rows = [
        ["local", cell(e2e["local"], ".2f")],
        *[[LABEL[m], cell(runs[m]["e2e_ms"]["kcoral_cached"], ".1f")] for m in MODES],
    ]
    table(
        "Q1: end-to-end time of the 4 KiB job (50 runs)",
        ["mode", "end-to-end ms: median [min, max]"],
        rows,
    )

    # Own scale per row: fresh is ~20x reused, which would hide its parts.
    breakdown_figure(
        [(LABEL[mode], runs[mode], "kcoral_cached") for mode in ("fresh", "reused")],
        "Where a 4 KiB request's time goes (median of each part; own scale per row)",
        f"ms   (the same job run locally in-process: {e2e['local']['median']:.2f} ms)",
        "q1_breakdown.png",
    )


def q2_size_scaling():
    # Reused workers only: tensor size adds about the same time with fresh workers.
    runs = {n: overhead("reused", n) for n in SIZES[1:]}  # 4 KiB is Q1's reused row
    breakdown_figure(
        [
            (
                f"{size_label(n)}\n(local {run['e2e_ms']['local']['median']:.1f} ms)",
                run,
                "kcoral_cached",
            )
            for n, run in runs.items()
        ],
        "Where a reused-worker request's time goes (median of each part; own scale per row)",
        "ms",
        "q2_breakdown.png",
    )

    rows = []
    for n in SIZES:
        for mode in MODES:
            run = overhead(mode, n)
            parts = breakdown(run)
            rows.append(
                [
                    size_label(n),
                    mode,
                    *(f"{parts[k]:.1f}" for k, _, _ in PARTS),
                    f"{run['e2e_ms']['kcoral_cached']['median']:.1f}",
                    f"{run['e2e_ms']['local']['median']:.2f}",
                ]
            )
    table(
        "Q1/Q2: breakdown of a cached-input request, ms (medians)",
        ["tensor", "workers", *(label for _, label, _ in PARTS), "KCoral total", "local"],
        rows,
    )


def q3_new_inputs():
    # Reused workers only: new inputs add about the same time with fresh workers.
    run = overhead("reused", SIZES[-1])
    breakdown_figure(
        [("cached inputs", run, "kcoral_cached"), ("new inputs", run, "kcoral_new")],
        f"Cached vs new inputs, {size_label(SIZES[-1])} per tensor, reused workers "
        "(median of each part)",
        "ms",
        "q3_new_inputs.png",
        sharex=True,
    )

    rows = []
    for n in SIZES:
        e = overhead("reused", n)["e2e_ms"]
        cached, new = e["kcoral_cached"]["median"], e["kcoral_new"]["median"]
        rows.append([size_label(n), f"{cached:.1f}", f"{new:.1f}", f"{new - cached:+.1f}"])
    table(
        "Q3: cached vs new inputs, reused workers, end-to-end ms (medians)",
        ["tensor", "cached inputs", "new inputs", "extra"],
        rows,
    )

    rows = []
    for n in SIZES:
        for mode in MODES:
            run = overhead(mode, n)
            e = run["e2e_ms"]
            extra = e["kcoral_new"]["median"] - e["kcoral_cached"]["median"]
            rows.append(
                [
                    size_label(n),
                    mode,
                    cell(e["kcoral_cached"], ".1f"),
                    cell(e["kcoral_new"], ".1f"),
                    f"{extra:+.1f}",
                    f"{run['first_request']['wall_ms']:.0f}",
                ]
            )
    table(
        "Q3: cached vs new inputs, ms: median [min, max] of 50",
        [
            "tensor",
            "workers",
            "cached inputs",
            "new inputs",
            "extra (median)",
            "first request after start",
        ],
        rows,
    )


def worker_restart(name):
    """Median seconds from a worker's retirement until it is ready again, from the
    event log of server NAME."""
    (path,) = glob.glob(os.path.join(OUT, "logs", name, "runs", "*", "events.jsonl"))
    with open(path) as f:
        events = [json.loads(line) for line in f]
    retired, restart = {}, []
    for e in events:
        t = datetime.datetime.fromisoformat(e["timestamp"]).timestamp()
        if e["event"] == "worker_retired":
            retired[e["worker_id"]] = t
        elif e["event"] == "worker_ready" and e["worker_id"] in retired:
            restart.append(t - retired.pop(e["worker_id"]))
    return statistics.median(restart)


def q4_throughput():
    runs = {mode: load(f"throughput_{mode}.json") for mode in MODES}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    rows = []
    for mode in MODES:
        levels = runs[mode]["levels"]
        c = [lv["concurrency"] for lv in levels]
        rps = [lv["requests_per_s"] for lv in levels]
        ax.plot(
            c, rps, color=SERIES[mode], marker="o", markersize=6, linewidth=2, label=LABEL[mode]
        )
        ax.annotate(
            f"{rps[-1]:.1f} req/s",
            (c[-1], rps[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
        # At the highest concurrency the GPU is never idle for lack of requests,
        # so the time between requests is what each request costs the GPU.
        per_request = 1e3 / rps[-1]
        held = levels[-1]["lease_held_ms"]["median"]
        rows.append(
            [
                mode,
                f"{rps[-1]:.2f}",
                f"{per_request:.1f}",
                f"{held:.1f}",
                f"{per_request - held:.1f}",
            ]
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(c, [str(v) for v in c])
    ax.set_xlabel("concurrent clients (8 workers per GPU)")
    ax.set_ylabel("requests per second")
    ax.set_title("Throughput vs clients", loc="left")
    ax.legend(loc="center right")
    save(fig, "q4_throughput.png")
    table(
        f"Q4: time per request at C = {c[-1]}, 8 workers per GPU (medians)",
        ["workers", "req/s", "ms per request (1000 / req/s)", "GPU lease held ms", "rest ms"],
        rows,
    )


def discussion_workers():
    rows = []
    for workers in (1, 2, 4, 8, 16):
        level = load(f"workers_fresh_w{workers}.json")["levels"][0]
        restart = worker_restart(f"workers_{workers}")
        rows.append([workers, f"{level['requests_per_s']:.2f}", f"{restart:.1f}"])
    table(
        f"Discussion: fresh workers, {level['concurrency']} clients, by workers per GPU (medians)",
        ["workers per GPU", "req/s", "worker restart s (retired to ready again)"],
        rows,
    )


def discussion_pipe_cost():
    pipe = load("pipe_cost.json")
    rows = []
    for n in SIZES:
        p = pipe[str(n)]
        handoff = {mode: breakdown(overhead(mode, n))["worker_handoff"] for mode in MODES}
        rows.append(
            [
                size_label(n),
                f"{p['inputs_ms']['median']:.1f}",
                f"{p['round_trip_ms']['median']:.1f}",
                f"{handoff['fresh']:.1f}",
                f"{handoff['reused']:.1f}",
            ]
        )
    table(
        "Discussion: bare pipe vs KCoral worker handoff, ms (medians)",
        [
            "tensor",
            "pipe: inputs",
            "pipe: inputs + result",
            "handoff, fresh (inputs)",
            "handoff, reused (inputs + result)",
        ],
        rows,
    )


def main():
    os.makedirs(FIGURES, exist_ok=True)
    q1_fixed_cost()
    q2_size_scaling()
    q3_new_inputs()
    q4_throughput()
    discussion_workers()
    discussion_pipe_cost()
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n\n".join(TABLES) + "\n")


if __name__ == "__main__":
    main()
