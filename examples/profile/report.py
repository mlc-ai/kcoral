"""Build the REPORT.md figures and tables from the JSON results in out/.

One function per report section. Figures go to figures/ (committed with the
report); tables are printed and written to out/tables.md for copying into REPORT.md.
"""

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
    ("server_outside_lease", "server: outside GPU lease", "#eda100"),
    ("lease_held", "server: GPU lease held", "#e87ba4"),
    ("rest", "HTTP transfer + decode", "#008300"),
)

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


def breakdown(run):
    """Median of each part of a cached-input request, from per-request samples."""
    raw = run["raw"]["breakdown_ms"]["kcoral_cached"]
    walls = run["raw"]["e2e_ms"]["kcoral_cached"]
    parts = {
        "client_build": [s["client_build_ms"] for s in raw],
        "server_outside_lease": [s["elapsed_ms"] - s["lease_held_ms"] for s in raw],
        "lease_held": [s["lease_held_ms"] for s in raw],
        "rest": [w - s["client_build_ms"] - s["elapsed_ms"] for w, s in zip(walls, raw)],
    }
    return {k: statistics.median(v) for k, v in parts.items()}


def q1_kernel_fidelity():
    rows = []
    for n in SIZES:
        for mode in MODES:
            k = overhead(mode, n)["kernel_us"]
            diff = k["kcoral"]["median"] - k["local"]["median"]
            rows.append(
                [
                    size_label(n),
                    mode,
                    cell(k["local"], ".2f"),
                    cell(k["kcoral"], ".2f"),
                    f"{diff:+.2f} ({100 * diff / k['local']['median']:+.2f}%)",
                ]
            )
    table(
        "Q1: kernel time (per-run CUPTI median), µs: median [min, max] of 30 runs",
        ["tensor", "workers", "local", "KCoral", "KCoral - local"],
        rows,
    )


def q2_fixed_cost():
    n = SIZES[0]
    runs = {mode: overhead(mode, n) for mode in MODES}
    e2e = runs["fresh"]["e2e_ms"]
    rows = [
        ["local, in-process", cell(e2e["local"], ".2f")],
        ["local, new Python process", cell(e2e["local_proc"], ".0f")],
        *[[LABEL[m], cell(runs[m]["e2e_ms"]["kcoral_cached"], ".1f")] for m in MODES],
    ]
    table(
        "Q2: end-to-end time of the 4 KiB job, ms: median [min, max] of 50",
        ["mode", "time"],
        rows,
    )

    # One panel per mode with its own scale: fresh is ~20x reused, which would hide its parts.
    fig, axes = plt.subplots(2, 1, figsize=(8, 2.8))
    for ax, mode in zip(axes, ("fresh", "reused")):
        left = 0.0
        parts = breakdown(runs[mode])
        for key, label, color in PARTS:
            ax.barh(
                0,
                parts[key],
                left=left,
                height=0.6,
                color=color,
                label=label,
                edgecolor=SURFACE,
                linewidth=2,
            )
            left += parts[key]
        total = runs[mode]["e2e_ms"]["kcoral_cached"]["median"]
        ax.text(left * 1.01, 0, f"{total:.1f} ms", va="center", color=INK)
        ax.set_xlim(0, left * 1.15)
        ax.set_yticks([0], [LABEL[mode]])
        ax.grid(axis="y", visible=False)
    axes[0].set_title(
        "Where a 4 KiB request's time goes (median of each part; own scale per row)", loc="left"
    )
    axes[1].set_xlabel(
        f"ms   (the same job run locally in-process: {e2e['local']['median']:.2f} ms)"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
    fig.tight_layout()
    save(fig, "q2_breakdown.png")


def q3_size_scaling():
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [n * 4 for n in SIZES]
    series = {
        "local": ("fresh", "local"),
        "reused": ("reused", "kcoral_cached"),
        "fresh": ("fresh", "kcoral_cached"),
    }
    for name, (mode, key) in series.items():
        stats = [overhead(mode, n)["e2e_ms"][key] for n in SIZES]
        med = [s["median"] for s in stats]
        err = [
            [m - s["min"] for m, s in zip(med, stats)],
            [s["max"] - m for m, s in zip(med, stats)],
        ]
        ax.errorbar(
            x,
            med,
            yerr=err,
            color=SERIES[name],
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            label=LABEL[name],
        )
        nudge = {"fresh": 6, "reused": -6}.get(name, 0)  # keep the two KCoral labels apart
        ax.annotate(
            f"{med[-1]:.0f} ms",
            (x[-1], med[-1]),
            xytext=(8, nudge),
            textcoords="offset points",
            va="center",
            color=INK,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [size_label(n) for n in SIZES])
    ax.set_xlabel("size of each input tensor (two inputs, one output of the same size)")
    ax.set_ylabel("end-to-end ms (median, min-max bars)")
    ax.set_title("End-to-end time vs tensor size (cached inputs)", loc="left")
    ax.legend(loc="lower right")
    save(fig, "q3_size.png")

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
        "Q2/Q3: breakdown of a cached-input request, ms (medians)",
        ["tensor", "workers", *(label for _, label, _ in PARTS), "KCoral total", "local"],
        rows,
    )


def q4_new_inputs():
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
        "Q4: cached vs new inputs, ms: median [min, max] of 50",
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


def q5_throughput():
    runs = {mode: load(f"throughput_{mode}.json") for mode in MODES}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"wspace": 0.45})
    for mode in MODES:
        levels = runs[mode]["levels"]
        c = [lv["concurrency"] for lv in levels]
        left.plot(
            c,
            [lv["requests_per_s"] for lv in levels],
            color=SERIES[mode],
            marker="o",
            markersize=6,
            linewidth=2,
            label=LABEL[mode],
        )
        left.annotate(
            f"{levels[-1]['requests_per_s']:.1f} req/s",
            (c[-1], levels[-1]["requests_per_s"]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
        med = [lv["wall_ms"]["median"] for lv in levels]
        right.plot(
            c, med, color=SERIES[mode], marker="o", markersize=6, linewidth=2, label=LABEL[mode]
        )
        right.fill_between(
            c,
            [lv["wall_ms"]["p5"] for lv in levels],
            [lv["wall_ms"]["p95"] for lv in levels],
            color=SERIES[mode],
            alpha=0.15,
            linewidth=0,
        )
    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(c, [str(v) for v in c])
        ax.set_xlabel("concurrent clients (8 workers)")
    left.set_ylabel("requests per second")
    left.set_title("Throughput", loc="left")
    right.set_ylabel("end-to-end ms (median, p5-p95 band)")
    right.set_title("Latency per request", loc="left")
    left.legend(loc="center left")
    save(fig, "q5_throughput.png")

    rows = []
    for mode in MODES:
        for lv in runs[mode]["levels"]:
            rows.append(
                [
                    mode,
                    lv["concurrency"],
                    f"{lv['requests_per_s']:.2f}",
                    cell(lv["wall_ms"], ".0f"),
                    f"{lv['queue_ms']['median']:.0f}",
                    f"{lv['lease_wait_ms']['median']:.0f}",
                    lv["wall_ms"]["n"],
                ]
            )
    table(
        "Q5: 4 KiB requests with cached inputs, sent back to back by C clients for 25 s",
        [
            "workers",
            "C",
            "req/s",
            "end-to-end ms: median [min, max]",
            "queue ms (median)",
            "lease wait ms (median)",
            "requests",
        ],
        rows,
    )


def discussion_exit_time():
    data = load("exit_time.json")
    rows = [
        [
            mode,
            f"{d['exit_ms']['median']:.0f}",
            f"{d['exit_ms']['min']:.0f}-{d['exit_ms']['max']:.0f}",
        ]
        for mode, d in data.items()
    ]
    ctx = data["ctx_destroy"]["ctx_destroy_ms"]["median"]
    table(
        f"Discussion: worker process exit time, ms (CUDA context destroy itself: {ctx:.0f} ms)",
        ["how the process ends", "median", "range"],
        rows,
    )


def discussion_reuse_state():
    steps = load("reuse_state.json")
    keys = (
        "pid",
        "gpu_free_mib",
        "torch_allocated_mib",
        "tensor_on_torch_module",
        "tensor_add_patched",
        "cudnn_benchmark",
        "env_var",
    )
    rows = []
    for step in steps:
        r = step["result"] or {}
        rows.append(
            [
                step["step"],
                step["status"],
                *(r.get(k, "—") for k in keys),
                step["error"]["kind"] if step["error"] else "",
            ]
        )
    table(
        "Discussion: state seen by successive requests on one reused worker",
        ["request", "status", *keys, "error kind"],
        rows,
    )


def main():
    os.makedirs(FIGURES, exist_ok=True)
    q1_kernel_fidelity()
    q2_fixed_cost()
    q3_size_scaling()
    q4_new_inputs()
    q5_throughput()
    discussion_exit_time()
    discussion_reuse_state()
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n\n".join(TABLES) + "\n")


if __name__ == "__main__":
    main()
