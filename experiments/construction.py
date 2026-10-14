"""Index construction: SPIMI vs BSBI at several memory budgets, and the 4-shard parallel speedup."""

import json
import os
import shutil
import statistics
import time
import tracemalloc

from common import CORPUS, FIGS, RESULTS, WORK, fmt_bytes, md_table, save_json, save_md
from lexis.construct import build_bsbi, build_parallel, build_spimi
from lexis.corpus import list_corpus

BUDGETS = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 32_000_000]


def warm_cache():
    for f in list_corpus(CORPUS):
        with open(os.path.join(CORPUS, f), "rb") as fh:
            fh.read()


def run(fn, budget, measure_memory=False):
    out = os.path.join(WORK, "build")
    shutil.rmtree(out, ignore_errors=True)
    if measure_memory:
        tracemalloc.start()
    st, meta = fn(CORPUS, out, budget)
    if measure_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        st["peak_traced"] = peak
    shutil.rmtree(out, ignore_errors=True)
    return st


def budgets_section():
    results = []
    rows = []
    for budget in BUDGETS:
        for name, fn in (("SPIMI", build_spimi), ("BSBI", build_bsbi)):
            times = [run(fn, budget) for _ in range(2)]
            st = min(times, key=lambda s: s["total_s"])
            mem = run(fn, budget, measure_memory=True)
            rec = {k: v for k, v in st.items()}
            rec["peak_traced"] = mem["peak_traced"]
            results.append(rec)
            rows.append([fmt_bytes(budget), name, rec["blocks"], fmt_bytes(rec["run_bytes"]),
                         fmt_bytes(rec["bytes_written"]), fmt_bytes(rec["sim_peak"]),
                         fmt_bytes(rec["peak_traced"]), f"{rec['invert_s']:.2f}", f"{rec['merge_s']:.2f}",
                         f"{rec['total_s']:.2f}"])
            print(rows[-1], flush=True)
    md = md_table(["Memory limit", "Algorithm", "Blocks", "Block bytes", "Total bytes written",
                   "Peak (budget model)", "Peak (tracemalloc)", "Invert s", "Merge s", "Total s"], rows)
    save_md("construction_budgets", md)

    return results


def parallel_section():
    par_rows = []
    par = {}
    for budget in (1_000_000, 4_000_000):
        seq = [run(build_spimi, budget) for _ in range(3)]
        par[budget] = {"sequential": seq}
        med = lambda xs, k: statistics.median(x[k] for x in xs)
        t_seq, i_seq, m_seq = med(seq, "total_s"), med(seq, "invert_s"), med(seq, "merge_s")
        par_rows.append([fmt_bytes(budget), "sequential SPIMI (1 process)", "—", f"{i_seq:.2f}",
                         f"{m_seq:.2f}", f"{t_seq:.2f}", "1.00×", "—", "—"])
        for n in (2, 4):
            pr = [run(lambda c, o, b: build_parallel(c, o, b, n_shards=n), budget) for _ in range(3)]
            par[budget][n] = pr
            t = med(pr, "total_s")
            mp = statistics.median(x["invert_s"] - x["startup_s"] for x in pr)
            rd = med(pr, "reduce_s")
            par_rows.append([fmt_bytes(budget), f"{n} shards (map) + {n} term ranges (reduce)",
                             f"{med(pr, 'startup_s'):.2f}", f"{mp:.2f}", f"{rd:.2f} + {med(pr, 'concat_s'):.2f}",
                             f"{t:.2f}", f"**{t_seq / t:.2f}×**", f"{i_seq / mp:.2f}×", f"{m_seq / rd:.2f}×"])
            print(par_rows[-1], flush=True)
    md2 = md_table(["Memory limit", "Build", "Process start s", "Invert (map) s", "Merge (reduce + concat) s",
                    "Total s", "Speedup (total)", "Speedup map", "Speedup merge"], par_rows)
    save_md("construction_parallel", md2)
    return par


def main():
    import sys
    warm_cache()
    if "--parallel-only" in sys.argv:
        data = json.load(open(os.path.join(RESULTS, "construction.json")))
        data["parallel"] = {str(k): v for k, v in parallel_section().items()}
        save_json("construction", data)
        return
    results = budgets_section()
    par = parallel_section()
    save_json("construction", {"budgets": results, "parallel": {str(k): v for k, v in par.items()}})

    # ------------------------------------------------------------------ figure
    from common import plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for name, mk in (("SPIMI", "o"), ("BSBI", "s")):
        rs = [r for r in results if r["algorithm"] == name]
        b = [r["budget"] / 1e6 for r in rs]
        axes[0].semilogx(b, [r["total_s"] for r in rs], marker=mk, label=name)
        axes[1].semilogx(b, [r["bytes_written"] / 1e6 for r in rs], marker=mk, label=name)
        axes[2].semilogx(b, [r["peak_traced"] / 1e6 for r in rs], marker=mk, label=name)
    axes[0].set(xlabel="memory limit (MB)", ylabel="build time (s)", title="Time")
    axes[1].set(xlabel="memory limit (MB)", ylabel="MB written", title="Bytes written")
    axes[2].set(xlabel="memory limit (MB)", ylabel="MB (tracemalloc)", title="Peak memory")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGS}/construction.png")


if __name__ == "__main__":
    main()
