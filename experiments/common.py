import json

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


DATA = os.path.join(ROOT, "lexis_dataset")
CORPUS = os.path.join(DATA, "corpus")
INDEX = os.path.join(ROOT, "index")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "report", "figures")
WORK = os.path.join(ROOT, "_work")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)



def __getattr__(name):
    """Lazy `plt`, so worker processes spawned on Windows (which re-import the
    main script) do not pay for importing matplotlib."""
    if name == "plt":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": .3,
                             "axes.spines.top": False, "axes.spines.right": False})
        globals()["plt"] = plt
        return plt
    raise AttributeError(name)


def save_json(name, obj):
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def md_table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save_md(name, text):
    with open(os.path.join(RESULTS, name + ".md"), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    print(text)


def ensure_index():
    from lexis.construct import build_spimi
    if not os.path.exists(os.path.join(INDEX, "dict.bin")):
        build_spimi(CORPUS, INDEX, 4_000_000)


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.2f} {unit}"
        n /= 1024
