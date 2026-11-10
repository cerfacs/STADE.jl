#!/usr/bin/env python3
"""Summarize battle-test TSV results (rows from validate_battle_status.jl / validate_battle_random.jl).
usage: battle_summarize.py OUT.md FILE.tsv [FILE.tsv ...]"""
import sys, collections, csv

def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            r = csv.DictReader(f, delimiter="\t")
            rows += list(r)
    return rows

def fam(s):
    s = s.split("|")[0]
    return "F11" if s.startswith("rnd_") else "F" + s[1] if s[0] == "f" and s[1].isdigit() else "?"

def obs_class(o, backend):
    ks = set(x.split("=")[0] for x in o.split(","))
    if ks == {"K"} or ks == {"R"}: return "K" if ks == {"K"} else "R"
    if len(ks) == 1: return ks.pop()
    return "mixed"

def main():
    out, paths = sys.argv[1], sys.argv[2:]
    rows = load(paths)
    st = [r for r in rows if r["verdict"] in ("ok", "false_offload", "missed_offload", "extra_offload", "other", "absent")]
    other = [r for r in rows if r not in st]
    L = []
    L.append("## Verdict counts by family\n")
    c = collections.Counter((fam(r["subject"]), r["verdict"]) for r in st)
    fams = sorted(set(f for f, _ in c))
    vs = ["ok", "false_offload", "missed_offload", "extra_offload", "other", "absent"]
    L.append("| Family | " + " | ".join(vs) + " |"); L.append("|---|" + "---|" * len(vs))
    for f in fams: L.append(f"| {f} | " + " | ".join(str(c[(f, v)]) for v in vs) + " |")
    L.append("\n## Confusion matrix (all rows): expected status against observed status\n")
    m = collections.Counter((r["expected"], obs_class(r["observed"], r["backend"])) for r in st)
    obsk = sorted(set(o for _, o in m)); expk = sorted(set(e for e, _ in m))
    L.append("| expected \\ observed | " + " | ".join(obsk) + " |"); L.append("|---|" + "---|" * len(obsk))
    for e in expk: L.append(f"| {e} | " + " | ".join(str(m[(e, o)]) for o in obsk) + " |")
    L.append("\n## Non-ok rows grouped (subject, loop, mode, expected, observed, verdict)\n")
    g = collections.defaultdict(set)
    for r in st:
        if r["verdict"] != "ok":
            g[(r["subject"], r["loop"], r["mode"], r["expected"], r["observed"], r["verdict"])].add((r["backend"], r["fuse"]))
    L.append("| Subject | Loop | Mode | Expected | Observed | Verdict | Backend, fuse |"); L.append("|---|---|---|---|---|---|---|")
    for k in sorted(g): L.append("| " + " | ".join(k) + " | " + "; ".join(f"{b}/{f}" for b, f in sorted(g[k])) + " |")
    L.append("\n## Other rows (generation errors, oracle or class mismatches)\n")
    g2 = collections.Counter((r["subject"], r["mode"], r["verdict"], r["observed"][:90]) for r in other)
    L.append("| Subject | Mode | Kind | Text | Rows |"); L.append("|---|---|---|---|---|")
    for k, n in sorted(g2.items()): L.append("| " + " | ".join(k) + f" | {n} |")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out, len(rows), "rows")
main()
