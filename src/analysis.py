"""Collects every finished run in outputs/ and writes the report material to results/:

    python -m src.analysis

results/
  summary.md                      all tables (mean ± std over seeds) + figure list
  table_main.csv / table_per_class_f1.csv / table_ordinal_errors.csv / table_ablation.csv
  curves_<model>.png, curves_all.png        training/validation loss and macro-F1
  confusion_matrices.png                    row-normalised, pooled over seeds
  error_distance.png, per_class_f1.png
  qualitative_baseline_fails_roberta_ok.md / qualitative_roberta_fails_baseline_ok.md (+ csv)
  tag_enrichment.csv                        how often each phenomenon appears in each group
  explain/                                  attention / occlusion heatmaps for one example
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import STAR_NAMES, aggregate, fmt
from .utils import load_json, log

BASELINE = "BiGRU + attention"


# =========================================================================== loading
def group_name(cfg: dict) -> str:
    if "pooling" in cfg:
        return f"RoBERTa ({cfg['pooling']} pooling, len {cfg['max_len']})"
    return BASELINE


def load_runs(out_dir="outputs") -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for mpath in sorted(Path(out_dir).glob("*/metrics.json")):
        d = mpath.parent
        cfg = load_json(d / "config.json")
        run = {"dir": d, "cfg": cfg, "seed": cfg["seed"], "metrics": load_json(mpath),
               "history": load_json(d / "history.json")}
        groups.setdefault(group_name(cfg), []).append(run)
    order = sorted(groups, key=lambda g: (g != BASELINE, "first" in g, g))
    return {g: sorted(groups[g], key=lambda r: r["seed"]) for g in order}


def main_roberta_group(groups) -> str | None:
    mean = [g for g in groups if g.startswith("RoBERTa (mean")]
    return ("RoBERTa (mean pooling, len 256)" if "RoBERTa (mean pooling, len 256)" in mean
            else (mean[0] if mean else None))


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


# =========================================================================== tables
def _agg(runs, split):
    return aggregate([r["metrics"][split] for r in runs])


def main_table(groups) -> pd.DataFrame:
    rows = []
    for g, runs in groups.items():
        te, va = _agg(runs, "test"), _agg(runs, "val")
        rows.append({
            "Model": g, "Seeds": ", ".join(str(r["seed"]) for r in runs),
            "Test accuracy (%)": fmt(te["accuracy"]), "Test macro-F1 (%)": fmt(te["macro_f1"]),
            "Test MAE (stars)": fmt(te["mae"], pct=False, digits=3),
            "Val macro-F1 (%)": fmt(va["macro_f1"]),
            "Trainable params (M)": f"{np.mean([r['metrics']['params_trainable'] for r in runs]) / 1e6:.1f}",
            "Train time / seed (min)": f"{np.mean([r['metrics']['train_time_min'] for r in runs]):.1f}",
            "Best val epoch": ", ".join(f"{r['metrics']['best_eval']['epoch']:.2f}" for r in runs),
        })
    return pd.DataFrame(rows)


def per_class_table(groups) -> pd.DataFrame:
    rows = []
    for g, runs in groups.items():
        pc = _agg(runs, "test")["per_class_f1"]
        row = {"Model": g}
        for k, name in enumerate(STAR_NAMES):
            row[f"F1 {name} (%)"] = fmt({"mean": pc["mean"][k], "std": pc["std"][k]})
        rows.append(row)
    return pd.DataFrame(rows)


def ordinal_table(groups) -> pd.DataFrame:
    rows = []
    for g, runs in groups.items():
        t = _agg(runs, "test")
        n = runs[0]["metrics"]["test"]["n"]
        rows.append({
            "Model": g,
            "Exact (%)": fmt(t["accuracy"]),
            "Off by 1 star (%)": fmt(t["off_by_one_rate"]),
            "Off by >=2 stars (%)": fmt(t["off_by_two_plus_rate"]),
            "Errors that are adjacent (%)": fmt(t["adjacent_share_of_errors"]),
            "4<->5 confusion rate (%)": fmt(t["rate_4v5_within_4_and_5"]),
            "1<->5 confusions (count / %d)" % n: f"{t['confusions_1v5']['mean']:.1f} ± {t['confusions_1v5']['std']:.1f}",
            "MAE (stars)": fmt(t["mae"], pct=False, digits=3),
        })
    return pd.DataFrame(rows)


def ablation_table(groups) -> pd.DataFrame:
    rows = []
    for g, runs in groups.items():
        if not g.startswith("RoBERTa"):
            continue
        va, te = _agg(runs, "val"), _agg(runs, "test")
        rows.append({"Variant": g, "Seeds": len(runs),
                     "Val macro-F1 (%)": fmt(va["macro_f1"]), "Val acc (%)": fmt(va["accuracy"]),
                     "Val MAE": fmt(va["mae"], pct=False, digits=3),
                     "Test macro-F1 (%)": fmt(te["macro_f1"]), "Test acc (%)": fmt(te["accuracy"]),
                     "Test MAE": fmt(te["mae"], pct=False, digits=3)})
    return pd.DataFrame(rows)


# =========================================================================== figures
def plot_curves(groups, results_dir) -> list[Path]:
    paths = []
    names = list(groups)
    fig_all, axes_all = plt.subplots(len(names), 2, figsize=(12, 3.6 * len(names)), squeeze=False)
    for gi, (g, runs) in enumerate(groups.items()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        for target_axes in (axes, axes_all[gi]):
            for ri, r in enumerate(runs):
                h = pd.DataFrame(r["history"])
                c = f"C{ri}"
                target_axes[0].plot(h["epoch"], h["train_loss"], "--o", color=c, ms=3, label=f"train (seed {r['seed']})")
                target_axes[0].plot(h["epoch"], h["val_loss"], "-o", color=c, ms=3, label=f"val (seed {r['seed']})")
                target_axes[1].plot(h["epoch"], h["train_macro_f1"], "--o", color=c, ms=3, label=f"train (seed {r['seed']})")
                target_axes[1].plot(h["epoch"], h["val_macro_f1"], "-o", color=c, ms=3, label=f"val (seed {r['seed']})")
                be = r["metrics"]["best_eval"]
                target_axes[1].axvline(be["epoch"], color=c, alpha=0.25, lw=4)
            target_axes[0].set(title=f"{g}\ncross-entropy loss", xlabel="epoch", ylabel="loss")
            target_axes[1].set(title=f"{g}\nmacro-F1 (shaded band = best checkpoint)", xlabel="epoch",
                               ylabel="macro-F1")
            for a in target_axes:
                a.grid(alpha=0.3)
                a.legend(fontsize=7)
        fig.tight_layout()
        p = Path(results_dir) / f"curves_{slug(g)}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    fig_all.text(0.5, 0.005, "Train values are averaged over the training batches since the previous "
                 "evaluation (dropout active).", ha="center", fontsize=8)
    fig_all.tight_layout(rect=(0, 0.02, 1, 1))
    p = Path(results_dir) / "curves_all.png"
    fig_all.savefig(p, dpi=150)
    plt.close(fig_all)
    return [p] + paths


def plot_confusions(groups, names, path) -> Path:
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.6), squeeze=False)
    for ax, g in zip(axes[0], names):
        cm = np.asarray(_agg(groups[g], "test")["pooled_confusion_matrix_normalized"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5],
               xlabel="predicted stars", ylabel="true stars",
               title=f"{g}\n(row-normalised, {len(groups[g])} seeds pooled)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_error_distance(groups, names, path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(names)
    for k, g in enumerate(names):
        d = _agg(groups[g], "test")["error_distance_dist"]
        x = np.arange(5) + (k - (len(names) - 1) / 2) * w
        ax.bar(x, np.asarray(d["mean"]) * 100, w, yerr=np.asarray(d["std"]) * 100, label=g, capsize=3)
    ax.set(xticks=range(5), xticklabels=["0 (correct)", "1", "2", "3", "4"], xlabel="|predicted − true| (stars)",
           ylabel="% of test reviews", title="Distance of predictions from the true rating", yscale="log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_per_class_f1(groups, names, path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(names)
    for k, g in enumerate(names):
        pc = _agg(groups[g], "test")["per_class_f1"]
        x = np.arange(5) + (k - (len(names) - 1) / 2) * w
        ax.bar(x, np.asarray(pc["mean"]) * 100, w, yerr=np.asarray(pc["std"]) * 100, label=g, capsize=3)
    ax.set(xticks=range(5), xticklabels=STAR_NAMES, ylabel="F1 (%)", title="Per-class F1 on the test set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


# =========================================================================== qualitative
_NEG = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely|cannot)\b|n't\b")
_CONTRAST = re.compile(r"\b(but|however|although|though|yet|except|unfortunately|otherwise|despite|whereas)\b")
_SARCASM = re.compile(r"\b(yeah right|thanks a lot|thanks for nothing|great job|just great|oh great|so much for|"
                      r"what a joke|love how|nice try|brilliant|genius|wow|lol)\b|!{2,}|\?!")
_POS = {"good", "great", "excellent", "love", "loved", "loves", "perfect", "amazing", "awesome", "nice", "happy",
        "best", "recommend", "wonderful", "fantastic", "beautiful", "sturdy", "comfortable", "pleased",
        "satisfied", "favorite", "solid", "cute", "easy", "works"}
_NEGW = {"bad", "poor", "terrible", "awful", "broke", "broken", "disappointed", "disappointing", "waste", "cheap",
         "return", "returned", "returning", "worst", "useless", "defective", "horrible", "flimsy", "refund",
         "stopped", "junk", "garbage", "hate", "problem", "issue", "issues", "died", "leaked", "leaks"}

TAG_HINTS = {
    "negation": "Negation reverses nearby sentiment words; the BiGRU has to carry 'not' forward to the word it "
                "modifies, while RoBERTa's self-attention lets every token see the negator directly.",
    "contrast": "The review pivots ('but', 'however', ...); the star rating usually follows one side of the pivot, "
                "so the model must weigh the clauses rather than count sentiment words.",
    "mixed sentiment": "Positive and negative words co-occur; middle ratings (2-4 stars) depend on their balance.",
    "sarcasm cue": "Possible sarcasm / irony (heuristic flag - confirm by reading): positive surface words with a "
                   "negative meaning.",
    "polarity mismatch": "The words' surface polarity disagrees with the rating (e.g. only positive words but 1-2 "
                         "stars), so lexical cues mislead.",
    "long (>150 words)": "Long review: the baseline truncates at 200 tokens and squeezes everything into one "
                         "attention-pooled vector; RoBERTa sees up to 256 sub-words with full self-attention.",
    "OOV-heavy": "Several words are <unk> for the baseline (misspellings, rare words); RoBERTa's byte-level BPE "
                 "never produces unknown tokens, it splits them into sub-words.",
    "very short (<=8 words)": "Very little evidence; a single word decides the prediction.",
}
TAG_PRIORITY = list(TAG_HINTS)


def tag_review(text: str, n_unk: int) -> list[str]:
    t = text.lower()
    words = re.findall(r"[a-z']+", t)
    pos, neg = any(w in _POS for w in words), any(w in _NEGW for w in words)
    tags = []
    if _NEG.search(t):
        tags.append("negation")
    if _CONTRAST.search(t):
        tags.append("contrast")
    if pos and neg:
        tags.append("mixed sentiment")
    if _SARCASM.search(t):
        tags.append("sarcasm cue")
    if len(words) > 150:
        tags.append("long (>150 words)")
    if n_unk >= 2:
        tags.append("OOV-heavy")
    if len(words) <= 8:
        tags.append("very short (<=8 words)")
    return tags


def _select(cands: pd.DataFrame, n: int, fail_col: str) -> pd.DataFrame:
    """Pick n examples covering as many phenomena as possible, preferring large errors."""
    if cands.empty:
        return cands
    cands = cands.assign(_err=(cands[fail_col] - cands["label"]).abs(),
                         _readable=cands["n_words"].between(5, 120))
    cands = cands.sort_values(["_err", "_readable", "n_words"], ascending=[False, False, True])
    chosen: list[int] = []
    while len(chosen) < min(n, len(cands)):
        before = len(chosen)
        for tag in TAG_PRIORITY:
            if len(chosen) >= n:
                break
            pool = cands[cands["tags"].apply(lambda ts: tag in ts) & ~cands["idx"].isin(chosen)]
            if tag != "long (>150 words)":
                pool = pool[pool["_readable"]] if pool["_readable"].any() else pool
            if not pool.empty:
                chosen.append(int(pool.iloc[0]["idx"]))
        if len(chosen) == before:  # no tagged examples left: fill by error size
            rest = cands[~cands["idx"].isin(chosen)]
            chosen.extend(rest["idx"].head(n - len(chosen)).astype(int).tolist())
    return cands.set_index("idx").loc[chosen].reset_index().drop(columns=["_err", "_readable"])


def _example_md(i: int, row, b_conf_col="b_conf", r_conf_col="r_conf") -> str:
    hints = "\n".join(f"  - *{t}*: {TAG_HINTS.get(t, '')}" for t in row["tags"]) or "  - (no heuristic tag fired)"
    text = row["text"] if len(row["text"]) <= 900 else row["text"][:900] + " …"
    return (f"### Example {i}  (id `{row['id']}`)\n\n"
            f"- **True rating:** {row['label'] + 1}★\n"
            f"- **BiGRU+attention:** {row['b_pred'] + 1}★ (p = {row[b_conf_col]:.2f})\n"
            f"- **RoBERTa:** {row['r_pred'] + 1}★ (p = {row[r_conf_col]:.2f})\n"
            f"- **Phenomena (auto-tagged):** {', '.join(row['tags']) or '-'}\n"
            f"- **Starting points for your hypothesis (auto-generated):**\n{hints}\n\n"
            f"> {text}\n")


def qualitative(groups, base_group, rob_group, data_dir, results_dir, n_main=8, n_rev=4) -> dict:
    from .data import load_reviews

    b_runs = {r["seed"]: r for r in groups[base_group]}
    r_runs = {r["seed"]: r for r in groups[rob_group]}
    common = sorted(set(b_runs) & set(r_runs))
    seed = 42 if 42 in common else common[0]
    b = pd.read_csv(b_runs[seed]["dir"] / "test_predictions.csv")
    r = pd.read_csv(r_runs[seed]["dir"] / "test_predictions.csv")
    merged, _ = load_reviews(data_dir)
    test = merged["test"]
    n_unk = np.load(Path(data_dir) / "classic" / "test.npz")["n_unk"]

    df = pd.DataFrame({"idx": np.arange(len(test)), "id": test["id"], "text": test["text"],
                       "label": test["label"], "n_unk": n_unk})
    pcols = [f"p{k}" for k in range(1, 6)]
    df["b_pred"] = b.set_index("idx").loc[df["idx"], "pred"].to_numpy()
    df["r_pred"] = r.set_index("idx").loc[df["idx"], "pred"].to_numpy()
    df["b_conf"] = b.set_index("idx").loc[df["idx"], pcols].to_numpy().max(1)
    df["r_conf"] = r.set_index("idx").loc[df["idx"], pcols].to_numpy().max(1)
    df["n_words"] = df["text"].str.split().str.len()
    df["tags"] = [tag_review(t, u) for t, u in zip(df["text"], df["n_unk"])]
    # lexical polarity vs rating
    lex = df["text"].str.lower().str.findall(r"[a-z']+")
    only_pos = lex.apply(lambda ws: any(w in _POS for w in ws) and not any(w in _NEGW for w in ws))
    only_neg = lex.apply(lambda ws: any(w in _NEGW for w in ws) and not any(w in _POS for w in ws))
    mismatch = (only_pos & (df["label"] <= 1)) | (only_neg & (df["label"] >= 3))
    df.loc[mismatch, "tags"] = df.loc[mismatch, "tags"].apply(lambda ts: ts + ["polarity mismatch"])

    b_ok, r_ok = df["b_pred"] == df["label"], df["r_pred"] == df["label"]
    contingency = pd.DataFrame(
        [[int((b_ok & r_ok).sum()), int((b_ok & ~r_ok).sum())],
         [int((~b_ok & r_ok).sum()), int((~b_ok & ~r_ok).sum())]],
        index=["BiGRU correct", "BiGRU wrong"], columns=["RoBERTa correct", "RoBERTa wrong"])

    main = _select(df[~b_ok & r_ok], n_main, "b_pred")
    rev = _select(df[b_ok & ~r_ok], n_rev, "r_pred")

    groups_mask = {"all test": np.ones(len(df), bool), "BiGRU wrong, RoBERTa right": ~b_ok & r_ok,
                   "RoBERTa wrong, BiGRU right": b_ok & ~r_ok, "both wrong": ~b_ok & ~r_ok}
    enrich = []
    for tag in TAG_PRIORITY:
        has = df["tags"].apply(lambda ts, t=tag: t in ts)
        row = {"Phenomenon": tag}
        for name, m in groups_mask.items():
            row[f"% of {name}"] = f"{100 * has[m].mean():.1f}" if m.sum() else "-"
        enrich.append(row)
    enrich = pd.DataFrame(enrich)

    out = Path(results_dir)
    cols = ["idx", "id", "label", "b_pred", "b_conf", "r_pred", "r_conf", "n_words", "n_unk", "tags", "text"]
    df[~b_ok & r_ok][cols].to_csv(out / "all_baseline_fails_roberta_ok.csv", index=False)
    df[b_ok & ~r_ok][cols].to_csv(out / "all_roberta_fails_baseline_ok.csv", index=False)
    main[cols].to_csv(out / "qualitative_baseline_fails_roberta_ok.csv", index=False)
    rev[cols].to_csv(out / "qualitative_roberta_fails_baseline_ok.csv", index=False)
    enrich.to_csv(out / "tag_enrichment.csv", index=False)
    for name, sel, title in (("qualitative_baseline_fails_roberta_ok.md", main,
                              "BiGRU+attention fails, RoBERTa succeeds"),
                             ("qualitative_roberta_fails_baseline_ok.md", rev,
                              "RoBERTa fails, BiGRU+attention succeeds")):
        body = "\n".join(_example_md(i + 1, row) for i, (_, row) in enumerate(sel.iterrows()))
        (out / name).write_text(f"# {title} (test set, seed {seed})\n\n{body}", encoding="utf-8")
    return {"seed": seed, "contingency": contingency, "main": main, "reverse": rev, "enrichment": enrich}


# =========================================================================== driver
def run_analysis(out_dir="outputs", data_dir="data", results_dir="results", explain=True) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    groups = load_runs(out_dir)
    if not groups:
        raise SystemExit(f"no finished runs found in {out_dir}/ (each needs a metrics.json)")
    log("runs found: " + "; ".join(f"{g}: seeds {[r['seed'] for r in rs]}" for g, rs in groups.items()))
    rob = main_roberta_group(groups)
    names = [g for g in (BASELINE, rob) if g in groups]

    tables = {"main": main_table(groups), "per_class_f1": per_class_table(groups),
              "ordinal_errors": ordinal_table(groups), "ablation": ablation_table(groups)}
    for k, t in tables.items():
        t.to_csv(results_dir / f"table_{k}.csv", index=False)

    figures = plot_curves(groups, results_dir)
    figures.append(plot_confusions(groups, list(groups), results_dir / "confusion_matrices.png"))
    figures.append(plot_error_distance(groups, names, results_dir / "error_distance.png"))
    figures.append(plot_per_class_f1(groups, names, results_dir / "per_class_f1.png"))

    qual, explain_info = None, None
    if BASELINE in groups and rob:
        qual = qualitative(groups, BASELINE, rob, data_dir, results_dir)
        if explain and not qual["main"].empty:
            try:
                from .explain import explain_example
                m = qual["main"]
                pick = m[m["tags"].apply(lambda ts: "negation" in ts or "contrast" in ts) & (m["n_words"] <= 80)]
                row = (pick if not pick.empty else m).iloc[0]
                seed = qual["seed"]
                b_run = next(r for r in groups[BASELINE] if r["seed"] == seed)["dir"]
                r_run = next(r for r in groups[rob] if r["seed"] == seed)["dir"]
                res = explain_example(row["text"], int(row["label"]), b_run, r_run, data_dir,
                                      results_dir / "explain", tag=f"test_{row['id']}")
                explain_info = {"id": row["id"], "text": row["text"], "label": int(row["label"]),
                                "special_share": res["roberta"]["special_token_attention_share"]}
            except Exception as e:  # noqa: BLE001 - explanations are optional here
                log(f"explanation step skipped ({type(e).__name__}: {e})")

    write_summary(results_dir, groups, tables, figures, qual, explain_info)
    log(f"wrote {results_dir / 'summary.md'}")
    return {"groups": groups, "tables": tables, "qualitative": qual, "explain": explain_info}


def write_summary(results_dir, groups, tables, figures, qual, explain_info) -> None:
    parts = ["# Q2 results summary (auto-generated)\n",
             "All numbers: mean ± sample std (ddof=1) over seeds; test set evaluated once per run "
             "with the checkpoint that had the best validation macro-F1.\n",
             "## Main comparison\n", to_markdown(tables["main"]),
             "\n## Per-class F1 (test)\n", to_markdown(tables["per_class_f1"]),
             "\n## Ordinal error profile (test)\n", to_markdown(tables["ordinal_errors"])]
    if len(tables["ablation"]) > 1:
        parts += ["\n## Ablation: mean pooling vs first-token (<s>) pooling\n",
                  "Decide on the **validation** columns; test columns are reported for completeness.\n",
                  to_markdown(tables["ablation"])]
    if qual is not None:
        parts += [f"\n## Where the models disagree (test, seed {qual['seed']})\n",
                  to_markdown(qual["contingency"].reset_index().rename(columns={"index": ""})),
                  "\n### How often each phenomenon occurs (auto-tagged, heuristic)\n",
                  to_markdown(qual["enrichment"]),
                  "\nSelected examples: `qualitative_baseline_fails_roberta_ok.md`, "
                  "`qualitative_roberta_fails_baseline_ok.md` (full lists in the `all_*.csv` files)."]
    if explain_info is not None:
        parts += [f"\n## Token-importance example (test id `{explain_info['id']}`, true "
                  f"{explain_info['label'] + 1}★)\n", f"> {explain_info['text'][:600]}\n",
                  f"RoBERTa's last layer sends {explain_info['special_share']:.0%} of its attention to "
                  "<s>/</s>; see `explain/` for the BiGRU attention, RoBERTa attention and occlusion maps."]
    parts += ["\n## Figures\n"] + [f"- `{Path(p).name}`" for p in figures]
    (Path(results_dir) / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless when run as a script
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--no_explain", action="store_true")
    a = p.parse_args()
    run_analysis(a.out_dir, a.data_dir, a.results_dir, explain=not a.no_explain)


if __name__ == "__main__":
    main()
