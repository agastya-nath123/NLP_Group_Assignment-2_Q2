"""Re-draws the report figures at print size from the saved Colab run files
(outputs/*/history.json and outputs/*/metrics.json). No retraining involved."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Run from the report/ folder placed inside the project root (next to outputs/ and results/).
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "outputs"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
MODELS = [("bigru_attn", "BiGRU + attention"), ("roberta_mean_len256", "RoBERTa (mean pooling)"),
          ("roberta_first_len256", "RoBERTa (first-token pooling)")]
SEEDS = [42, 13]
COL = {42: "tab:blue", 13: "tab:orange"}

# ---------------------------------------------------------------- training curves
fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.1))
for r, (key, name) in enumerate(MODELS):
    for s in SEEDS:
        h = json.load(open(RUNS / f"{key}_seed{s}" / "history.json"))
        m = json.load(open(RUNS / f"{key}_seed{s}" / "metrics.json"))
        ep = [e["epoch"] for e in h]
        axes[r, 0].plot(ep, [e["train_loss"] for e in h], "--", marker="o", ms=2.5, lw=1, color=COL[s], label=f"train, seed {s}")
        axes[r, 0].plot(ep, [e["val_loss"] for e in h], "-", marker="o", ms=2.5, lw=1.3, color=COL[s], label=f"val, seed {s}")
        axes[r, 1].plot(ep, [100 * e["train_macro_f1"] for e in h], "--", marker="o", ms=2.5, lw=1, color=COL[s], label=f"train, seed {s}")
        axes[r, 1].plot(ep, [100 * e["val_macro_f1"] for e in h], "-", marker="o", ms=2.5, lw=1.3, color=COL[s], label=f"val, seed {s}")
        axes[r, 1].axvline(m["best_eval"]["epoch"], color=COL[s], alpha=0.25, lw=3)
    axes[r, 0].set(title=f"{name}: loss", ylabel="cross-entropy loss")
    axes[r, 1].set(title=f"{name}: macro-F1", ylabel="macro-F1 (%)")
    for a in axes[r]:
        a.grid(alpha=0.3); a.set_xlabel("epoch")
    axes[r, 0].legend(ncol=2, loc="upper right", frameon=False)
fig.tight_layout(h_pad=0.8)
fig.savefig(OUT / "fig_curves.pdf"); plt.close(fig)

# ---------------------------------------------------------------- confusion matrices
def pooled(key):
    cm = sum(np.asarray(json.load(open(RUNS / f"{key}_seed{s}" / "metrics.json"))["test"]["confusion_matrix"]) for s in SEEDS)
    return cm / cm.sum(1, keepdims=True)
cb, cr, cf = pooled("bigru_attn"), pooled("roberta_mean_len256"), pooled("roberta_first_len256")
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15))
for ax, cm, title in ((axes[0], cb, "BiGRU + attention"), (axes[1], cr, "RoBERTa (mean)"),
                      (axes[2], cf, "RoBERTa (first token)")):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_title(title, fontsize=7.5)
d = 100 * (cr - cb)
ax = axes[3]
ax.imshow(d, cmap="RdBu_r", norm=TwoSlopeNorm(0, -6, 6))
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{d[i, j]:+.1f}", ha="center", va="center", fontsize=5.6,
                color="white" if abs(d[i, j]) > 4 else "black")
ax.set_title("RoBERTa (mean) − BiGRU, % pts", fontsize=7.5)
for k, ax in enumerate(axes):
    ax.set(xticks=range(5), yticks=range(5), xticklabels=range(1, 6), yticklabels=range(1, 6), xlabel="predicted stars")
    ax.tick_params(labelsize=6.5)
    if k == 0:
        ax.set_ylabel("true stars")
fig.tight_layout(w_pad=0.4)
fig.savefig(OUT / "fig_confusion.pdf"); plt.close(fig)

# ---------------------------------------------------------------- per-class F1 and error distance
ALL = [("bigru_attn", "BiGRU + attention"), ("roberta_mean_len256", "RoBERTa (mean)"),
       ("roberta_first_len256", "RoBERTa (first token)")]
def stat(key, field):
    a = np.asarray([json.load(open(RUNS / f"{key}_seed{s}" / "metrics.json"))["test"][field] for s in SEEDS], float)
    return a.mean(0), a.std(0, ddof=1)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
w = 0.26
for k, (key, name) in enumerate(ALL):
    m, sd = stat(key, "per_class_f1")
    axes[0].bar(np.arange(5) + (k - 1) * w, 100 * m, w, yerr=100 * sd, capsize=1.5, label=name)
    m, sd = stat(key, "error_distance_dist")
    axes[1].bar(np.arange(5) + (k - 1) * w, 100 * m, w, yerr=100 * sd, capsize=1.5, label=name)
axes[0].set(xticks=range(5), xticklabels=[f"{i} star" + ("s" if i > 1 else "") for i in range(1, 6)],
            ylabel="test F1 (%)", ylim=(45, 85), title="Per-class F1")
axes[1].set(xticks=range(5), xticklabels=["0 (correct)", "1", "2", "3", "4"], yscale="log",
            xlabel="|predicted − true| (stars)", ylabel="% of test reviews (log)",
            title="Distance of predictions from the true rating")
for a in axes:
    a.grid(alpha=0.3, axis="y")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "fig_bars.pdf"); plt.close(fig)
print("figures written to", OUT)
