"""Reads each word's cell colour from the token-importance heatmaps in results/explain/
so the same colours can be shown as compact coloured text in the LaTeX report."""
import json
import numpy as np
from PIL import Image
from pathlib import Path
HERE = Path(__file__).resolve().parent
EX = str(HERE.parent / "results" / "explain" / "test_en_0600154_")

def grid_box(a):
    dark = (a[:, :, :3] < 110).all(-1)
    h, w = dark.shape
    rows = np.where(dark.sum(1) > 0.5 * w)[0]                 # top / bottom frame lines
    top, bot = rows.min(), rows.max()
    cols = np.where(dark[top:bot].sum(0) > 0.8 * (bot - top))[0]   # vertical frame lines
    groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1)    # heatmap L, heatmap R, colorbar L, R
    return top + 2, bot - 2, groups[0].max() + 1, groups[1].min() - 1

def cell_colours(path, n_words, per_row=10):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    top, bot, left, right = grid_box(a)
    n_rows = int(np.ceil(n_words / per_row))
    cw, ch = (right - left) / per_row, (bot - top) / n_rows
    out = []
    for k in range(n_words):
        r, c = divmod(k, per_row)
        x0, y0 = int(left + c * cw + 0.06 * cw), int(top + r * ch + 0.10 * ch)
        patch = a[y0:y0 + int(0.18 * ch), x0:x0 + int(0.12 * cw)].reshape(-1, 3)
        out.append([int(v) for v in np.median(patch, 0)])
    return out, (top, bot, left, right)

BIGRU = "i do n’t have cravings after taking this at night ! i ’m horrible with eating late at night , i do it almost every night after dinner and i ’m sure it ’s part of the reason i ’m overweight . but when i take this , the cravings are gone ! !".split()
ROB = "I don ’ t have cravings after taking this at night ! I ’ m horrible with eating late at night , I do it almost every night after dinner and I ’ m sure it ’ s part of the reason I ’ m overweight . But when I take this , the cravings are gone !!".split()
res = {}
for key, words in (("bigru_attention", BIGRU), ("roberta_attention", ROB), ("roberta_occlusion", ROB)):
    cols, box = cell_colours(EX + key + ".png", len(words))
    res[key] = list(zip(words, cols))
    print(key, len(words), "grid", box)
    print("  ", [(w, c) for w, c in res[key] if max(c) - min(c) > 60][:12])
json.dump(res, open(HERE / "explain_colours.json", "w"), ensure_ascii=False, indent=0)
