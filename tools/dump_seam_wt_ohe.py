"""Precompute per-SEAM WT one-hot for the SEAM viewer's hyp/obs toggle.

The SEAM .npy maps are the var200 variable region = full 230 bp insert
sliced [15:215] (200 bp; 15 bp adapter each side frozen). Observed
contribution = hypothetical_map * onehot(WT_var_region). situate's venv
has no pandas, so we precompute the WT one-hot here (see
feedback_ui_no_runtime_compute) into a static .npz the app loads at start.

Channel order A,C,G,T and shape (200,4) match the SEAM maps and
render_logo_png exactly; N/non-ACGT -> all-zero row (mirrors the Koo OHE
path in app.py).

Run with a venv that has pandas (e.g. MoConSwap_mpra/seam_venv):
    seam_venv/bin/python tools/dump_seam_wt_ohe.py
Writes: data/seam/wt_ohe.npz  (ohe (900,200,4) int8, seq_idx (900,) int32)
"""
import pickle
from pathlib import Path

import numpy as np

PKL = Path("/grid/koo/home/pmantill/projects/Virtual_Experiments/"
           "MoConSwap_mpra/SEAM_jointlib900/libraries/jointlib900_library.pkl")
OUT = Path(__file__).resolve().parent.parent / "data/seam/wt_ohe.npz"

VAR_START, VAR_END = 15, 215           # must match SEAM_attr_standardtorch.py
SEAM_L = VAR_END - VAR_START           # 200
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}

df = pickle.load(open(PKL, "rb"))["df"].sort_values("seq_idx")
seq_idx = df["seq_idx"].to_numpy(np.int32)
ohe = np.zeros((len(df), SEAM_L, 4), np.int8)

for r, (_, row) in enumerate(df.iterrows()):
    s = row["sequence"]
    if len(s) != 230:
        raise SystemExit(f"seq_idx {row['seq_idx']} insert len {len(s)} != 230")
    for j, ch in enumerate(s[VAR_START:VAR_END].upper()):
        ci = BASE.get(ch, -1)
        if ci >= 0:
            ohe[r, j, ci] = 1

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, ohe=ohe, seq_idx=seq_idx)
print(f"wrote {OUT}  ohe{ohe.shape} seq_idx[{seq_idx.min()}..{seq_idx.max()}]"
      f" n={len(seq_idx)}")
# Sanity: every position is exactly one base (no N expected in var region).
per_pos = ohe.sum(axis=2)
print(f"positions with !=1 base: {(per_pos != 1).sum()} (expect 0 if no Ns)")
