"""Precompute the static seq_idx -> (cell_type, actbin) map for the SEAM
3-panel viewer's activity-bin selector.

The mapping is fixed (it comes from how jointlib900 was stratified: 100 seqs
per actbin x 3 bins x 3 cell types = 900) so the UI reads this CSV at startup
instead of unpickling a pandas DataFrame at runtime (situate's venv has no
pandas; see feedback_ui_no_runtime_compute).

Run with a venv that has pandas (e.g. MoConSwap_mpra/seam_venv):
    seam_venv/bin/python tools/dump_seam_actbins.py
Writes: data/seam/actbins.csv  (columns: seq_idx,cell_type,actbin)
"""
import pickle
from pathlib import Path

PKL = Path("/grid/koo/home/pmantill/projects/Virtual_Experiments/"
           "MoConSwap_mpra/SEAM_jointlib900/libraries/jointlib900_library.pkl")
OUT = Path(__file__).resolve().parent.parent / "data/seam/actbins.csv"

df = pickle.load(open(PKL, "rb"))["df"]
df = df[["seq_idx", "cell_type", "actbin"]].sort_values("seq_idx")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(df)} rows)")
print(df.groupby(["cell_type", "actbin"]).size().to_string())
