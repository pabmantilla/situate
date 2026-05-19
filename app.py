import argparse, csv, io, threading
from pathlib import Path
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from fast_logo import fast_logo
import matplotlib.pyplot as plt

PX = 5          # pixels per sequence position (must match frontend)
H = 140         # logo pixel height (must match frontend)
DPI = 100
_LOGO_CACHE: dict[tuple, bytes] = {}

HERE = Path(__file__).parent

# SEAM 3-panel viewer. Tree layout: SEAM_ROOT/{ct}/{seq_idx}/{map}.npy
# ct in {HepG2,K562,WTC11}; seq_idx an int dir name; each .npy is
# (200,4) float (hypothetical, mean-centered DeepLIFT). A background job
# is still populating this; most dirs are currently empty -> every
# access path below must tolerate missing files / an empty tree.
SEAM_ROOT = Path(
    "/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/"
    "SEAM_jointlib900/results/foregrounds")
SEAM_CTS = ["HepG2", "K562", "WTC11"]
SEAM_MAPS = ["foreground_scaled", "average_background",
             "average_background_scaled", "wt_attribution",
             "ref_cluster_avg"]
SEAM_L = 200
_SEAM_LOGO_CACHE: dict[tuple, bytes] = {}

# Precomputed static activity-bin map for SEAM jointlib900 seqs.
# data/seam/actbins.csv has columns seq_idx,cell_type,actbin (900 rows,
# fixed). Parsed with stdlib csv (no pandas), keyed by int seq_idx ->
# {"actbin","cell_type"}. Tolerate the file being absent.
SEAM_ACTBINS = ["activating", "neutral", "repressing"]
SEAM_ACTBIN: dict[int, dict] = {}
_actbins_csv = HERE / "data" / "seam" / "actbins.csv"
if _actbins_csv.exists():
    with open(_actbins_csv) as fh:
        for row in csv.DictReader(fh):
            try:
                _si = int(row["seq_idx"])
            except (TypeError, ValueError, KeyError):
                continue
            SEAM_ACTBIN[_si] = {
                "actbin": (row.get("actbin", "") or "").strip(),
                "cell_type": (row.get("cell_type", "") or "").strip(),
            }

# Per-seq_idx WT one-hot (200,4 A,C,G,T) for the SEAM hyp/obs toggle.
# Precomputed by tools/dump_seam_wt_ohe.py; missing file -> obs disabled.
SEAM_WT_OHE: dict[int, "np.ndarray"] = {}
_wt_ohe_npz = HERE / "data" / "seam" / "wt_ohe.npz"
if _wt_ohe_npz.exists():
    try:
        _z = np.load(_wt_ohe_npz)
        for _i, _si in enumerate(_z["seq_idx"]):
            SEAM_WT_OHE[int(_si)] = _z["ohe"][_i].astype(np.float32)
    except Exception:
        SEAM_WT_OHE = {}


# Koo Hippo-MPRA deeplift attributions are the only built-in defaults.
# Each h5 holds `attr` (N,4,L); rows are aligned 1:1 to manifest.csv
# (the 56975-row library with the 5 raw NA rows already dropped), so
# attr row i  <->  manifest row i. The raw joint_library_combined.csv
# (56980 rows) is reference only and is NOT indexed by attr row.
CELL_LINES = ["HepG2", "K562", "WTC11"]
DEFAULT_CL = "HepG2"
# Koo-lab attribution methods. "intgrad" = integrated gradients, 100
# baseline shuffles. Files are {cl}_{method}.h5, same layout as deeplift.
METHODS = ["deeplift", "intgrad"]
DEFAULT_METHOD = "deeplift"
ATTR_DIR = HERE / "data" / "attributions"
MANIFEST = HERE / "data" / "library" / "manifest.csv"


def attr_path(method, cl):
    return ATTR_DIR / f"{cl}_{method}.h5"


def dkey(method, cl):
    return f"{method}:{cl}"


def load_arr(p):
    p = str(p)
    if p.endswith(".h5") or p.endswith(".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            if "attr" in f:
                a = f["attr"][:]
            else:
                a = next(f[k][:] for k in f if f[k].ndim == 3)
        return np.asarray(a, dtype=np.float32)
    a = np.load(p)
    if hasattr(a, "files"):
        a = a["arr_0"]
    return np.asarray(a, dtype=np.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default=None)
    ap.add_argument("--ohe", default=None)
    ap.add_argument("--annot", default=None)
    ap.add_argument("--names", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seam-root", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()


args = parse_args()

if args.seam_root:
    SEAM_ROOT = Path(args.seam_root)

# Two on-disk SEAM trees, picked from the menu via the `src` query param:
# "orig" = the original foregrounds (contaminated/invalid adapter run,
# usually empty), "var200" = the corrected [15:215] rerun. var200 is the
# default since it's the only valid attribution data.
SEAM_SRCS = {
    "orig": SEAM_ROOT,
    "var200": SEAM_ROOT.parent / "foregrounds_var200",
}
SEAM_DEFAULT_SRC = "var200"


def seam_root_for(src):
    return SEAM_SRCS.get(src, SEAM_SRCS[SEAM_DEFAULT_SRC])

# Dataset registry. With no --attr the 3 Koo cell lines are the only
# selectable datasets; --attr overrides to a single "custom" dataset.
if args.attr:
    METHODS = ["custom"]
    DEFAULT_METHOD = "custom"
    CELL_LINES = ["custom"]
    DEFAULT_CL = "custom"
    DATASETS = {dkey("custom", "custom"): args.attr}
else:
    DATASETS = {dkey(m, cl): str(attr_path(m, cl))
                for m in METHODS for cl in CELL_LINES}
    missing = [k for k, p in DATASETS.items() if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing attribution h5 for: {missing}")


def resolve(method, cl):
    """Clamp a (method, cl) request to a valid registered dataset."""
    if dkey(method, cl) in DATASETS:
        return method, cl
    return DEFAULT_METHOD, DEFAULT_CL


_ATTR: dict[str, np.ndarray] = {}


def get_attr(cl, method=DEFAULT_METHOD):
    k = dkey(method, cl)
    a = _ATTR.get(k)
    if a is None:
        a = load_arr(DATASETS[k])
        if a.shape != (N, 4, L):
            raise SystemExit(
                f"{k} attr shape {a.shape} != expected ({N},4,{L})")
        _ATTR[k] = a
    return a


_d0 = load_arr(DATASETS[dkey(DEFAULT_METHOD, DEFAULT_CL)])
N, _, L = _d0.shape
_ATTR[dkey(DEFAULT_METHOD, DEFAULT_CL)] = _d0

OUT = args.out or "edits.csv"

# Default startup seed when no --annot is passed: the HepG2_manual
# foregrounds-only annotation set (SEAM foreground_scaled spans only;
# Koo int-keyed rows intentionally excluded). The running OUT cache
# still wins over this if edits.csv exists (see seed block below).
DEFAULT_ANNOT = HERE / "data" / "annotations" / "HepG2_manual" / \
    "foregrounds_seed.csv"
ANNOT = args.annot or (str(DEFAULT_ANNOT) if DEFAULT_ANNOT.exists()
                        else None)

# manifest.csv is row-aligned to attr (NA rows already dropped). Read it
# once for names + sequences. --names overrides names; --attr custom uses
# neither manifest names nor a one-hot.
ADAPTER = 15  # 15bp adapter each side of the 230bp insert; attr is [15:215]
NAMES = None
OHE = None
ACT = None       # measured activity per cell line: {cl: (N,) float32 log2FC}
if args.attr is None and MANIFEST.exists():
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N:
        raise SystemExit(
            f"manifest rows {len(rows)} != attr rows {N} "
            f"(NA-row alignment broken)")
    NAMES = [r.get("name", "") for r in rows]
    # {cl}_log2FC is the measured MPRA activity (manifest is row-aligned
    # to attr). Blank / non-numeric -> NaN, surfaced to the UI as None.
    ACT = {}
    for _cl in CELL_LINES:
        _col = f"{_cl}_log2FC"
        _vals = []
        for r in rows:
            try:
                _vals.append(float(r.get(_col, "")))
            except (TypeError, ValueError):
                _vals.append(float("nan"))
        ACT[_cl] = np.asarray(_vals, np.float32)
    seqs = [r.get("sequence", "") for r in rows]
    if any(len(s) != 230 for s in seqs):
        raise SystemExit("manifest sequence not 230bp; cannot crop to attr")
    # One-hot the 230bp insert, slice the inner 200bp [15:215] to align
    # with attr's 15bp adapter trim. Channel order A,C,G,T (matches attr).
    codes = np.frombuffer("".join(seqs).encode(), np.uint8).reshape(N, 230)
    lut = np.full(256, -1, np.int64)
    for i, b in enumerate(b"ACGT"):
        lut[b] = i
        lut[b + 32] = i  # lowercase
    ci = lut[codes]
    oh = np.eye(4, dtype=np.float32)[np.where(ci >= 0, ci, 0)]  # (N,230,4)
    oh[ci < 0] = 0.0  # N / non-ACGT -> no observed letter
    OHE = np.ascontiguousarray(
        oh.transpose(0, 2, 1)[:, :, ADAPTER:ADAPTER + L])  # (N,4,L)
    if OHE.shape != (N, 4, L):
        raise SystemExit(f"derived OHE {OHE.shape} != ({N},4,{L})")
if args.names and Path(args.names).exists():
    with open(args.names) as fh:
        NAMES = [ln.rstrip("\n") for ln in fh]
HAS_OHE = OHE is not None
HAS_NAMES = bool(NAMES) and any(s.strip() for s in NAMES)

_PRED: dict[str, np.ndarray] = {}


def get_pred(cl, method=DEFAULT_METHOD):
    k = dkey(method, cl)
    p = _PRED.get(k)
    if p is None:
        try:
            import h5py
            with h5py.File(DATASETS[k], "r") as f:
                p = (f["predictions"][:] if "predictions" in f
                     else np.zeros(N, np.float32))
        except Exception:
            p = np.zeros(N, np.float32)
        p = np.asarray(p, np.float32)
        if p.shape[0] != N:
            p = np.zeros(N, np.float32)
        _PRED[k] = p
    return _PRED[k]


def seq_name(idx):
    if NAMES is not None and 0 <= idx < len(NAMES):
        return NAMES[idx]
    return ""


def seq_act(idx, cl):
    a = ACT.get(cl) if ACT else None
    if a is None or not (0 <= idx < len(a)):
        return None
    v = float(a[idx])
    return None if v != v else round(v, 4)   # NaN -> None


state = {}            # universal seq idx -> [ {motif_name,start,end,strand} ]
# SEAM panels are keyed by the string "SEAM:{ct}:{seq}:{map}" and kept
# separate from `state` (whose write_out sorts by int and would break on
# string keys). Both are persisted into the same edits.csv.
seam_state: dict[str, list] = {}


def load_into_state(path):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            sn = row.get("sequence_name", "")
            span = {
                "motif_name": row.get("motif_name", ""),
                "start": int(float(row["start"])),
                "end": int(float(row["end"])),
                "strand": row.get("strand", "+") or "+",
            }
            try:
                idx = int(sn)
            except (TypeError, ValueError):
                if sn.startswith("SEAM:"):
                    seam_state.setdefault(sn, []).append(span)
                continue
            state.setdefault(idx, []).append(span)


# Continuous cache: OUT is the running annotation .csv, keyed by universal
# seq idx, so it survives paging / cell-line / dataset changes AND server
# restarts. Seed from --annot first, then the existing OUT (OUT wins).
if ANNOT and Path(ANNOT).exists():
    load_into_state(ANNOT)
if Path(OUT).exists() and (not ANNOT or Path(OUT) != Path(ANNOT)):
    state.clear()
    seam_state.clear()
    load_into_state(OUT)


def _write_rows(w):
    w.writerow(["motif_name", "sequence_name", "start", "end", "strand"])
    for idx in sorted(state):
        for s in sorted(state[idx], key=lambda x: x["start"]):
            w.writerow([s["motif_name"], idx, s["start"], s["end"], s["strand"]])
    # SEAM rows after the int-keyed Koo rows; sequence_name is the full
    # "SEAM:{ct}:{seq}:{map}" string so load_into_state can route it back.
    for sn in sorted(seam_state):
        for s in sorted(seam_state[sn], key=lambda x: x["start"]):
            w.writerow([s["motif_name"], sn, s["start"], s["end"], s["strand"]])


def write_out():
    with open(OUT, "w", newline="") as fh:
        _write_rows(csv.writer(fh))


def csv_text():
    buf = io.StringIO()
    _write_rows(csv.writer(buf))
    return buf.getvalue()


# Each non-default cell line's ~182 MB h5 is otherwise loaded lazily on
# its first /logo request, freezing the UI ~5 s. Warm them (and preds)
# in the background at startup so cell-line switches never stall. The
# default line is already loaded above (_d0).
def _warm():
    for m in METHODS:
        for cl in CELL_LINES:
            try:
                get_attr(cl, m)
                get_pred(cl, m)
            except Exception:
                pass


threading.Thread(target=_warm, daemon=True).start()

app = FastAPI()
BASES = ["A", "C", "G", "T"]


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/meta")
def meta():
    return {"n_seqs": int(N), "seq_len": int(L), "has_ohe": HAS_OHE,
            "has_names": HAS_NAMES, "cell_lines": CELL_LINES,
            "cell_line": DEFAULT_CL, "methods": METHODS,
            "method": DEFAULT_METHOD}


def ordered(q: str, sort: str, desc: bool, cl: str, method: str):
    idxs = list(range(N))
    if q:
        ql = q.lower()
        idxs = [i for i in idxs if ql in seq_name(i).lower()]
    if sort == "name":
        idxs.sort(key=lambda i: seq_name(i))
    elif sort == "pred":
        p = get_pred(cl, method)
        idxs.sort(key=lambda i: float(p[i]))
    if desc:
        idxs.reverse()
    return idxs


@app.get("/batch")
def batch(start: int = 0, n: int = 10, q: str = "", sort: str = "idx",
          dir: str = "asc", cl: str = DEFAULT_CL,
          method: str = DEFAULT_METHOD):
    em, ecl = resolve(method, cl)
    idxs = ordered(q, sort, dir == "desc", ecl, em)
    total = len(idxs)
    start = max(0, min(start, total))
    page = idxs[start:start + n]
    pred = get_pred(ecl, em)
    return {"total": total, "items": [
        {"idx": i, "name": seq_name(i),
         "pred": round(float(pred[i]), 4),
         "act": seq_act(i, ecl),
         "spans": state.get(i, [])} for i in page]}


@app.get("/pos")
def pos(idx: int, q: str = "", sort: str = "idx", dir: str = "asc",
        cl: str = DEFAULT_CL, method: str = DEFAULT_METHOD):
    # Position of an absolute seq idx within the current ordered list
    # (-1 if filtered out). Lets the UI jump by absolute idx.
    em, ecl = resolve(method, cl)
    idxs = ordered(q, sort, dir == "desc", ecl, em)
    try:
        return {"pos": idxs.index(idx)}
    except ValueError:
        return {"pos": -1}


@app.get("/logo/{idx}.png")
def logo(idx: int, mode: str = "hyp", cl: str = DEFAULT_CL,
         method: str = DEFAULT_METHOD):
    if idx < 0 or idx >= N:
        return Response(status_code=404)
    if dkey(method, cl) not in DATASETS:
        return Response(status_code=404)
    mode = "obs" if (mode == "obs" and HAS_OHE) else "hyp"
    key = (idx, mode, method, cl)
    cached = _LOGO_CACHE.get(key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    attr = get_attr(cl, method)[idx]  # (4, L), channel order A,C,G,T
    if mode == "obs":
        arr = (attr * OHE[idx]).T  # (L, 4)
    else:
        arr = attr.T               # (L, 4)
    png = render_logo_png(arr)
    _LOGO_CACHE[key] = png
    return Response(content=png, media_type="image/png")


def render_logo_png(arr):
    """Render an (Lp,4) A,C,G,T attribution array to a transparent PNG.

    Shared by /logo (hyp/obs) and /seam/logo; symmetric ylim so the zero
    baseline is vertically centered. Behavior here is byte-identical to
    the original inlined /logo hyp path.
    """
    arr = np.asarray(arr, dtype=np.float32)
    Lp = arr.shape[0]
    ymin = 0.0
    ymax = 0.0
    for p in range(Lp):
        v = arr[p]
        ymax = max(ymax, float(v[v > 0].sum()))
        ymin = min(ymin, float(v[v < 0].sum()))
    M = max(abs(ymin), abs(ymax))
    if M == 0:
        M = 1.0

    fig = plt.figure(figsize=(Lp * PX / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fast_logo(arr, ax, ylim=(-M, M))
    ax.set_xlim(-0.5, Lp - 0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=True)
    plt.close(fig)
    return buf.getvalue()


@app.post("/set/{idx}")
async def set_spans(idx: int, request: Request):
    body = await request.json()
    spans = []
    for s in body.get("spans", []):
        st, en = int(s["start"]), int(s["end"])
        st = max(0, min(st, L))
        en = max(0, min(en, L))
        if en <= st:
            continue
        spans.append({
            "motif_name": s.get("motif_name", ""),
            "start": st, "end": en,
            "strand": s.get("strand", "+") or "+",
        })
    if spans:
        state[idx] = spans
    else:
        state.pop(idx, None)
    write_out()
    return {"ok": True}


@app.post("/clear")
def clear_cache():
    # The only way to wipe the continuous cache (red button in UI).
    state.clear()
    write_out()
    return {"ok": True}


@app.get("/export.csv")
def export():
    return Response(
        csv_text(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=situate_annotations.csv"},
    )


def seam_npy_path(ct, seq, mp, src=SEAM_DEFAULT_SRC):
    return seam_root_for(src) / ct / str(seq) / f"{mp}.npy"


@app.get("/seam/available")
def seam_available(actbin: str = "", src: str = SEAM_DEFAULT_SRC):
    # Scan the selected SEAM tree; for each {ct,seq_idx} list maps whose
    # .npy exists. Tolerates a missing root / empty tree. Returns ALL
    # matching entries (the frontend paginates client-side); optional
    # `actbin` filter (activating|neutral|repressing) applied here.
    # `has_wt_ohe` tells the UI whether the obs toggle is usable.
    seam_root = seam_root_for(src)
    flt = actbin if actbin in SEAM_ACTBINS else ""
    entries = []
    total = 0
    if seam_root.is_dir():
        for ct in SEAM_CTS:
            ctd = seam_root / ct
            if not ctd.is_dir():
                continue
            for sd in sorted(ctd.iterdir(),
                             key=lambda p: (not p.name.isdigit(), p.name)):
                if not sd.is_dir():
                    continue
                maps = [m for m in SEAM_MAPS if (sd / f"{m}.npy").exists()]
                if not maps:
                    continue
                ab = ""
                if sd.name.isdigit():
                    ab = SEAM_ACTBIN.get(int(sd.name), {}).get("actbin", "")
                if flt and ab != flt:
                    continue
                total += 1
                sk = f"SEAM:{ct}:{sd.name}"
                entries.append({
                    "ct": ct, "seq": sd.name, "maps": maps,
                    "actbin": ab,
                    "spans": {m: seam_state.get(f"{sk}:{m}", [])
                              for m in maps},
                })
    return {"total": total, "entries": entries,
            "has_wt_ohe": bool(SEAM_WT_OHE)}


@app.get("/seam/logo")
def seam_logo(ct: str, seq: str, map: str, src: str = SEAM_DEFAULT_SRC,
              mode: str = "hyp"):
    if ct not in SEAM_CTS or map not in SEAM_MAPS:
        return Response(status_code=404)
    if src not in SEAM_SRCS:
        src = SEAM_DEFAULT_SRC
    p = seam_npy_path(ct, seq, map, src)
    if not p.exists():
        return Response(status_code=404)
    key = (src, ct, seq, map, mode)
    cached = _SEAM_LOGO_CACHE.get(key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")
    try:
        arr = np.asarray(np.load(p), dtype=np.float32)  # (200,4) A,C,G,T
    except Exception:
        return Response(status_code=404)
    if arr.ndim != 2 or arr.shape != (SEAM_L, 4):
        return Response(status_code=404)
    if mode == "obs":
        o = SEAM_WT_OHE.get(int(seq)) if str(seq).isdigit() else None
        if o is not None:
            arr = arr * o
    png = render_logo_png(arr)
    _SEAM_LOGO_CACHE[key] = png
    return Response(content=png, media_type="image/png")


@app.post("/seam/set")
async def seam_set(request: Request):
    body = await request.json()
    ct, seq, mp = body.get("ct"), str(body.get("seq")), body.get("map")
    if ct not in SEAM_CTS or mp not in SEAM_MAPS:
        return Response(status_code=404)
    spans = []
    for s in body.get("spans", []):
        st, en = int(s["start"]), int(s["end"])
        st = max(0, min(st, SEAM_L))
        en = max(0, min(en, SEAM_L))
        if en <= st:
            continue
        spans.append({
            "motif_name": s.get("motif_name", ""),
            "start": st, "end": en,
            "strand": s.get("strand", "+") or "+",
        })
    k = f"SEAM:{ct}:{seq}:{mp}"
    if spans:
        seam_state[k] = spans
    else:
        seam_state.pop(k, None)
    write_out()
    return {"ok": True}


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
