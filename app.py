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


# Koo Hippo-MPRA deeplift attributions are the only built-in defaults.
# Each h5 holds `attr` (N,4,L); rows are aligned 1:1 to manifest.csv
# (the 56975-row library with the 5 raw NA rows already dropped), so
# attr row i  <->  manifest row i. The raw joint_library_combined.csv
# (56980 rows) is reference only and is NOT indexed by attr row.
CELL_LINES = ["HepG2", "K562", "WTC11"]
DEFAULT_CL = "HepG2"
ATTR_DIR = HERE / "data" / "attributions"
MANIFEST = HERE / "data" / "library" / "manifest.csv"


def cl_path(cl):
    return ATTR_DIR / f"{cl}_deeplift.h5"


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
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()


args = parse_args()

# Dataset registry. With no --attr the 3 Koo cell lines are the only
# selectable datasets; --attr overrides to a single "custom" dataset.
if args.attr:
    DATASETS = {"custom": args.attr}
    CELL_LINES = ["custom"]
    DEFAULT_CL = "custom"
else:
    DATASETS = {cl: str(cl_path(cl)) for cl in CELL_LINES}
    missing = [cl for cl in CELL_LINES if not Path(DATASETS[cl]).exists()]
    if missing:
        raise SystemExit(f"missing attribution h5 for: {missing}")

_ATTR: dict[str, np.ndarray] = {}


def get_attr(cl):
    a = _ATTR.get(cl)
    if a is None:
        a = load_arr(DATASETS[cl])
        if a.shape != (N, 4, L):
            raise SystemExit(
                f"{cl} attr shape {a.shape} != expected ({N},4,{L})")
        _ATTR[cl] = a
    return a


_d0 = load_arr(DATASETS[DEFAULT_CL])
N, _, L = _d0.shape
_ATTR[DEFAULT_CL] = _d0

OUT = args.out or "edits.csv"

# manifest.csv is row-aligned to attr (NA rows already dropped). Read it
# once for names + sequences. --names overrides names; --attr custom uses
# neither manifest names nor a one-hot.
ADAPTER = 15  # 15bp adapter each side of the 230bp insert; attr is [15:215]
NAMES = None
OHE = None
if args.attr is None and MANIFEST.exists():
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N:
        raise SystemExit(
            f"manifest rows {len(rows)} != attr rows {N} "
            f"(NA-row alignment broken)")
    NAMES = [r.get("name", "") for r in rows]
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


def get_pred(cl):
    p = _PRED.get(cl)
    if p is None:
        try:
            import h5py
            with h5py.File(DATASETS[cl], "r") as f:
                p = (f["predictions"][:] if "predictions" in f
                     else np.zeros(N, np.float32))
        except Exception:
            p = np.zeros(N, np.float32)
        p = np.asarray(p, np.float32)
        if p.shape[0] != N:
            p = np.zeros(N, np.float32)
        _PRED[cl] = p
    return _PRED[cl]


def seq_name(idx):
    if NAMES is not None and 0 <= idx < len(NAMES):
        return NAMES[idx]
    return ""


state = {}            # universal seq idx -> [ {motif_name,start,end,strand} ]


def load_into_state(path):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                idx = int(row["sequence_name"])
            except (KeyError, ValueError):
                continue
            state.setdefault(idx, []).append({
                "motif_name": row.get("motif_name", ""),
                "start": int(float(row["start"])),
                "end": int(float(row["end"])),
                "strand": row.get("strand", "+") or "+",
            })


# Continuous cache: OUT is the running annotation .csv, keyed by universal
# seq idx, so it survives paging / cell-line / dataset changes AND server
# restarts. Seed from --annot first, then the existing OUT (OUT wins).
if args.annot and Path(args.annot).exists():
    load_into_state(args.annot)
if Path(OUT).exists() and (not args.annot or
                           Path(OUT) != Path(args.annot)):
    state.clear()
    load_into_state(OUT)


def write_out():
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["motif_name", "sequence_name", "start", "end", "strand"])
        for idx in sorted(state):
            for s in sorted(state[idx], key=lambda x: x["start"]):
                w.writerow([s["motif_name"], idx, s["start"], s["end"], s["strand"]])


def csv_text():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["motif_name", "sequence_name", "start", "end", "strand"])
    for idx in sorted(state):
        for s in sorted(state[idx], key=lambda x: x["start"]):
            w.writerow([s["motif_name"], idx, s["start"], s["end"], s["strand"]])
    return buf.getvalue()


# Each non-default cell line's ~182 MB h5 is otherwise loaded lazily on
# its first /logo request, freezing the UI ~5 s. Warm them (and preds)
# in the background at startup so cell-line switches never stall. The
# default line is already loaded above (_d0).
def _warm():
    for cl in CELL_LINES:
        try:
            get_attr(cl)
            get_pred(cl)
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
            "cell_line": DEFAULT_CL}


def ordered(q: str, sort: str, desc: bool, cl: str):
    idxs = list(range(N))
    if q:
        ql = q.lower()
        idxs = [i for i in idxs if ql in seq_name(i).lower()]
    if sort == "name":
        idxs.sort(key=lambda i: seq_name(i))
    elif sort == "pred":
        p = get_pred(cl)
        idxs.sort(key=lambda i: float(p[i]))
    if desc:
        idxs.reverse()
    return idxs


@app.get("/batch")
def batch(start: int = 0, n: int = 10, q: str = "", sort: str = "idx",
          dir: str = "asc", cl: str = DEFAULT_CL):
    idxs = ordered(q, sort, dir == "desc", cl if cl in DATASETS else DEFAULT_CL)
    total = len(idxs)
    start = max(0, min(start, total))
    page = idxs[start:start + n]
    pred = get_pred(cl) if cl in DATASETS else None
    return {"total": total, "items": [
        {"idx": i, "name": seq_name(i),
         "pred": (round(float(pred[i]), 4) if pred is not None else None),
         "spans": state.get(i, [])} for i in page]}


@app.get("/pos")
def pos(idx: int, q: str = "", sort: str = "idx", dir: str = "asc",
        cl: str = DEFAULT_CL):
    # Position of an absolute seq idx within the current ordered list
    # (-1 if filtered out). Lets the UI jump by absolute idx.
    idxs = ordered(q, sort, dir == "desc", cl if cl in DATASETS else DEFAULT_CL)
    try:
        return {"pos": idxs.index(idx)}
    except ValueError:
        return {"pos": -1}


@app.get("/logo/{idx}.png")
def logo(idx: int, mode: str = "hyp", cl: str = DEFAULT_CL):
    if idx < 0 or idx >= N:
        return Response(status_code=404)
    if cl not in DATASETS:
        return Response(status_code=404)
    mode = "obs" if (mode == "obs" and HAS_OHE) else "hyp"
    key = (idx, mode, cl)
    cached = _LOGO_CACHE.get(key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    attr = get_attr(cl)[idx]  # (4, L), channel order A,C,G,T
    if mode == "obs":
        arr = (attr * OHE[idx]).T  # (L, 4)
    else:
        arr = attr.T               # (L, 4)
    arr = np.asarray(arr, dtype=np.float32)

    # Symmetric ylim so the zero baseline is vertically centered.
    ymin = 0.0
    ymax = 0.0
    for p in range(L):
        v = arr[p]
        ymax = max(ymax, float(v[v > 0].sum()))
        ymin = min(ymin, float(v[v < 0].sum()))
    M = max(abs(ymin), abs(ymax))
    if M == 0:
        M = 1.0

    fig = plt.figure(figsize=(L * PX / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fast_logo(arr, ax, ylim=(-M, M))
    ax.set_xlim(-0.5, L - 0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=True)
    plt.close(fig)
    png = buf.getvalue()
    _LOGO_CACHE[key] = png
    return Response(content=png, media_type="image/png")


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


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
