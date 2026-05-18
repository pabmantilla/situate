"""Vendored DNA attribution logo renderer (core of EigenMaps fast_logo.py).

Only the glyph-path rendering primitive is kept; the HDF5 wrapper, batch
save loop, and CLI from the source were intentionally left out.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt  # noqa: F401  (Agg-backed; imported for callers)
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath


# ---------------------------------------------------------------------------
# DNA color palette
# ---------------------------------------------------------------------------

_DNA_COLORS: dict[str, tuple[float, float, float]] = {
    "A": (0.0, 0.5, 0.0),
    "C": (0.0, 0.0, 1.0),
    "G": (1.0, 0.65, 0.0),
    "T": (1.0, 0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Glyph geometry cache
# ---------------------------------------------------------------------------

class _GlyphCache:
    """Lazily populated cache of pre-computed glyph geometry for A/C/G/T."""

    def __init__(self) -> None:
        self.verts: dict[str, np.ndarray] = {}
        self.codes: dict[str, np.ndarray] = {}
        self.xmin: dict[str, float] = {}
        self.ymin: dict[str, float] = {}
        self.w: dict[str, float] = {}
        self.h: dict[str, float] = {}
        # Pre-flipped geometry (for negative attributions drawn upside-down)
        self.flip_verts: dict[str, np.ndarray] = {}
        self.flip_ymin: dict[str, float] = {}
        self.flip_h: dict[str, float] = {}
        self.ref_w: float = 0.0
        self.ready: bool = False

    def build(
        self,
        font_name: str = "sans",
        font_weight: str = "bold",
        ref_char: str = "E",
    ) -> None:
        """Compute and cache glyph geometry (called once on first use)."""
        if self.ready:
            return
        fp = fm.FontProperties(family=font_name, weight=font_weight)
        for ch in "ACGT":
            tp = TextPath((0, 0), ch, size=1, prop=fp)
            ext = tp.get_extents()
            v = np.array(tp.vertices, dtype=np.float64)
            self.verts[ch] = v
            self.codes[ch] = np.array(tp.codes, dtype=np.uint8)
            self.xmin[ch] = float(ext.xmin)
            self.ymin[ch] = float(ext.ymin)
            self.w[ch] = float(ext.width)
            self.h[ch] = float(ext.height)
            fv = v.copy()
            fv[:, 1] = -fv[:, 1]
            self.flip_verts[ch] = fv
            self.flip_ymin[ch] = float(fv[:, 1].min())
            self.flip_h[ch] = float(fv[:, 1].max()) - self.flip_ymin[ch]
        self.ref_w = TextPath((0, 0), ref_char, size=1, prop=fp).get_extents().width
        self.ready = True


_CACHE = _GlyphCache()


# ---------------------------------------------------------------------------
# Core: fast_logo
# ---------------------------------------------------------------------------

def fast_logo(
    values: np.ndarray,
    ax: Axes,
    width: float = 0.95,
    height_scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Render a single attribution logo on a Matplotlib axis.

    Parameters
    ----------
    values:
        Attribution matrix of shape ``(L, 4)`` with columns ordered A/C/G/T.
        Positive values stack upward; negative values are drawn flipped
        (upside-down) and stack downward.
    ax:
        Axis to draw onto.
    width:
        Horizontal glyph width as a fraction of one position unit.
    height_scale:
        Scalar multiplier applied to all attribution values before drawing.
    ylim:
        Fixed y-axis limits.  When omitted the limits are inferred from the
        data with a small 5 % padding.
    """
    _CACHE.build()
    if values.ndim == 2 and values.shape[0] == 4 and values.shape[1] != 4:
        values = values.T
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected values shape (L, 4), got {values.shape}")

    seq_len = values.shape[0]
    chars = list("ACGT")
    patches: list[PathPatch] = []
    facecolors: list[tuple[float, float, float]] = []
    y_min = 0.0
    y_max = 0.0

    for pos in range(seq_len):
        vs = values[pos] * float(height_scale)
        order = np.argsort(vs)
        vs_sorted = vs[order]
        cs = [chars[i] for i in order]

        # Negative attributions accumulate below zero; positive ones above.
        floor = float(np.sum(vs_sorted[vs_sorted < 0]))
        pos_min = floor

        for v, ch in zip(vs_sorted, cs):
            h = abs(float(v))
            if h == 0.0:
                continue
            ceiling = floor + h
            flip = v < 0
            bx = pos - width / 2.0

            if flip:
                vt = _CACHE.flip_verts[ch]
                oy, oh = _CACHE.flip_ymin[ch], _CACHE.flip_h[ch]
            else:
                vt = _CACHE.verts[ch]
                oy, oh = _CACHE.ymin[ch], _CACHE.h[ch]
            ow = _CACHE.w[ch]
            ox = _CACHE.xmin[ch]

            hstretch = min(width / ow, width / _CACHE.ref_w)
            cw = hstretch * ow
            shift = (width - cw) / 2.0
            vstretch = h / oh

            new_verts = vt.copy()
            new_verts[:, 0] = (vt[:, 0] - ox) * hstretch + bx + shift
            new_verts[:, 1] = (vt[:, 1] - oy) * vstretch + floor

            patches.append(PathPatch(MplPath(new_verts, _CACHE.codes[ch])))
            facecolors.append(_DNA_COLORS[ch])
            floor = ceiling

        pos_max = floor
        y_min = min(y_min, pos_min)
        y_max = max(y_max, pos_max)

    pc = PatchCollection(
        patches,
        match_original=False,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection(pc)
    ax.set_xlim(-0.5, seq_len - 0.5)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        if y_max == y_min:
            y_max = y_min + 1.0
        pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)
