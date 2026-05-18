# situate

Local FastAPI + vanilla-JS app for interactively editing motif boundary
annotations on attribution / SEAM-foreground maps. Drag span edges to adjust
boundaries, drag empty plot to create a span, click + Delete to remove,
"Set" to persist a row, "Download CSV" for the full table.

Run:

    ./situate

then open http://<node>:8000

With no args it loads the hardcoded default attribution data
(`/grid/koo/home/pmantill/projects/kcee-ui/data/attributions/koo_standardtorch/K562_deeplift.h5`,
dataset `attr`, 56975 x 4 x 200), starts with no annotations, and writes
edits to `./edits.csv`.

These DeepLIFT maps are **hypothetical** (all 4 channels nonzero per
position), so the default render is the **Hypothetical** stacked logo
(positive letters up, negative down). The top-bar **Hypothetical |
Observed** toggle defaults to Hypothetical; **Observed** (single
attr x one-hot letter) is only enabled when `--ohe` supplies a one-hot
array.

Optional overrides: `--attr` (.h5/.hdf5 or .npz/.npy), `--ohe`,
`--annot` (pre-load spans), `--out` (default `./edits.csv`).
