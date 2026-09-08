# MAGO-SP repo notes

MICCAI 2025 paper code for detection and correction of water-fat swaps in magnitude-only VIBE MRI. Main code lives in `papers/vibe_inversion/`.

## Layout
- `papers/vibe_inversion/recon_mevibe.py` — CPU MEVIBE reconstruction (per-voxel `scipy.optimize.least_squares`). Holds the **module-level default peak model** (`freqs_ppm`, `alpha_p`) that flows into every downstream fit.
- `papers/vibe_inversion/recon_mevibe_gpu.py` — batched PyTorch fit. Imports the defaults from `recon_mevibe` via `_DEFAULT_ALPHA_P` / `_DEFAULT_FREQS_PPM`, so a change in `recon_mevibe.py` automatically propagates.
- `papers/vibe_inversion/mago_methods.py` — standalone `mago(...)`, `magorino(...)`, `mago_sp(...)` (+ `*_ISMRM`).
- `papers/vibe_inversion/pipeline.py` — full pipeline entry points (`pipeline`, `pipeline_bids`).
- `papers/vibe_inversion/tests/` — test scripts. `nako_mevibe.py` / `run_mevibe_test.py` / `test.py` / `test2.py` pass explicit `alpha_p` / `freqs_ppm` per experiment — they do NOT rely on the module default.

## Default fat peak model
The active default is **Hamilton 9-peak liver** (Hamilton et al., NMR Biomed 2011, https://doi.org/10.1002/nbm.1622). See `recon_mevibe.py` lines ~11-24 — Ren marrow (MAGO-SP paper), Zhong, Hernando, UKBB alternates are kept as commented lines to swap in quickly.

To change the default: edit the two active `freqs_ppm = ...` / `alpha_p = ...` lines in `recon_mevibe.py` and update the "current default" markers in `README.md` (Fat-peak models section paragraph + the table row). The GPU file needs no change.

## Sign convention
`freqs_ppm` is fed into `get_freqs_hz(freqs_ppm, MagneticFieldStrength)` and multiplied by the scanner center frequency directly. Ren / Hamilton / Hernando / Zhong sets are **water-referenced** (mostly negative). UKBB sets are **TMS-referenced** — `tests/run_mevibe_test.py` subtracts 4.7 to convert. Dropping in a new set without checking silently offsets everything by ~600 Hz at 3 T.

## Commit style
See recent log — short lowercase subjects (`ruff check`, `add pyproject`, `prevent nan, ignore labels`). No conventional-commits prefixes.
