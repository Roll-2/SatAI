# Tools

Run everything from the project root with the project virtualenv
(`.venv/bin/python …`).

| File | What it is |
|---|---|
| `satai.py` | Command-line classifier (`--input`, `--batch-dir`, `--backend auto\|coreml\|pytorch`) |
| `convert_to_coreml.py` | Exports `Models/astro_net_v2_best.pt` to `Models/AstroNetV2.mlpackage` and verifies logit equivalence |
| `smoke_test.py` | 33 end-to-end checks on synthetic data (preprocessing, model, training loop, inference, CLI) |
| `test_app_bridge.py` | 32 headless checks of the desktop bridge, both backends, upload cleanup and the `why()`/`identify_object()` chips |
| `make_app.py` | Rebuilds `SatAi.app` (icon + launcher) after moving the project or changing the app entry point |
| `visualize_neurons.py` | Filters, activations, saliency and Dense1 map for one crop (optional, needs matplotlib) |
| `make_model_diagram.py` | Optional 2D poster generator from the live model and a real crop; the main visual map is the Blender scene in `Visuals/` |
| `build_png_validation.py` | Builds photo-like PNG validation mosaics from staged crops |
| `detached_runner.py` | Launches a long acquisition/streaming command fully detached from the terminal |
| `training/` | Model inspection, backup and retraining chain — see [`training/README.md`](training/README.md) |

The remaining one-off diagnostics are in `Trash/scripts/`; nothing in
`App/`, `Src/` or `Tools/` imports them.
