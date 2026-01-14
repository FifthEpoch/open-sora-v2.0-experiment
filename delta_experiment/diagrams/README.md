# δ-TTA diagrams (camera-ready)

This folder contains:
- `PROMPTS.md`: three standalone prompts (A/B/C) for image generators to create vector-style diagrams.
- `diagram_lib.py`: small Matplotlib-based vector diagram primitives + a detailed Open-Sora v2.0 baseline layout.
- `overlays.py`: Option A/B/C δ injection overlays.
- `render_all.py`: renderer that exports each diagram to **PDF + SVG**.

## Render (PDF + SVG)

From the repo root:

```bash
python3 -m delta_experiment.diagrams.render_all --out-dir delta_experiment/diagrams/out
```

Optional: render only one option:

```bash
python3 -m delta_experiment.diagrams.render_all --only a
python3 -m delta_experiment.diagrams.render_all --only b
python3 -m delta_experiment.diagrams.render_all --only c
```

Outputs are written to `delta_experiment/diagrams/out/` as:
- `delta_a.pdf`, `delta_a.svg`
- `delta_b.pdf`, `delta_b.svg`
- `delta_c.pdf`, `delta_c.svg`

