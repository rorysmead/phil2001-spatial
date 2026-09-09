# Spatial Evolution Explorer

Lattice-based evolutionary game simulator — a teaching resource for PHIL 2001, *Ethics and Evolutionary Games*. Replaces the GenLab web app used in prior years.

## Use it (nothing to install)

- **Explore with controls:** open the widget → **https://rorysmead.github.io/phil2001-spatial/**
- **Read / edit the code:** open in Colab →
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rorysmead/phil2001-spatial/blob/main/notebook.ipynb)

## What it does

- 1D/2D lattices, wrapped or bounded; von Neumann and Moore neighbourhoods (r=1, r=2).
- Any symmetric game up to 16 strategies, with presets (PD, Stag Hunt, Hawk–Dove, RPS, Nash demand, Lewis signalling, …).
- Seven update rules (imitation, best response, Fermi, death–birth, death–selection).
- Realized assortment `r` measured each generation — the same `r` as the replicator tool's slider, so lattice runs cross-check against the well-mixed model.
- Seeded and shareable (config encodes into the URL); CSV / PNG export.

## Run locally

```bash
pip install numpy matplotlib marimo
python3 tests.py        # verify the model
marimo run app.py       # the widget
```

## Files

`spatial.py` (model) · `render.py` (plots) · `app.py` (widget) · `notebook.ipynb` (Colab) · `tests.py` (checks).

Build record, design rationale, reproduced results, and deployment notes: [`NOTES.md`](NOTES.md).


## License

MIT (c) 2026 Rory Smead and contributors. See [LICENSE](LICENSE). Contributions are welcome under the same license, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use this tool in teaching or research, please cite it. See [CITATION.cff](CITATION.cff), or use the "Cite this repository" button on GitHub.
