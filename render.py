"""
render.py -- the drawing layer for the spatial-evolution explorer.

Matplotlib only. There is no model here: everything in this file takes a
finished `Sim` (or a bare lattice) and turns it into a picture or a file. If you
are looking for how the simulation WORKS, read spatial.py instead.

The one rule this file follows: a strategy keeps the same colour everywhere --
in the lattice, in the frequency plot, in the legend. Colour is the reader's
only handle on which strategy is which, so it must not move.
"""
import csv
import io

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

# A qualitative palette, ordered so the first few are distinguishable to the most
# common forms of colour blindness -- strategy 1 vs strategy 2 is the comparison
# students make most often, so blue/orange leads. (GenLab opened with chartreuse
# and indigo, which is a hard pair to tell apart on a projector.)
PALETTE = [
    "#3B7EA1",  # blue
    "#E07B39",  # orange
    "#4C9A6A",  # green
    "#B5536A",  # rose
    "#8A6FA8",  # purple
    "#C4A13A",  # ochre
    "#6BA8C4",  # light blue
    "#A0654A",  # brown
    "#7D8B99",  # slate
    "#D4A0B0",  # pink
    "#5B7355",  # olive
    "#C9713A",  # rust
    "#7A6C93",  # violet
    "#9DB068",  # sage
    "#4A5D6B",  # dark slate
    "#D9B48F",  # sand
]


def strategy_cmap(k):
    """A colormap that maps strategy index -> its fixed colour."""
    return mcolors.ListedColormap(PALETTE[:k])


def _names(model, names=None):
    if names is not None:
        return list(names)
    return [f"S{i + 1}" for i in range(model.k)]


# ------------------------------------------------------------------------------
# The lattice itself
# ------------------------------------------------------------------------------

def draw_lattice(sim, names=None, ax=None, title=None, show_grid_lines=None):
    """Draw the current lattice.

    Drawn with imshow on a fixed colour scale, so the picture is comparable
    across generations and across runs -- an image, not a table of coloured
    cells, which is what lets this stay smooth at 200x200 where GenLab's HTML
    table of <td> elements did not.
    """
    m = sim.model
    ax = ax or plt.subplots(figsize=(5.2, 5.2))[1]
    ax.imshow(sim.grid, cmap=strategy_cmap(m.k), vmin=-0.5, vmax=m.k - 0.5,
              interpolation="nearest")
    # thin cell borders help only on small lattices; above ~40 they are the picture
    if show_grid_lines is None:
        show_grid_lines = max(m.rows, m.cols) <= 40
    if show_grid_lines:
        ax.set_xticks(np.arange(-0.5, m.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, m.rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="minor", length=0)
    freqs = sim.snapshot()["frequencies"]
    lab = "   ".join(f"{n} {100 * f:.0f}%" for n, f in zip(_names(m, names), freqs))
    ax.set_title(title or f"generation {sim.generation}\n{lab}", fontsize=10)
    return ax.figure


def draw_spacetime(sim, names=None, ax=None):
    """Space across, time downwards -- the right picture for a 1D lattice.

    A single row of cells says almost nothing on its own; stacking the
    generations is what makes the dynamics visible. Needs keep_grids=True.
    """
    m = sim.model
    st = sim.spacetime()
    ax = ax or plt.subplots(figsize=(6.4, 4.4))[1]
    ax.imshow(st, cmap=strategy_cmap(m.k), vmin=-0.5, vmax=m.k - 0.5,
              interpolation="nearest", aspect="auto")
    ax.set_xlabel("site")
    ax.set_ylabel("generation")
    ax.set_title("space-time diagram", fontsize=10)
    return ax.figure


# ------------------------------------------------------------------------------
# The numbers over time
# ------------------------------------------------------------------------------

def plot_frequencies(sim, names=None, ax=None):
    """Strategy frequencies against generation."""
    m = sim.model
    freqs = sim.series("frequencies")
    gens = sim.series("generation")
    ax = ax or plt.subplots(figsize=(6.4, 3.0))[1]
    for i, name in enumerate(_names(m, names)):
        ax.plot(gens, freqs[:, i], color=PALETTE[i], linewidth=1.8, label=name)
    ax.set_xlabel("generation")
    ax.set_ylabel("frequency")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max(int(gens[-1]), 1))
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax.figure


def plot_assortment(sim, ax=None):
    """Realized assortment r against generation.

    The line to watch. It starts near 0 on a random lattice and climbs as
    clusters form: that rise IS the mechanism by which space changes the
    outcome. Where it settles is the number to carry over to the replicator
    explorer's r slider.

    Gaps in the line are generations where the population went monomorphic and r
    is undefined -- not missing data.
    """
    r = sim.series("assortment")
    gens = sim.series("generation")
    ax = ax or plt.subplots(figsize=(6.4, 2.6))[1]
    ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.plot(gens, r, color="#333333", linewidth=1.8)
    ax.set_xlabel("generation")
    ax.set_ylabel("assortment  r")
    ax.set_ylim(-1.05, 1.05)
    # fix the axis to the run: once the population goes monomorphic r is nan, and
    # autoscaling would otherwise shrink the axis to the handful of defined points
    ax.set_xlim(0, max(int(gens[-1]), 1))
    ax.spines[["top", "right"]].set_visible(False)
    return ax.figure


def dashboard(sim, names=None, figsize=(11.0, 5.2)):
    """Lattice, frequencies and assortment in one figure."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 0.8],
                          hspace=0.45, wspace=0.25)
    ax_grid = fig.add_subplot(gs[:, 0])
    if sim.model.is_1d and sim.grids:
        draw_spacetime(sim, names, ax=ax_grid)
    else:
        draw_lattice(sim, names, ax=ax_grid)
    plot_frequencies(sim, names, ax=fig.add_subplot(gs[0, 1]))
    plot_assortment(sim, ax=fig.add_subplot(gs[1, 1]))
    return fig


# ------------------------------------------------------------------------------
# Getting the run out of the tool
# ------------------------------------------------------------------------------

def series_csv(sim, names=None):
    """The whole run as CSV text: one row per generation.

    Columns: generation, one frequency column per strategy, assortment, mean
    payoff. Paste into a spreadsheet, or hand it in with the homework.
    """
    m = sim.model
    out = io.StringIO()
    w = csv.writer(out)
    labels = _names(m, names)
    w.writerow(["generation"] + [f"freq_{n}" for n in labels]
               + ["assortment", "mean_payoff"])
    for h in sim.history:
        w.writerow([h["generation"]] + [f"{v:.6f}" for v in h["frequencies"]]
                   + [f"{h['assortment']:.6f}", f"{h['mean_payoff']:.6f}"])
    return out.getvalue()


def save_run(sim, stem, names=None, dpi=150):
    """Write `stem.png` (the dashboard) and `stem.csv` (the series). Returns both paths."""
    fig = dashboard(sim, names)
    png, csv_path = f"{stem}.png", f"{stem}.csv"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    with open(csv_path, "w", newline="") as fh:
        fh.write(series_csv(sim, names))
    return png, csv_path
