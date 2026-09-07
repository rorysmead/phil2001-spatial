"""
app.py -- the explorer WIDGET (marimo reactive notebook).

For students who want to *play* with the model, not read code. Everything
updates live as you change a control. The model is imported from spatial.py and
the drawing from render.py, so this file only wires controls to pictures.

Editing note: this is a marimo notebook. Each `@app.cell` function is a reactive
cell -- marimo re-runs a cell automatically whenever a value it reads (a slider,
dropdown, or another cell's output) changes. A cell's LAST expression is what it
displays. So to change the UI you edit these cells; there is no manual "refresh".

How it stays responsive: the whole run is simulated ONCE up to the generation
limit and stored, and the generation control then scrubs through the stored run
(`Sim.at`). Playing the animation is therefore just reading out frames, not
re-simulating, which is what makes it usable in the browser.

The payoff matrix lives in a marimo STATE rather than in the widgets, because
four different things can replace it -- picking a preset, changing the number of
strategies, rolling random payoffs, or loading a shared link -- while the number
boxes themselves are only for typing in. Whenever the state changes the boxes are
rebuilt from it; typing in a box does not write back to the state.

Run it locally:
    pip install marimo matplotlib numpy
    marimo run app.py            # read-only app mode (what students use)
    marimo edit app.py           # editable mode

Share with no install:
    marimo export html-wasm app.py -o build/ --mode run
    cp spatial.py render.py build/          # REQUIRED, see the README
"""
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    # Loading the model modules, in the browser and out of it.
    #
    # Three things go wrong if you just write `import spatial` here, and all
    # three are silent -- the page simply stays blank:
    #
    #   1. A WASM export bundles the NOTEBOOK, not the files it imports, so
    #      spatial.py and render.py are not in the build at all. They have to be
    #      copied next to index.html at deploy time (the README and the Pages
    #      workflow both do this) and then fetched into Pyodide's filesystem.
    #   2. The fetch must use the ABSOLUTE url from mo.notebook_location(). A
    #      relative "./spatial.py" resolves against marimo's virtual filesystem
    #      and quietly returns a few hundred bytes of something else, which then
    #      fails to parse as Python.
    #   3. marimo's WASM loader scans this file for imports and installs them
    #      from PyPI. A bare `import spatial` makes it hunt for a package called
    #      "spatial" and fail. Hence importlib, which it cannot see -- and hence
    #      the otherwise pointless `import matplotlib` below, which is how
    #      matplotlib gets installed at all, since the only real import of it is
    #      inside render.py where the scanner never looks.
    #
    # Outside the browser the emscripten branch is skipped and this is an
    # ordinary import.
    import importlib
    import sys

    import matplotlib  # noqa: F401  (declares the dependency for the WASM build)
    import numpy as np

    import marimo as mo

    if sys.platform == "emscripten":
        import pathlib

        import pyodide.http

        _base = str(mo.notebook_location()).rstrip("/")
        for _name in ("spatial.py", "render.py"):
            pathlib.Path(_name).write_text(
                pyodide.http.open_url(f"{_base}/{_name}").getvalue())
        if "." not in sys.path:
            sys.path.insert(0, ".")

    sp = importlib.import_module("spatial")
    R = importlib.import_module("render")
    return mo, np, R, sp


@app.cell
def _(mo, sp):
    # A configuration can arrive in the page URL as ?cfg=..., so a link
    # reproduces someone else's run exactly rather than approximately.
    _raw = mo.query_params().get("cfg")
    try:
        shared = sp.decode_config(_raw) if _raw else None
    except Exception:
        shared = None          # a corrupted link should not break the page
    start_model, start_seed, start_init, start_probs, start_names, start_label = (
        shared if shared else (None, 0, "random", None, None, None))
    return start_model, start_seed, start_init, start_probs, start_names, start_label


@app.cell
def _(mo):
    mo.md(
        """
        # Spatial evolution

        Players sit on a lattice, play a game with their **neighbours**, and copy
        or are replaced according to how well those neighbours did. The picture on
        the left is the population; the plots on the right are what happened.

        The line to watch is **assortment r**. Space has no direct effect on which
        strategy wins -- it works by making like meet like, and `r` measures how
        much of that the lattice has built. Read `r` off a run, then set the
        replicator explorer's `r` slider to that value and see whether a
        well-mixed population goes the same way.
        """
    )
    return


# ------------------------------------------------------------------------------
# The game: preset / number of strategies / random payoffs, all via one state
# ------------------------------------------------------------------------------

@app.cell
def _(mo, np, sp, start_label, start_model, start_names):
    _default = "Prisoner's Dilemma - Nowak & May (b=1.85)"
    if start_model is not None:
        _A = np.asarray(start_model.A, float)
        _names = (tuple(start_names) if start_names
                  else tuple(f"S{i + 1}" for i in range(len(_A))))
        _init_game = {"A": _A, "names": _names,
                      "label": start_label or "shared link"}
    else:
        _A, _n = sp.GAMES[_default]
        _init_game = {"A": np.asarray(_A, float), "names": tuple(_n),
                      "label": _default}
    # the single source of truth for the payoff matrix; the number boxes below
    # are rebuilt from it whenever it changes
    get_game, set_game = mo.state(_init_game)
    return get_game, set_game


@app.cell
def _(get_game, mo, np, set_game, sp):
    def _use_preset(name):
        _A, _n = sp.GAMES[name]
        set_game({"A": np.asarray(_A, float), "names": tuple(_n), "label": name})

    def _set_k(k):
        def _resize(g):
            k_i = int(k)
            if k_i == len(g["A"]):
                return g
            names = tuple(list(g["names"][:k_i])
                          + [f"S{i + 1}" for i in range(len(g["names"]), k_i)])
            return {"A": sp.resize_matrix(g["A"], k_i), "names": names,
                    "label": f"{g['label']}, resized to k={k_i}"}
        set_game(_resize)

    def _roll(_):
        def _random(g):
            _A, _n = sp.random_game(len(g["A"]), low=0, high=10)
            return {"A": _A, "names": _n, "label": "random payoffs"}
        set_game(_random)

    preset = mo.ui.dropdown(options=list(sp.GAMES.keys()), value=None,
                            label="Load a preset", on_change=_use_preset)
    kdim = mo.ui.dropdown(options={str(i): i for i in range(2, 17)},
                          value=str(len(get_game()["A"])),
                          label="Strategies (k)", on_change=_set_k)
    roll = mo.ui.button(label="Random payoffs (integers 0-10)", on_change=_roll)
    mo.vstack([
        mo.hstack([preset, kdim, roll], justify="start", gap=1.2),
        # the dropdown resets to "--" after loading, so state the current game
        mo.md(f"Currently loaded: **{get_game()['label']}** "
              f"({len(get_game()['A'])} strategies)")
        .style({"font-size": "12px", "color": "#555"}),
    ], gap=0.3)
    return


@app.cell
def _(get_game, mo):
    # the editable payoff matrix, rebuilt from the state
    _g = get_game()
    names = _g["names"]
    n_strat = len(_g["A"])
    # NOTE: no .style() on these. mo.ui.number(...).style(...) returns an Html
    # WRAPPER rather than a UIElement, and mo.ui.array needs real elements --
    # feeding it wrapped ones fails later with "'Html' object has no attribute
    # '_clone'". Width is handled by the scrolling container instead.
    matrix = mo.ui.array([
        mo.ui.number(value=float(_g["A"][i][j]), step=0.05)
        for i in range(n_strat) for j in range(n_strat)])
    return matrix, n_strat, names


@app.cell
def _(matrix, mo, n_strat, names):
    _lw, _cw = ("70px", "92px") if n_strat > 6 else ("100px", "108px")
    _fs = "11px" if n_strat > 6 else "13px"
    _head = mo.hstack([mo.md("").style({"width": _lw, "flex": "0 0 auto"}),
                       *[mo.md(f"*{names[j]}*")
                         .style({"width": _cw, "flex": "0 0 auto",
                                 "font-size": "11px"})
                         for j in range(n_strat)]], justify="start", gap=0.25)
    _rows = [mo.hstack([mo.md(f"**{names[i]}**")
                        .style({"width": _lw, "flex": "0 0 auto",
                                "font-size": _fs}),
                        *[mo.vstack([matrix[i * n_strat + j]])
                          .style({"width": _cw, "flex": "0 0 auto"})
                          for j in range(n_strat)]],
                       justify="start", gap=0.25)
             for i in range(n_strat)]
    mo.vstack([
        mo.md("**Payoff matrix** — row = my strategy, column = theirs; the number "
              "is what *I* earn. Edit any cell."),
        mo.vstack([_head, *_rows], gap=0.25).style(
            {"overflow-x": "auto", "max-width": "100%"}),
    ])
    return


# ------------------------------------------------------------------------------
# Lattice, dynamics, run controls
# ------------------------------------------------------------------------------

@app.cell
def _(mo, sp, start_model):
    _m0 = start_model
    # default 100: below about 100x100 the Nowak & May preset fixates at 0 or 1
    # instead of the coexistence it is famous for, which is a poor first
    # impression AND a misleading one (see nowak_may() in spatial.py)
    size = mo.ui.slider(10, 150, value=(_m0.cols if _m0 else 100), step=5,
                        label="Lattice size (n x n)", show_value=True)
    hood = mo.ui.dropdown(options=list(sp.NEIGHBOURHOODS.keys()),
                          value=(_m0.neighbourhood if _m0 else "Moore (8)"),
                          label="Neighbourhood")
    wrap = mo.ui.checkbox(value=(_m0.wrap if _m0 else True),
                          label="Wrap edges (torus)")
    self_play = mo.ui.checkbox(value=(_m0.self_play if _m0 else True),
                               label="Play against yourself too")
    payoff_mode = mo.ui.radio(options=["total", "average"],
                              value=(_m0.payoff_mode if _m0 else "total"),
                              label="Payoff over neighbours", inline=True)
    lattice_controls = mo.vstack([size, hood, wrap, self_play, payoff_mode])
    return size, hood, wrap, self_play, payoff_mode, lattice_controls


@app.cell
def _(mo, sp, start_model):
    _m0 = start_model
    _rule_value = next(k for k, v in sp.RULE_LABELS.items()
                       if v == (_m0.rule if _m0 else "imitate_best"))
    rule = mo.ui.dropdown(options=sp.RULE_LABELS, value=_rule_value,
                          label="Update rule")
    schedule = mo.ui.dropdown(options=list(sp.SCHEDULES),
                              value=(_m0.schedule if _m0 else "synchronous"),
                              label="Who revises")
    update_prob = mo.ui.slider(0.05, 1.0, value=(_m0.update_prob if _m0 else 0.3),
                               step=0.05, label="...with probability",
                               show_value=True)
    beta = mo.ui.slider(0.01, 2.0, value=(_m0.beta if _m0 else 0.1), step=0.01,
                        label="Selection intensity  beta", show_value=True)
    death_rate = mo.ui.slider(0.05, 1.0, value=(_m0.death_rate if _m0 else 0.5),
                              step=0.05, label="Death rate", show_value=True)
    selection = mo.ui.slider(0.0, 1.0, value=(_m0.selection if _m0 else 1.0),
                             step=0.02, label="Selection strength  w",
                             show_value=True)
    mutation = mo.ui.slider(0.0, 0.2, value=(_m0.mutation if _m0 else 0.0),
                            step=0.005, label="Mutation rate", show_value=True)
    return rule, schedule, update_prob, beta, death_rate, selection, mutation


@app.cell
def _(beta, death_rate, mo, mutation, rule, schedule, selection, sp, update_prob):
    # show only the knobs the chosen rule actually uses, and explain the rule
    # underneath rather than inside the dropdown label
    _extra = []
    if rule.value == "fermi":
        _extra.append(beta)
    if rule.value in ("death_birth", "death_selection"):
        _extra.append(selection)
    if rule.value == "death_selection":
        _extra.append(death_rate)
    _sched = [schedule] + ([update_prob] if schedule.value == "subset" else [])
    rule_controls = mo.vstack([
        rule,
        mo.md(sp.RULE_NOTES[rule.value]).style({"font-size": "12px",
                                                "color": "#555", "margin": "0"}),
        *_extra, *_sched, mutation])
    return (rule_controls,)


@app.cell
def _(mo, np, start_seed):
    get_seed, set_seed = mo.state(int(start_seed))

    # plain name, not _roll_seed: underscore-prefixed names are cell-LOCAL in
    # marimo and cannot be handed to another cell
    def roll_seed(_):
        set_seed(int(np.random.default_rng().integers(0, 1_000_000)))
    return get_seed, set_seed, roll_seed


@app.cell
def _(mo, roll_seed):
    # THIS BUTTON MUST LIVE IN ITS OWN CELL, separate from the seed box it
    # updates. If the two share a cell, clicking re-runs that cell, and marimo
    # then PRESERVES the number's existing value rather than adopting the new
    # `value=` -- so the state changes to a fresh seed and the box keeps showing
    # the old one, silently. (Measured: state 256, widget 0. Split into two
    # cells: state 160, widget 160.) Same trap for any button that writes to a
    # widget's value.
    new_seed = mo.ui.button(label="New random seed", on_change=roll_seed)
    return (new_seed,)


@app.cell
def _(get_seed, mo, set_seed):
    # ...and the seed box gets its own cell too, reading the state
    seed = mo.ui.number(value=get_seed(), step=1, label="Random seed",
                        on_change=set_seed)
    return (seed,)


@app.cell
def _(mo, sp, start_init):
    init = mo.ui.dropdown(options=list(sp.INITIAL_CONDITIONS), value=start_init,
                          label="Starting lattice")
    patch = mo.ui.slider(1, 15, value=5, step=1, label="...patch size",
                         show_value=True)
    gens = mo.ui.slider(20, 1000, value=150, step=10, label="Generations to run",
                        show_value=True)
    return init, patch, gens


@app.cell
def _(gens, init, mo, new_seed, patch, seed):
    run_controls = mo.vstack([init] + ([patch] if init.value == "patch" else [])
                             + [mo.hstack([seed, new_seed], justify="start",
                                          gap=0.5), gens])
    return (run_controls,)


@app.cell
def _(init, mo, n_strat, names, start_probs):
    # starting proportions, one box per strategy; equal shares by default
    _even = round(100.0 / n_strat, 2)
    _vals = (start_probs if (start_probs and len(start_probs) == n_strat)
             else [_even] * n_strat)
    proportions = mo.ui.array([
        mo.ui.number(value=float(_vals[i]), start=0.0, stop=100.0, step=0.5)
        for i in range(n_strat)])
    _boxes = [mo.vstack([mo.md(f"<span style='font-size:11px'>{names[i]}</span>"),
                         proportions[i]], gap=0.1).style({"width": "104px"})
              for i in range(n_strat)]
    proportion_ui = (
        mo.vstack([mo.md("**Starting proportions (%)** — rescaled to sum to 100, "
                         "so relative sizes are what matter."),
                   mo.hstack(_boxes, justify="start", gap=0.4, wrap=True)])
        if init.value == "random" else mo.md(""))
    return proportions, proportion_ui


@app.cell
def _(lattice_controls, mo, proportion_ui, rule_controls, run_controls):
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("### Lattice"), lattice_controls]).style(
                {"min-width": "280px"}),
            mo.vstack([mo.md("### Dynamics"), rule_controls]).style(
                {"min-width": "300px"}),
            mo.vstack([mo.md("### Run"), run_controls]).style(
                {"min-width": "280px"}),
        ], justify="start", gap=2, wrap=True),
        proportion_ui,
    ], gap=1)
    return


# ------------------------------------------------------------------------------
# Build the model, run it, draw it
# ------------------------------------------------------------------------------

@app.cell
def _(beta, death_rate, hood, matrix, mutation, n_strat, np, payoff_mode, rule,
      schedule, selection, self_play, size, sp, update_prob, wrap):
    A = np.array(matrix.value, float).reshape(n_strat, n_strat)
    try:
        model = sp.Model(A=A, rows=size.value, cols=size.value,
                         neighbourhood=hood.value, wrap=wrap.value,
                         payoff_mode=payoff_mode.value, self_play=self_play.value,
                         rule=rule.value, schedule=schedule.value,
                         update_prob=update_prob.value, beta=beta.value,
                         death_rate=death_rate.value, selection=selection.value,
                         mutation=mutation.value)
        model_error = None
    except ValueError as exc:
        model, model_error = None, str(exc)
    return model, model_error


@app.cell
def _(gens, init, model, np, patch, proportions, seed, sp):
    # simulate the WHOLE run once; the generation control then scrubs through it
    probs = np.array(proportions.value, float)
    if model is None or probs.sum() <= 0:
        run, run_error = None, (None if model is not None
                                else "starting proportions cannot all be zero")
    else:
        run = sp.Sim(model, seed=int(seed.value), init=init.value,
                     probs=probs, patch=int(patch.value),
                     keep_grids=True).run(int(gens.value))
        run_error = None
    return run, run_error, probs


@app.cell
def _(mo):
    playing = mo.ui.switch(value=False, label="Play")
    ticker = mo.ui.refresh(options=["0.1s", "0.25s", "0.5s", "1s"],
                           default_interval="0.25s", label="")
    return playing, ticker


@app.cell
def _(mo):
    get_frame, set_frame = mo.state(0)
    return get_frame, set_frame


@app.cell
def _(gens, playing, set_frame, ticker):
    # auto-advance. This cell WRITES the frame counter but never reads it, which
    # is what keeps marimo from seeing a cycle.
    ticker
    if playing.value:
        set_frame(lambda t: (t + 1) % (int(gens.value) + 1))
    return


@app.cell
def _(gens, mo, set_frame):
    step_btn = mo.ui.button(label="Step",
                            on_change=lambda _: set_frame(
                                lambda t: min(t + 1, int(gens.value))))
    back_btn = mo.ui.button(label="Back",
                            on_change=lambda _: set_frame(lambda t: max(t - 1, 0)))
    end_btn = mo.ui.button(label="To end",
                           on_change=lambda _: set_frame(int(gens.value)))
    reset_btn = mo.ui.button(label="Restart", on_change=lambda _: set_frame(0))
    mo.hstack([reset_btn, back_btn, step_btn, end_btn, playing, ticker],
              justify="start", gap=0.6)
    return


@app.cell
def _(R, get_frame, mo, model_error, names, run, run_error):
    _problem = model_error or run_error
    if _problem is not None:
        view = None
        picture = mo.md(f"**That will not run.** {_problem}").callout("warn")
    else:
        view = run.at(get_frame())
        picture = R.dashboard(view, names=names)
    picture
    return (view,)


@app.cell
def _(mo, np, view):
    # the numbers, stated rather than left to be squinted off the plot
    if view is None:
        report = mo.md("")
    else:
        _s = view.snapshot()
        _r = _s["assortment"]
        _rtxt = "undefined (one strategy left)" if np.isnan(_r) else f"{_r:+.3f}"
        report = mo.md(
            f"""
            | | |
            |---|---|
            | generation | {view.generation} |
            | assortment `r` | **{_rtxt}** |
            | mean payoff | {_s['mean_payoff']:.3f} |
            | frozen? | {"yes -- nothing changed last generation" if view.frozen else "no"} |
            """
        )
    report
    return


@app.cell
def _(R, get_game, init, mo, model, names, probs, seed, sp, view):
    # take the run away with you: the data, and a link that reproduces it exactly
    if view is None:
        outputs = mo.md("")
    else:
        _cfg = sp.encode_config(model, seed=int(seed.value), init=init.value,
                                probs=probs, names=names,
                                label=get_game()["label"])
        outputs = mo.vstack([
            mo.download(data=R.series_csv(view, names).encode(),
                        filename="spatial-run.csv", label="Download the data (CSV)"),
            mo.md("**Share this exact run** -- add this to the end of the page "
                  "address, after a `?`:"),
            mo.ui.text_area(value=f"cfg={_cfg}", rows=3, full_width=True),
        ])
    outputs
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### Things worth trying

        1. **Nowak & May's result, and what it rests on.** Preset *Prisoner's
           Dilemma - Nowak & May*, Moore (8), synchronous, imitate the best, size
           100. Cooperators survive at around 30%. Now switch *Play against
           yourself* off: cooperation dies completely. Then turn it back on and
           set the lattice to 20 -- it fixates at 0 or 1 instead. The published
           result needs both the self-interaction and the big lattice.
        2. **Does the clock matter?** Same setup, change *Who revises* from
           synchronous to sequential. Huberman & Glance argued in 1993 that the
           spatial chaos depends on every cell moving in lockstep.
        3. **Where selection acts.** Run a Prisoner's Dilemma under
           *death-birth* and then under *death-selection*, with the same
           selection strength w. Same births, same deaths, same lattice -- only
           the step that selection acts on has moved. Then turn w down to 0.02
           and raise the benefit: weak selection is where the b/c > k result
           lives, and at full strength cooperation loses at every b/c.
        4. **Does meaning evolve?** Preset *Lewis signalling*, 16 strategies.
           Only two of them (`01|01` and `10|10`) are signalling systems, where
           the sender's map is one-to-one and the receiver inverts it. Does the
           lattice find one? Does it find *both*, in different regions? Nothing
           made signal 0 mean state 0 -- that has to be settled, not discovered.
        5. **Conventions with a boundary.** The *driving game*: two equally good
           conventions, nothing to choose between them. Watch what a lattice does
           that a well-mixed population cannot -- hold both at once.
        6. **Is any of this about your game?** Press *Random payoffs* a few
           times. Presets are games somebody chose because they illustrate
           something, which is a biased sample. How much of what you have seen
           survives a game nobody designed?
        7. **Read r and cross-check it.** Note the assortment r a run settles at,
           then put that number into the replicator explorer's r slider. If the
           well-mixed model with that much correlation agrees, space mattered
           only through assortment. If it does not, that gap is the finding.
        """
    )
    return


if __name__ == "__main__":
    app.run()
