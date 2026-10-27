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
limit and stored, then handed to the player (player.py), a canvas in the page
that plays, pauses and scrubs through the stored run itself. No Python runs
between generations. The older design redrew a Matplotlib figure every frame,
and swapping one PNG for the next made the picture blink between generations.
"Run 500 more" carries the same simulation on from where it ended (same random
stream), so an extended run is exactly the run a longer slider setting gives.

The payoff matrix lives in a marimo STATE rather than in the widgets, because
five different things can replace it -- picking a preset, changing the number of
strategies, rolling random payoffs, loading a shared link, or pressing "Apply
payoffs" -- and the number boxes are rebuilt from the state whenever it changes.
The boxes sit in one form, so typing changes nothing until the form is applied.

Run it locally:
    pip install marimo matplotlib numpy
    marimo run app.py            # read-only app mode (what students use)
    marimo edit app.py           # editable mode

Share with no install:
    marimo export html-wasm app.py -o build/ --mode run
    cp spatial.py render.py player.py build/    # REQUIRED, see the README
"""
import marimo

app = marimo.App(width="full")


@app.cell
def _():
    # Loading the model modules, in the browser and out of it.
    #
    # Three things go wrong if you just write `import spatial` here, and all
    # three are silent -- the page simply stays blank:
    #
    #   1. A WASM export bundles the NOTEBOOK, not the files it imports, so
    #      spatial.py, render.py and player.py are not in the build. They have to be
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
        for _name in ("spatial.py", "render.py", "player.py"):
            pathlib.Path(_name).write_text(
                pyodide.http.open_url(f"{_base}/{_name}").getvalue())
        if "." not in sys.path:
            sys.path.insert(0, ".")

    sp = importlib.import_module("spatial")
    R = importlib.import_module("render")
    P = importlib.import_module("player")
    return P, R, mo, np, sp


@app.cell
def _(mo, sp):
    # A configuration can arrive in the page URL as ?cfg=..., so a link
    # reproduces someone else's run exactly rather than approximately.
    _raw = mo.query_params().get("cfg")
    try:
        shared = sp.decode_config(_raw) if _raw else None
        # the widget offers k = 2..16 (one colour each); a bigger shared game
        # is a link made elsewhere, and the k menu would refuse to build
        if shared is not None and shared[0].k > 16:
            raise ValueError("more strategies than the widget can show")
        link_problem = None
    except Exception as _exc:
        shared = None          # a corrupted link should not break the page
        link_problem = (f"The shared link could not be opened ({_exc}), so the "
                        "default setup is shown instead.") if _raw else None
    (start_model, start_seed, start_init, start_probs, start_names, start_label,
     start_extras) = (shared if shared
                      else (None, 0, "random", None, None, None, {}))

    # How long a run can be: the slider reaches SLIDER_MAX, and "Run 500 more"
    # (MORE at a time) carries a run on up to MAX_GENERATIONS in all. The cap is
    # memory: every generation's lattice is kept, 22 KB each at 150 x 150.
    SLIDER_MAX, MORE, MAX_GENERATIONS = 2000, 500, 5000

    def clamp(value, lo, hi):
        """Fit a shared value into a control's range, or the control will not build."""
        return min(max(value, lo), hi)

    # A link made in the notebook can hold values the controls cannot show.
    # They are clamped to fit, which changes the run, so say which ones.
    _adjusted = []
    if start_model is not None:
        _m = start_model
        for _name, _val, _lo, _hi in (
                ("lattice size", _m.cols, 10, 150),
                ("update probability", _m.update_prob, 0.05, 1.0),
                ("beta", _m.beta, 0.01, 2.0),
                ("death rate", _m.death_rate, 0.05, 1.0),
                ("mutation rate", _m.mutation, 0.0, 0.2),
                ("patch size", start_extras.get("patch", 5), 1, 15),
                ("generations", start_extras.get("generations", 150), 20,
                 MAX_GENERATIONS)):
            if clamp(_val, _lo, _hi) != _val:
                _adjusted.append(f"{_name} {_val:g} -> {clamp(_val, _lo, _hi):g}")
        if not _m.is_1d and _m.rows != _m.cols:
            _adjusted.append(f"lattice {_m.rows}x{_m.cols} -> {_m.cols}x{_m.cols}")
    # a shared run longer than the slider reaches opens as the slider at its
    # top plus that many generations "more", so it replays exactly
    _total = clamp(start_extras.get("generations", 150), 20, MAX_GENERATIONS)
    start_gens = clamp(_total, 20, SLIDER_MAX)
    start_more = _total - start_gens
    if _adjusted:
        link_problem = ("This shared run had settings the controls cannot show, "
                        "so it is NOT the exact run that was shared: "
                        + "; ".join(_adjusted) + ".")
    return (MAX_GENERATIONS, MORE, SLIDER_MAX, clamp, link_problem,
            start_extras, start_gens, start_init, start_label, start_model,
            start_more, start_names, start_probs, start_seed)


@app.cell
def _(link_problem, mo):
    _intro = mo.md(
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
    ).style({"max-width": "60rem"})
    mo.vstack([_intro] + ([mo.md(link_problem).callout("warn")] if link_problem
                          else []))
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
    # are rebuilt from it whenever it changes.
    #
    # allow_self_loops: the preset, k and random-payoff controls write this
    # state from their own cell, and that cell also shows the "Currently
    # loaded" line and sets the k dropdown from it. Without self loops marimo
    # does not re-run the cell whose callback called the setter, so loading a
    # 16-strategy preset left the line reading the old game and the dropdown
    # reading k = 2. No cell calls set_game while it runs, only in callbacks,
    # so a self loop cannot cycle.
    get_game, set_game = mo.state(_init_game, allow_self_loops=True)
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

    import html as _html

    preset = mo.ui.dropdown(options=list(sp.GAMES.keys()), value=None,
                            label="Load a preset", on_change=_use_preset)
    kdim = mo.ui.dropdown(options={str(i): i for i in range(2, 17)},
                          value=str(len(get_game()["A"])),
                          label="Strategies (k)", on_change=_set_k)
    roll = mo.ui.button(label="Random payoffs (integers 0-10)", on_change=_roll)
    mo.vstack([
        mo.hstack([preset, kdim, roll], justify="start", gap=1.2),
        # the dropdown resets to "--" after loading, so state the current game
        # the label can come from a shared link, so it is escaped, not markdown
        mo.md(f"Currently loaded: **{_html.escape(str(get_game()['label']))}** "
              f"({len(get_game()['A'])} strategies)")
        .style({"font-size": "12px", "color": "#555"}),
    ], gap=0.3)
    return


@app.cell
def _(get_game, mo, np, set_game):
    # The editable payoff matrix, rebuilt from the state.
    #
    # All entries sit in ONE form, and "Apply payoffs" writes the whole matrix
    # back to the state at once. Do not go back to one live box per entry: each
    # box then triggers its own full re-simulation, and in the browser quickly
    # typed edits get lost, so the run is drawn from a stale mix of old and new
    # payoffs while the boxes show the new ones (found in the replicator tool).
    # Writing back to the state is also what keeps an edit when k is changed.
    import html as _html

    _g = get_game()
    names = _g["names"]
    n_strat = len(_g["A"])

    def _apply(v):
        if v is None:
            return
        _A = np.array([[v[f"a{i}_{j}"] for j in range(n_strat)]
                       for i in range(n_strat)], float)
        if np.array_equal(_A, _g["A"]):
            return
        _label = _g["label"]
        set_game({"A": _A, "names": names,
                  "label": _label if _label.endswith("(edited)")
                  else f"{_label} (edited)"})

    def _cell(text):
        # braces would be read as form placeholders
        return _html.escape(str(text)).replace("{", "&#123;").replace("}", "&#125;")

    _fs = "11px" if n_strat > 6 else "13px"
    _head = "".join(f"<th style='font-weight:normal;font-style:italic;"
                    f"font-size:11px;text-align:left'>{_cell(n)}</th>" for n in names)
    _rows = "".join(
        f"<tr><th style='text-align:left;font-size:{_fs};padding-right:8px;"
        f"white-space:nowrap'>{_cell(names[i])}</th>"
        + "".join(f"<td>{{a{i}_{j}}}</td>" for j in range(n_strat)) + "</tr>"
        for i in range(n_strat))
    _layout = ("<div style='overflow-x:auto;max-width:100%'><table>"
               f"<tr><th></th>{_head}</tr>{_rows}</table></div>")
    # no `step`: marimo snaps typed values to the step (1.33 -> 1.35 with 0.05)
    payoff_form = mo.Html(_layout).batch(**{
        f"a{i}_{j}": mo.ui.number(value=float(_g["A"][i][j]))
        for i in range(n_strat) for j in range(n_strat)}).form(
        submit_button_label="Apply payoffs", on_change=_apply,
        validate=lambda v: ("Fill in every payoff." if v and any(
            x is None for x in v.values()) else None))
    mo.vstack([
        mo.md("**Payoff matrix** — row = my strategy, column = theirs; the number "
              "is what *I* earn. Edit any cells, then press **Apply payoffs**."),
        payoff_form,
    ])
    return n_strat, names


# ------------------------------------------------------------------------------
# Lattice, dynamics, run controls
# ------------------------------------------------------------------------------

@app.cell
def _(clamp, mo, sp, start_model):
    _m0 = start_model
    # default 100: below about 100x100 the Nowak & May preset fixates at 0 or 1
    # instead of the coexistence it is famous for, which is a poor first
    # impression AND a misleading one (see nowak_may() in spatial.py)
    size = mo.ui.slider(10, 150, value=(clamp(_m0.cols, 10, 150) if _m0 else 100),
                        step=5, label="Lattice size (n x n)", show_value=True)
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
def _(clamp, mo, sp, start_model):
    _m0 = start_model
    _rule_value = next(k for k, v in sp.RULE_LABELS.items()
                       if v == (_m0.rule if _m0 else "imitate_best"))
    rule = mo.ui.dropdown(options=sp.RULE_LABELS, value=_rule_value,
                          label="Update rule")
    schedule = mo.ui.dropdown(options=list(sp.SCHEDULES),
                              value=(_m0.schedule if _m0 else "synchronous"),
                              label="Who revises")
    update_prob = mo.ui.slider(0.05, 1.0,
                               value=(clamp(_m0.update_prob, 0.05, 1.0) if _m0 else 0.3),
                               step=0.05, label="...with probability",
                               show_value=True)
    beta = mo.ui.slider(0.01, 2.0, value=(clamp(_m0.beta, 0.01, 2.0) if _m0 else 0.1),
                        step=0.01,
                        label="Selection intensity  beta", show_value=True)
    death_rate = mo.ui.slider(0.05, 1.0,
                              value=(clamp(_m0.death_rate, 0.05, 1.0) if _m0 else 0.5),
                              step=0.05, label="Death rate", show_value=True)
    selection = mo.ui.slider(0.0, 1.0, value=(_m0.selection if _m0 else 1.0),
                             step=0.02, label="Selection strength  w",
                             show_value=True)
    mutation = mo.ui.slider(0.0, 0.2, value=(clamp(_m0.mutation, 0.0, 0.2) if _m0 else 0.0),
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
    if schedule.value == "sequential":
        _sched.append(mo.md(
            "Sequential is the slow schedule: on a 100x100 lattice about half a "
            "minute per 150 generations with Moore (8), and several minutes with a "
            "radius-2 neighbourhood.")
            .style({"font-size": "12px", "color": "#555", "margin": "0"}))
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
    return get_seed, roll_seed, set_seed


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
    # a cleared box gives None; keep the last seed rather than crash the run
    seed = mo.ui.number(value=get_seed(), step=1, label="Random seed",
                        on_change=lambda v: set_seed(int(v)) if v is not None else None)
    return (seed,)


@app.cell
def _(SLIDER_MAX, clamp, mo, sp, start_extras, start_gens, start_init):
    init = mo.ui.dropdown(options=list(sp.INITIAL_CONDITIONS), value=start_init,
                          label="Starting lattice")
    patch = mo.ui.slider(1, 15, value=clamp(start_extras.get("patch", 5), 1, 15),
                         step=1, label="...patch size", show_value=True)
    gens = mo.ui.slider(20, SLIDER_MAX, value=start_gens,
                        step=10, label="Generations to run", show_value=True)
    return gens, init, patch


@app.cell
def _(gens, init, mo, new_seed, patch, seed):
    run_controls = mo.vstack([init] + ([patch] if init.value == "patch" else [])
                             + [mo.hstack([seed, new_seed], justify="start",
                                          gap=0.5), gens])
    return (run_controls,)


@app.cell
def _(clamp, init, mo, n_strat, names, start_probs):
    # Starting proportions, one box per strategy, equal shares by default. One
    # form for the same reason as the payoff matrix; until it is applied the
    # defaults (or a shared link's values) are used.
    import html as _html

    _even = round(100.0 / n_strat, 2)
    start_shares = ([clamp(float(x), 0.0, 100.0) for x in start_probs]
                    if (start_probs and len(start_probs) == n_strat)
                    else [_even] * n_strat)
    _boxes = "".join(
        "<div style='display:inline-block;margin:0 10px 6px 0;vertical-align:top'>"
        f"<div style='font-size:11px'>{_html.escape(str(names[i])).replace('{', '&#123;').replace('}', '&#125;')}</div>"
        f"{{p{i}}}</div>" for i in range(n_strat))
    proportion_form = mo.Html(f"<div>{_boxes}</div>").batch(**{
        f"p{i}": mo.ui.number(value=float(start_shares[i]), start=0.0, stop=100.0)
        for i in range(n_strat)}).form(
        submit_button_label="Apply proportions",
        validate=lambda v: ("Fill in every proportion." if v and any(
            x is None for x in v.values()) else None))
    proportion_ui = (
        mo.vstack([mo.md("**Starting proportions (%)** — rescaled to sum to 100, "
                         "so relative sizes are what matter."), proportion_form])
        if init.value == "random" else mo.md(""))
    return proportion_form, proportion_ui, start_shares


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
def _(beta, death_rate, get_game, hood, mutation, payoff_mode, rule, schedule,
      selection, self_play, size, sp, update_prob, wrap):
    A = get_game()["A"]
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
def _(mo, start_more):
    # Generations added by "Run 500 more", and the setup they were added to
    # ("key", see below). None is the setup the page opened with, so a shared
    # link to a long run gets its extra generations on the first run.
    get_more, set_more = mo.state({"key": None, "extra": int(start_more)})
    # The last run, kept between re-runs of the run cell so that "Run 500 more"
    # carries it on from where it ended instead of re-running from generation
    # 0. Changed in place by that cell; nothing reacts to it.
    run_cache = {}
    return get_more, run_cache, set_more


@app.cell
def _(MAX_GENERATIONS, get_more, get_seed, gens, init, mo, model, n_strat, np,
      patch, proportion_form, run_cache, seed, sp, start_shares):
    # simulate the WHOLE run once; the player then plays and scrubs through it
    # (a cleared seed box reads None: use the last seed it held)
    run_seed = int(seed.value) if seed.value is not None else int(get_seed())
    _v = proportion_form.value
    probs = (np.array([_v[f"p{i}"] for i in range(n_strat)], float) if _v
             else np.array(start_shares, float))
    run_key, run_from = None, 0
    if model is None:
        run, run_error = None, None
    elif init.value == "random" and probs.sum() <= 0:
        run, run_error = None, "starting proportions cannot all be zero"
    else:
        # Everything that decides the run, as one string. Extra generations
        # belong to the setup they were added to: change any control and they
        # are dropped, and they stay dropped if the control is put back.
        run_key = sp.encode_config(model, seed=run_seed, init=init.value,
                                   probs=probs, patch=int(patch.value),
                                   generations=int(gens.value))
        run_cache.setdefault("first_key", run_key)
        _more = get_more()
        _owner = run_cache["first_key"] if _more["key"] is None else _more["key"]
        if _owner != run_key:
            run_cache["spent"] = _more
        _extra = (_more["extra"] if _owner == run_key
                  and run_cache.get("spent") is not _more else 0)
        _total = min(int(gens.value) + _extra, MAX_GENERATIONS)
        _old = run_cache.get("sim") if run_cache.get("key") == run_key else None
        if _old is not None and _old.generation <= _total:
            # the same Sim, so the same random stream: carrying a run on gives
            # exactly the run a longer setting would have given from the start
            run = _old
            run_from = run.generation if run.generation < _total else 0
        else:
            run = sp.Sim(model, seed=run_seed, init=init.value,
                         probs=probs, patch=int(patch.value), keep_grids=True)
        if run.generation < _total:
            with mo.status.spinner(
                    title=f"Running generations {run.generation + 1}-{_total}..."):
                run.run(_total - run.generation)
        run_cache.update(key=run_key, sim=run)
        run_error = None
    return probs, run, run_error, run_from, run_key, run_seed


@app.cell
def _(P, mo, model_error, names, run, run_error, run_from):
    # The run, played in the page by player.py: one canvas repainted in place,
    # so nothing blinks between generations. Play / Step / the slider / the
    # space and arrow keys all work inside it without calling back into Python.
    _problem = model_error or run_error
    if _problem is not None:
        player_page = None
        picture = mo.md(f"**That will not run.** {_problem}").callout("warn")
    else:
        # fit_player keeps the page under marimo's 8 MB output limit by
        # thinning generations on very long, badly compressing runs;
        # P.iframe_html (inside it), not mo.iframe: see its docstring.
        # After "Run 500 more" it opens where the run had ended and plays on.
        player_page, _tag, _stride = P.fit_player(run, names=names,
                                                  start=run_from,
                                                  autoplay=run_from > 0)
        picture = mo.Html(_tag)
    picture
    return (player_page,)


@app.cell
def _(MAX_GENERATIONS, MORE, gens, mo, player_page, run, run_key, set_more):
    # "Run 500 more": carry this run on from where it ended. In its own cell,
    # under the player, like the seed button; it writes the state the run cell
    # reads, and the run cell extends the stored Sim rather than starting over.
    if player_page is None:
        more_ui = mo.md("")
    else:
        _total = run.generation
        _next = min(_total + MORE, MAX_GENERATIONS)
        more = mo.ui.button(
            # near the cap the last step is shorter; at the cap it is disabled
            label=f"Run {_next - _total or MORE} more generations",
            disabled=_next <= _total,
            on_change=lambda _: set_more({"key": run_key,
                                          "extra": _next - int(gens.value)}))
        _note = (f"{_total} generations so far. "
                 + (f"A run can go to {MAX_GENERATIONS} in all."
                    if _next > _total else
                    f"That is the most one run can go ({MAX_GENERATIONS})."))
        more_ui = mo.hstack([more, mo.md(_note).style(
            {"font-size": "12px", "color": "#555"})], justify="start", gap=0.8,
            align="center")
    more_ui
    return


@app.cell
def _(P, R, get_game, init, mo, model, names, patch, player_page, probs, run,
      run_seed, sp):
    # take the run away with you: the data, the animation, and a link that
    # reproduces it exactly
    if player_page is None:
        outputs = mo.md("")
    else:
        _cfg = sp.encode_config(model, seed=run_seed, init=init.value,
                                probs=probs, names=names,
                                label=get_game()["label"], patch=int(patch.value),
                                # the whole run, extensions included
                                generations=int(run.generation))
        outputs = mo.vstack([
            mo.hstack([
                # the whole run, every generation (not just up to where it is paused)
                mo.download(data=lambda: R.series_csv(run, names).encode(),
                            filename="spatial-run.csv", mimetype="text/csv",
                            label="Download the data (CSV)"),
                # lazy, like the CSV: a long run's page is megabytes, and an eager
                # download would ship it to the browser a second time every run.
                # Every generation, even when the player above had to thin them.
                mo.download(data=lambda: P.player_html(run, names=names).encode(),
                            filename="spatial-run.html",
                            mimetype="text/html",
                            label="Download the animation (HTML, plays offline)"),
            ], justify="start", gap=0.8),
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
           synchronous to sequential (it takes longer to compute). Huberman &
           Glance argued in 1993 that the spatial chaos depends on every cell
           moving in lockstep.
        3. **Where selection acts.** Switch *Play against yourself* off first:
           it pays every cooperator an extra b - c, and with it on and a high
           benefit (b = 8, c = 1) cooperation wins even at full strength, the
           opposite of the published pattern. Then run
           the *Prisoner's Dilemma (b=1.9, c=1)* under *death-birth* and then
           under *death-selection*, with the same selection strength w. Same
           births, same deaths, same lattice -- only the step that selection acts
           on has moved. Then turn w down to 0.02 and raise the benefit b (top
           row is b - c, -c; bottom row is b, 0): weak selection is where the
           b/c > k result lives, and at full strength cooperation loses at every
           b/c.
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
    ).style({"max-width": "60rem"})
    return


if __name__ == "__main__":
    app.run()
