"""
spatial.py -- the model core for the PHIL 2001 spatial-evolution explorer.
================================================================================

This single file is the whole model. Every front-end (the Colab notebook, the
marimo widget, the drawing helpers) imports from here, so there is exactly ONE
place where the dynamics live. If you want to understand or change how the model
behaves, this is the file to read and edit.

It has no dependencies beyond NumPy.

------------------------------------------------------------------------------
What a spatial evolutionary model says, in one sentence
------------------------------------------------------------------------------
Players sit at fixed positions on a lattice, play a game with their NEIGHBOURS
(not with the population at large), and then revise their strategy by looking at
how their neighbours did. That is the whole idea. Everything below is bookkeeping
for that sentence.

The one thing space buys you is ASSORTMENT: neighbours of a cooperator are
disproportionately cooperators, because clusters of like types stay together.
Space is not an independent lever on prosociality -- it is a GENERATOR of
assortment, and assortment is the lever. This tool measures the assortment it
generates (`assortment()`, section 6), so you can take that number over to the
replicator explorer's `r` slider and ask whether the well-mixed model with that
much correlation predicts what the lattice actually did. When it does, you have
learned that space mattered only through assortment. When it does not, the
difference is the interesting part.

------------------------------------------------------------------------------
The four choices that define a spatial model
------------------------------------------------------------------------------
Every spatial model makes four decisions, and every one of them can change the
answer. They are separate controls here, on purpose:

    1. WHO PLAYS WHOM   the neighbourhood (section 1)
    2. WHAT THEY EARN   total vs average payoff, self-interaction (section 2)
    3. HOW THEY REVISE  the update rule (section 3)
    4. WHEN THEY REVISE the schedule: all at once, or one at a time (section 4)

Choice 4 is the one people forget. Nowak & May's (1992) famous "spatial chaos"
in the Prisoner's Dilemma turned out to depend on every cell updating in lockstep;
Huberman & Glance (PNAS 1993) showed that updating cells one at a time makes the
dynamic patterns collapse. A conclusion that survives a change of schedule is
about the game. One that does not is about the clock.

------------------------------------------------------------------------------
MAP OF THIS FILE  (sections are banner-commented below)
------------------------------------------------------------------------------
    1. Geometry              lattices, neighbourhoods, shifting the grid
    2. Payoffs               what each cell earns from its neighbours
    3. Update rules          the five ways a cell can revise its strategy
    4. Scheduling            synchronous / random subset / sequential; mutation
    5. Model and Sim         the objects you actually use
    6. Measurement           frequencies, realized assortment, frozen states
    7. Initial conditions    random, patch, single mutant, stripes
    8. A library of games    Prisoner's Dilemma, Stag Hunt, Hawk-Dove, ...

Everything random goes through ONE seeded generator (`Sim.rng`), so a run is
completely reproducible from its seed. That matters: "run it a few times and see
if the result is reliable" is only a meaningful instruction if you can also say
exactly which runs you did.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np

# ------------------------------------------------------------------------------
# 1. Geometry: lattices, neighbourhoods, shifting the grid
# ------------------------------------------------------------------------------
#
# A population is an integer array `grid` of shape (rows, cols); grid[r, c] is
# the index of the strategy played at that site. A ONE-DIMENSIONAL lattice is
# just a grid with one row, shape (1, N) -- that way there is a single code path
# for 1D and 2D, and the only difference is which neighbour offsets we use.
#
# A neighbourhood is a list of OFFSETS (dr, dc), never including (0, 0): a cell
# is not its own neighbour. (Whether it also plays a game against itself is a
# separate choice -- see `self_play` in section 2.)

def neighbour_offsets(kind, radius=1):
    """The list of (dr, dc) offsets defining a neighbourhood.

    kind = "line"   1D: the `radius` sites to the left and right (2*radius total)
    kind = "vn"     von Neumann: |dr| + |dc| <= radius   (radius 1 -> 4 neighbours)
    kind = "moore"  Moore:       max(|dr|, |dc|) <= radius (radius 1 -> 8, radius 2 -> 24)
    """
    radius = int(radius)
    if radius < 1:
        raise ValueError("radius must be at least 1")
    if kind not in ("line", "vn", "moore"):
        raise ValueError(f"unknown neighbourhood kind {kind!r}")
    if kind == "line":
        return [(0, d) for d in range(-radius, radius + 1) if d != 0]
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if kind == "vn" and abs(dr) + abs(dc) > radius:
                continue
            out.append((dr, dc))
    return out


# The menu the front-ends show. Name -> (kind, radius).
NEIGHBOURHOODS = {
    "von Neumann (4)": ("vn", 1),
    "Moore (8)": ("moore", 1),
    "von Neumann r=2 (12)": ("vn", 2),
    "Moore r=2 (24)": ("moore", 2),
    "1D line, 2 neighbours": ("line", 1),
    "1D line, 4 neighbours": ("line", 2),
    "1D line, 8 neighbours": ("line", 4),
}


def _shift(a, dr, dc, wrap):
    """Look up the neighbour at offset (dr, dc) for every site at once.

    Returns (values, valid) where

        values[..., r, c] == a[..., r + dr, c + dc]     (where that site exists)
        valid[r, c]       == True if that site exists

    The shift acts on the LAST TWO axes, so `a` may carry leading axes (we use
    that to shift a whole stack of arrays in one call).

    On a WRAPPED lattice (a torus) every site exists, so `valid` is all True and
    the lookup is a roll. On a BOUNDED lattice the sites off the edge simply do
    not exist -- they are marked invalid, NOT given a payoff of zero. That
    distinction is the single most common bug in hand-rolled spatial models: if
    off-grid neighbours contribute 0 to a total, edge cells are quietly compared
    against interior cells that have more neighbours, and with negative payoffs
    the edge becomes the best place on the board. Here an absent neighbour is
    absent, and section 2 lets you divide by the number of neighbours you
    actually have.
    """
    if wrap:
        return np.roll(a, shift=(-dr, -dc), axis=(-2, -1)), np.ones(a.shape[-2:], bool)

    rows, cols = a.shape[-2:]
    values = np.zeros_like(a)
    valid = np.zeros((rows, cols), bool)
    # source window (where we read) and destination window (where it lands)
    sr0, sr1 = max(0, dr), rows + min(0, dr)
    sc0, sc1 = max(0, dc), cols + min(0, dc)
    dr0, dr1 = max(0, -dr), rows + min(0, -dr)
    dc0, dc1 = max(0, -dc), cols + min(0, -dc)
    if sr0 < sr1 and sc0 < sc1:
        values[..., dr0:dr1, dc0:dc1] = a[..., sr0:sr1, sc0:sc1]
        valid[dr0:dr1, dc0:dc1] = True
    return values, valid


def neighbour_stack(a, offsets, wrap):
    """Every neighbour's value, stacked.

    Returns (values, valid), both with shape (len(offsets),) + a.shape:
    values[m, r, c] is the value held by neighbour m of site (r, c).
    """
    vals, valids = [], []
    for dr, dc in offsets:
        v, ok = _shift(a, dr, dc, wrap)
        vals.append(v)
        valids.append(ok)
    return np.stack(vals), np.stack(valids)


# ------------------------------------------------------------------------------
# 2. Payoffs: what each cell earns from its neighbours
# ------------------------------------------------------------------------------
#
# The game is SYMMETRIC and given by one k x k matrix A: a player using strategy
# i against a player using strategy j earns A[i, j]. A cell plays one round
# against each of its neighbours and adds up what it gets.
#
# Two options change what "adds up" means:
#
#   payoff_mode="total"    sum over neighbours (the classic choice)
#   payoff_mode="average"  divide by the number of neighbours
#
# On a wrapped lattice every cell has the same number of neighbours, so the two
# differ only by a constant factor -- and every update rule here is invariant to
# a positive rescaling, so THE CHOICE DOES NOT MATTER on a torus. On a BOUNDED
# lattice it matters a great deal: with "total", a corner cell with 3 neighbours
# is being compared to an interior cell with 8, and the comparison is mostly
# measuring the boundary. Use "average" when the lattice has edges. (This is the
# measurement point in miniature: a number means nothing until you say what it
# is a number *of*.)
#
#   self_play=True         the cell also plays one round against itself
#
# Some presentations of the spatial Prisoner's Dilemma include self-interaction
# and some do not; it shifts each cell's payoff by A[i, i], which is not a
# constant across cells, so it can change outcomes. Off by default.

def payoff_field(grid, A, offsets, wrap, payoff_mode="total", self_play=False):
    """Payoffs for every site, for every strategy it *might* have played.

    Returns (payoff_all, payoff, neighbour_strategies, neighbour_valid):

        payoff_all[s, r, c]   what site (r, c) would earn playing strategy s
                              against its CURRENT neighbours
        payoff[r, c]          what it actually earns  = payoff_all[grid[r,c], r, c]
        neighbour_strategies  (m, rows, cols) the strategy of each neighbour
        neighbour_valid       (m, rows, cols) which of those neighbours exist

    Computing the counterfactual payoffs for all k strategies costs almost
    nothing extra and is exactly what the best-response rule needs, so we always
    return it.
    """
    A = np.asarray(A, float)
    k = A.shape[0]
    nb_strat, nb_valid = neighbour_stack(grid, offsets, wrap)

    # counts[j, r, c] = how many of my neighbours play strategy j
    counts = np.stack([((nb_strat == j) & nb_valid).sum(axis=0) for j in range(k)]
                      ).astype(float)
    n_partners = nb_valid.sum(axis=0).astype(float)

    # payoff_all[s] = sum_j A[s, j] * counts[j]
    payoff_all = np.tensordot(A, counts, axes=([1], [0]))

    if self_play:
        # one extra round against a copy of yourself: strategy s meets s
        payoff_all = payoff_all + np.diag(A)[:, None, None]
        n_partners = n_partners + 1.0

    if payoff_mode == "average":
        payoff_all = payoff_all / np.maximum(n_partners, 1.0)
    elif payoff_mode != "total":
        raise ValueError("payoff_mode must be 'total' or 'average'")

    payoff = np.take_along_axis(payoff_all, grid[None], axis=0)[0]
    return payoff_all, payoff, nb_strat, nb_valid


# ------------------------------------------------------------------------------
# 3. Update rules: the five ways a cell can revise its strategy
# ------------------------------------------------------------------------------
#
# A rule takes the current grid plus the payoffs computed in section 2 and
# returns a PROPOSED new strategy for every site. (Section 4 decides which of
# those proposals actually take effect.) They are grouped by what they assume
# about the players, and the grouping is itself the lesson -- a rule is a
# behavioural hypothesis, not a technicality:
#
#   IMITATION, deterministic  -- cultural: nobody dies, people copy
#     imitate_best        copy whoever did best nearby, including yourself
#                         ("unconditional imitation"; the Nowak-May rule)
#   OPTIMISATION
#     best_response       play what would have been best against the neighbours
#                         you just faced (myopic best response). This one is
#                         INNOVATIVE: it can introduce a strategy nobody nearby
#                         is using. Every other rule here can only copy.
#   IMITATION, stochastic
#     imitate_prob        copy someone nearby with probability rising in payoff
#     fermi               compare with ONE random neighbour; switch with a
#                         logistic probability in the payoff difference
#     proportional_imit   compare with ONE random neighbour; switch with
#                         probability proportional to how much better they did
#   BIRTH AND DEATH  -- biological: individuals die and are replaced
#     death_birth         a site dies; the neighbours compete to fill it, with
#                         success proportional to their payoff (selection acts
#                         on BIRTH)
#     death_selection     doing badly is what kills you; the empty site is then
#                         filled by a random neighbour (selection acts on DEATH)
#
# `proportional_imit` is the one to know for continuity with the other tool: its
# mean-field limit (well-mixed, large population) IS the replicator dynamic. It
# is the microscopic story behind the curves in the replicator explorer, which
# makes the two directly comparable -- same dynamic, one with space and one
# without.
#
# The last two are the point of contact with evolutionary graph theory, and they
# are deliberately an ADVERSARIAL PAIR: same births, same deaths, same lattice,
# with selection moved from one step to the other. If a result survives that
# swap it is about the game; if it does not, it is about where you put selection
# -- and this is a case where it genuinely does not. See their docstrings.
#
# ------------------------------------------------------------------------------
# A property every rule here has, and GenLab's did not: MEANINGFULNESS
# ------------------------------------------------------------------------------
# Payoffs are an interval scale. Only differences and ratios OF differences carry
# meaning; the zero point and the unit are conventions. So any rule worth using
# must give the same answer when payoffs are put through a positive affine
# transformation, payoff -> alpha * payoff + c with alpha > 0. Otherwise the
# model's conclusions are partly about how you happened to write the numbers.
#
# All seven rules below are invariant to that transformation, with ONE stated
# exception: `fermi` is invariant to the shift c but not to the scale alpha,
# because its beta carries units of 1/payoff -- rescaling payoffs by alpha is
# exactly the same as rescaling beta by alpha, and beta is a knob you set. That
# is a property of the rule, not a bug, but it does mean "beta = 1" has no
# meaning until you say what a unit of payoff is.
#
# The original GenLab probabilistic-imitation rule failed this: it drew a
# neighbour with probability payoff_j / sum(payoffs), a RATIO of payoffs, which
# is not meaningful on an interval scale -- and which divides by zero when the
# payoffs sum to zero and inverts itself when they are negative. `imitate_prob`
# below fixes it by measuring every payoff from the worst in the neighbourhood,
# so only differences ever enter. (tests.py checks all of this.)

def _argmax_random(values, rng, valid=None):
    """Index of the maximum along axis 0, with ties broken UNIFORMLY at random.

    Tie-breaking matters more than it sounds. Early on, whole regions of the
    lattice are exactly tied; a deterministic tie-break (say, "lowest index
    wins") silently favours one strategy and can decide the run.
    """
    v = values if valid is None else np.where(valid, values, -np.inf)
    is_best = v == v.max(axis=0, keepdims=True)
    keys = np.where(is_best, rng.random(v.shape), -1.0)
    return keys.argmax(axis=0)


def _pick_random_valid(valid, rng):
    """For every site, the index of one uniformly-chosen existing neighbour."""
    return np.where(valid, rng.random(valid.shape), -1.0).argmax(axis=0)


def _roulette(weights, valid, rng):
    """Index sampled with probability proportional to `weights` (>= 0)."""
    w = np.where(valid, np.maximum(weights, 0.0), 0.0)
    total = w.sum(axis=0)
    # all-zero weights (everyone tied at the bottom): fall back to uniform
    w = np.where(total[None] > 0, w, valid.astype(float))
    total = w.sum(axis=0)
    u = rng.random(total.shape) * total
    idx = (np.cumsum(w, axis=0) < u[None]).sum(axis=0)
    return np.clip(idx, 0, w.shape[0] - 1)


def _closed_stacks(grid, payoff, offsets, wrap):
    """Self + neighbours, as stacks. Index 0 is always the cell itself."""
    nb_pay, nb_valid = neighbour_stack(payoff, offsets, wrap)
    nb_strat, _ = neighbour_stack(grid, offsets, wrap)
    pay = np.concatenate([payoff[None], nb_pay])
    strat = np.concatenate([grid[None], nb_strat])
    valid = np.concatenate([np.ones((1,) + grid.shape, bool), nb_valid])
    return pay, strat, valid


def rule_imitate_best(grid, pf, model, rng):
    """Copy the highest-paid player in the closed neighbourhood (self included)."""
    _, payoff, _, _ = pf
    pay, strat, valid = _closed_stacks(grid, payoff, model.offsets, model.wrap)
    pick = _argmax_random(pay, rng, valid)
    return np.take_along_axis(strat, pick[None], axis=0)[0]


def rule_best_response(grid, pf, model, rng):
    """Play the strategy that would have earned most against these neighbours.

    Myopic: it assumes the neighbours will not move, which they will. Note this
    is the only rule here that can introduce a strategy that has gone locally
    extinct, so it is the only one whose rest points are (local) Nash equilibria
    rather than "nobody nearby is doing better".
    """
    payoff_all, _, _, _ = pf
    return _argmax_random(payoff_all, rng)


def rule_imitate_prob(grid, pf, model, rng):
    """Copy someone in the closed neighbourhood with probability ~ their payoff.

    Payoffs can be negative and probabilities cannot, so every payoff is
    measured from the WORST payoff in the neighbourhood: your weight is how much
    better than the local floor you did. That is what makes the rule meaningful
    on an interval scale (see the note at the top of this section) -- the naive
    version, weight = payoff_j / sum of payoffs, is a ratio of interval-scale
    quantities and is not.

    The reference point is still a convention: measuring from the local worst is
    not the only choice, and a different choice is a different rule. There is no
    scale-free canonical form of "copy in proportion to success". If you want a
    rule with no such choice in it, use `fermi` or `proportional_imit`, which
    only ever look at the difference between two players.
    """
    _, payoff, _, _ = pf
    pay, strat, valid = _closed_stacks(grid, payoff, model.offsets, model.wrap)
    floor = np.where(valid, pay, np.inf).min(axis=0, keepdims=True)
    pick = _roulette(pay - floor, valid, rng)
    return np.take_along_axis(strat, pick[None], axis=0)[0]


def rule_fermi(grid, pf, model, rng):
    """Pairwise comparison with one random neighbour (Fermi / logistic rule).

    Adopt neighbour j's strategy with probability

        p = 1 / (1 + exp(-beta * (payoff_j - payoff_i)))

    beta is the INTENSITY OF SELECTION: beta -> 0 is a random walk (payoffs
    ignored), beta -> infinity is "always copy anyone who did better". The
    standard stochastic rule in the statistical-physics literature on spatial
    games (Szabo & Fath 2007). Handles negative payoffs, and never gets stuck
    the way a deterministic rule does.

    MIND THE UNITS. beta multiplies a payoff difference, so it carries units of
    1/payoff: it is the only quantity in this file that is not invariant to
    rescaling payoffs, and "beta = 1" means nothing until you say what a unit of
    payoff is. It is easy to be fooled here. With payoff_mode="total", 8
    neighbours and payoffs of order 1, the typical difference between two
    neighbours is about 8, so beta = 1 puts the logistic at 0.9997 -- you think
    you are running a stochastic rule and you are running a deterministic one.
    Hence the default beta = 0.1. If you want beta to be comparable to the
    numbers in your payoff matrix, set payoff_mode="average".
    """
    _, payoff, _, _ = pf
    nb_pay, nb_valid = neighbour_stack(payoff, model.offsets, model.wrap)
    nb_strat, _ = neighbour_stack(grid, model.offsets, model.wrap)
    pick = _pick_random_valid(nb_valid, rng)
    other_pay = np.take_along_axis(nb_pay, pick[None], axis=0)[0]
    other_strat = np.take_along_axis(nb_strat, pick[None], axis=0)[0]
    # clipped to keep exp() finite at large beta; the clip is far outside the
    # region where p differs from 0 or 1 in double precision
    z = np.clip(model.beta * (other_pay - payoff), -500, 500)
    p = 1.0 / (1.0 + np.exp(-z))
    switch = rng.random(grid.shape) < p
    return np.where(switch, other_strat, grid)


def rule_proportional_imit(grid, pf, model, rng):
    """Compare with one random neighbour; copy with probability ~ the gap.

    Adopt neighbour j's strategy with probability max(0, payoff_j - payoff_i)/D,
    where D is the largest payoff difference the game can produce, so that the
    probability is well defined. Never copies someone who did worse.

    This is the rule whose well-mixed limit is the REPLICATOR DYNAMIC. Run it
    here on a lattice, run the replicator explorer on the same payoff matrix,
    and the difference between the two is exactly what space contributed.
    """
    _, payoff, _, _ = pf
    nb_pay, nb_valid = neighbour_stack(payoff, model.offsets, model.wrap)
    nb_strat, _ = neighbour_stack(grid, model.offsets, model.wrap)
    pick = _pick_random_valid(nb_valid, rng)
    other_pay = np.take_along_axis(nb_pay, pick[None], axis=0)[0]
    other_strat = np.take_along_axis(nb_strat, pick[None], axis=0)[0]
    p = np.clip((other_pay - payoff) / model.max_payoff_gap(), 0.0, 1.0)
    switch = rng.random(grid.shape) < p
    return np.where(switch, other_strat, grid)


def rule_death_birth(grid, pf, model, rng):
    """A site dies; its neighbours compete to fill it in proportion to payoff.

    SELECTION ACTS ON BIRTH. Who dies is decided by the schedule (section 4) and
    has nothing to do with how well anyone did -- death is blind. What happens
    next is not: the vacancy is filled by a copy of one of the dead site's
    neighbours, chosen with probability rising in that neighbour's payoff
    (measured from the worst neighbour, for the interval-scale reason above).

    Note the OPEN neighbourhood: the occupant is dead, so it does not compete to
    replace itself. That one detail is the whole difference between this rule and
    `imitate_prob`, which keeps the incumbent in the running -- and it is enough
    to change which strategies win, because it removes the incumbency advantage
    that makes imitation rules sticky.

    `selection` (w) sets HOW MUCH payoff matters: a competitor's chance is
    proportional to (1 - w) + w * standing, where standing is 1 for the best
    neighbour and 0 for the worst. w = 1 is full-strength selection (chance
    proportional to how far above the local floor you are); w = 0 is neutral
    drift, where the winner is picked at random and payoffs are ignored
    entirely. Everything in between is weak selection.

    That knob is not decoration. This is the death-birth (DB) updating of
    evolutionary graph theory, and Ohtsuki, Hauert, Lieberman & Nowak (Nature
    441, 2006) show that under DB updating on a regular graph of degree k,
    cooperation in a donation game is favoured when roughly b/c > k -- a
    startlingly simple condition. Their result is a WEAK-SELECTION one, and it
    does not survive being run at w = 1. Measured here (60x60 torus, Moore-8 so
    k = 8, random 50/50 start, cooperator frequency at generation 400, 4 seeds):

        b/c     w = 1 (full)      w = 0.02 (weak)
          2     0.00 every run    0.03 0.02 0.01 0.03
          4     0.00 every run    0.07 0.11 0.03 0.04
          8     0.00 every run    0.18 0.10 0.17 0.13
         12     0.00 every run    0.11 0.22 0.24 0.29
         20     0.00 0.00 0.00 0.07   0.17 0.24 0.30 0.36

    At full strength the highest-paid neighbour nearly always wins the empty
    site, the rule collapses towards "imitate the best", and defection takes
    everything at every b/c. Turn selection down and cooperation appears and
    rises with b/c, as the condition would suggest.

    DO NOT READ THAT TABLE AS A REPLICATION. Ohtsuki et al.'s claim is about a
    FIXATION PROBABILITY -- a single cooperative mutant fixes more often than a
    neutral one, ρ_C > 1/N -- and what is tabulated above is a frequency at
    generation 400 from a mixed start, which is a different quantity and, at
    weak selection with no mutation, still a transient. The tool is showing you
    something consistent with the condition, not a test of it. Testing it
    properly means many single-mutant runs carried to fixation. Worth doing;
    not done here.
    """
    _, payoff, _, _ = pf
    nb_pay, nb_valid = neighbour_stack(payoff, model.offsets, model.wrap)
    nb_strat, _ = neighbour_stack(grid, model.offsets, model.wrap)
    # each competitor's standing among the competitors: 1 = best, 0 = worst
    lo = np.where(nb_valid, nb_pay, np.inf).min(axis=0, keepdims=True)
    hi = np.where(nb_valid, nb_pay, -np.inf).max(axis=0, keepdims=True)
    span = hi - lo
    standing = np.where(span > 0, (nb_pay - lo) / np.where(span > 0, span, 1.0), 1.0)
    w = model.selection
    pick = _roulette((1.0 - w) + w * standing, nb_valid, rng)
    return np.take_along_axis(nb_strat, pick[None], axis=0)[0]


def rule_death_selection(grid, pf, model, rng):
    """Doing badly is what kills you; a random neighbour then fills the gap.

    SELECTION ACTS ON DEATH -- the mirror image of `death_birth`. A site's
    probability of dying is

        p = death_rate * (1 - selection * standing)

    where `standing` is where its payoff sits between the worst and the best in
    its own closed neighbourhood: 1 for the local best, 0 for the local worst.
    The two knobs separate cleanly: `death_rate` is how fast the population
    turns over at all, and `selection` (w) is how much of that turnover payoffs
    are allowed to explain. At w = 1 the local best never dies and the local
    worst dies with probability `death_rate`; at w = 0 everyone dies at
    `death_rate` regardless of payoff, which is pure drift. This mirrors the
    same knob in `death_birth`, so the two rules can be compared at matched
    selection strength -- which is the only way the comparison means anything.

    Because standing is built from differences and a ratio of differences, the
    rule is invariant to rescaling payoffs, and it is well defined when every
    payoff is negative -- which matters for the spite game, where they are.

    When the whole neighbourhood is tied there is nothing for selection to act
    on and nobody dies. So a monomorphic lattice is absorbing, as it should be.

    Then the empty site is filled by a copy of a UNIFORMLY RANDOM neighbour --
    birth is blind here.

    Why have both this and `death_birth`? Because the pair is a controlled
    experiment on a modelling choice that usually goes unexamined. The births,
    the deaths, the lattice and the game are the same; only the step that
    selection acts on has moved. If cooperation survives one and not the other,
    then "space favours cooperation" was never the right description of what was
    happening -- the finding belongs to the update rule as much as to the
    structure. Run the same donation game under both and compare.
    """
    _, payoff, _, _ = pf
    pay_c, _, valid_c = _closed_stacks(grid, payoff, model.offsets, model.wrap)
    lo = np.where(valid_c, pay_c, np.inf).min(axis=0)
    hi = np.where(valid_c, pay_c, -np.inf).max(axis=0)
    span = hi - lo
    standing = np.where(span > 0, (payoff - lo) / np.where(span > 0, span, 1.0), 1.0)
    dies = rng.random(grid.shape) < model.death_rate * (1.0 - model.selection * standing)

    nb_strat, nb_valid = neighbour_stack(grid, model.offsets, model.wrap)
    pick = _pick_random_valid(nb_valid, rng)
    replacement = np.take_along_axis(nb_strat, pick[None], axis=0)[0]
    return np.where(dies, replacement, grid)


RULES = {
    "imitate_best": rule_imitate_best,
    "best_response": rule_best_response,
    "imitate_prob": rule_imitate_prob,
    "fermi": rule_fermi,
    "proportional_imit": rule_proportional_imit,
    "death_birth": rule_death_birth,
    "death_selection": rule_death_selection,
}

# Short display names for the front-ends -- kept short on purpose, because the
# family and the description belong in RULE_NOTES below, not in a dropdown.
RULE_LABELS = {
    "Imitate the best": "imitate_best",
    "Myopic best response": "best_response",
    "Proportional imitation": "imitate_prob",
    "Pairwise comparison (Fermi)": "fermi",
    "Local replicator": "proportional_imit",
    "Death-birth": "death_birth",
    "Death-selection": "death_selection",
}

# One line per rule, shown under the dropdown so the family is visible without
# making the menu unreadable.
RULE_NOTES = {
    "imitate_best": "**Imitation, deterministic.** Copy whoever did best nearby, "
                    "yourself included. The Nowak-May rule.",
    "best_response": "**Optimisation, innovative.** Play what would have been best "
                     "against the neighbours you just faced. The only rule here that "
                     "can introduce a strategy nobody nearby is using.",
    "imitate_prob": "**Imitation, stochastic.** Copy someone nearby with probability "
                    "rising in their payoff, measured from the worst in the neighbourhood.",
    "fermi": "**Imitation, stochastic.** Compare with one random neighbour and switch "
             "with a logistic probability in the payoff difference. `beta` is the "
             "intensity of selection -- and it carries units, so read it against the "
             "size of your payoffs.",
    "proportional_imit": "**Imitation, stochastic.** Compare with one random neighbour "
                         "and copy with probability proportional to how much better they "
                         "did. Its well-mixed limit IS the replicator dynamic.",
    "death_birth": "**Birth and death.** A site dies (the schedule decides which) and "
                   "its neighbours compete to fill it in proportion to payoff -- "
                   "selection acts on BIRTH. Turn `w` down for weak selection.",
    "death_selection": "**Birth and death.** Doing badly is what kills you, and the gap "
                       "is filled by a random neighbour -- selection acts on DEATH. The "
                       "mirror of death-birth; compare them at matched `w`.",
}


# ------------------------------------------------------------------------------
# 4. Scheduling: when do cells revise, and mutation
# ------------------------------------------------------------------------------
#
#   "synchronous"  every cell revises at the same instant, all looking at the
#                  same snapshot. The classic choice, and an odd one if you
#                  think about it -- it says the whole population has a shared
#                  clock. Produces the striking symmetric patterns.
#   "subset"       each cell revises independently with probability p per
#                  generation. p = 1 is synchronous; small p approximates
#                  continuous, uncoordinated revision. This is the honest
#                  middle, and it is cheap.
#   "sequential"   cells revise ONE AT A TIME in random order, each seeing the
#                  updates already made this sweep. The strictest asynchronous
#                  updating. It is slow (one full recomputation per cell), so
#                  keep the lattice small when you use it.
#
# Compare "synchronous" and "sequential" on the Nowak-May Prisoner's Dilemma and
# you reproduce the Huberman-Glance (1993) objection in about thirty seconds.
#
# THE SCHEDULE IS ALSO WHO DIES. For the two birth-death rules the schedule is
# doing double duty: it decides which sites die this generation. So "synchronous"
# there means EVERY site dies and is replaced at once -- a Wright-Fisher-style
# whole-population turnover, in which the neighbours competing to fill a vacancy
# are themselves dead. That is a real model, but it is not the one evolutionary
# graph theory usually means. For the Moran-style process behind results like
# Ohtsuki et al.'s b/c > k -- one death at a time, the rest of the population
# standing still -- use schedule="subset" with a small update_prob, or
# "sequential" if you want them strictly one at a time.
#
# MUTATION (`mutation` rate mu): after revising, each cell independently adopts
# a uniformly random strategy with probability mu. Small mu keeps the system off
# absorbing states and is what makes long-run frequencies meaningful.

SCHEDULES = ("synchronous", "subset", "sequential")


def _mutate(grid, k, mu, rng):
    if mu <= 0:
        return grid
    hit = rng.random(grid.shape) < mu
    if not hit.any():
        return grid
    return np.where(hit, rng.integers(0, k, size=grid.shape), grid)


# ------------------------------------------------------------------------------
# 5. Model and Sim: the objects you actually use
# ------------------------------------------------------------------------------

@dataclass
class Model:
    """Everything that defines the model except the current state and the seed.

    A `Model` is immutable in spirit -- build a new one (`dataclasses.replace`)
    rather than mutating a running simulation, so that a saved configuration
    always describes exactly one run.
    """
    A: np.ndarray                       # k x k symmetric payoff matrix
    rows: int = 20
    cols: int = 20
    neighbourhood: str = "Moore (8)"
    wrap: bool = True
    payoff_mode: str = "total"          # "total" | "average"
    self_play: bool = False
    rule: str = "imitate_best"
    schedule: str = "synchronous"       # "synchronous" | "subset" | "sequential"
    update_prob: float = 0.3            # only used by schedule="subset"
    beta: float = 0.1                   # only used by rule="fermi"; see its docstring
    death_rate: float = 0.5             # only used by rule="death_selection"
    selection: float = 1.0              # birth-death rules: 0 = neutral drift, 1 = full
    mutation: float = 0.0

    offsets: list = field(init=False, repr=False)

    def __post_init__(self):
        self.A = np.asarray(self.A, float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square payoff matrix")
        if self.A.shape[0] < 2:
            raise ValueError("need at least 2 strategies")
        if self.neighbourhood not in NEIGHBOURHOODS:
            raise ValueError(f"unknown neighbourhood {self.neighbourhood!r}")
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.payoff_mode not in ("total", "average"):
            raise ValueError("payoff_mode must be 'total' or 'average'")

        kind, radius = NEIGHBOURHOODS[self.neighbourhood]
        if kind == "line":
            self.rows = 1              # a 1D lattice is a one-row grid, always
        self.offsets = neighbour_offsets(kind, radius)
        self.rows, self.cols = int(self.rows), int(self.cols)
        if self.rows < 1 or self.cols < 1 or self.rows * self.cols < 2:
            raise ValueError("a lattice needs at least two sites")

        # A lattice can be too SMALL for its neighbourhood. On a torus, np.roll
        # wraps all the way round: if the lattice is only 2*radius wide, two
        # different offsets land on the same site and it is counted twice -- and
        # in the worst case (a one-row grid with a 2D neighbourhood) the offsets
        # (-1,0) and (1,0) both land on the CELL ITSELF, so it silently plays
        # against itself twice while self_play is off. Neither is a model anyone
        # intends, and both are invisible in the picture, so refuse them.
        if self.wrap:
            need = 2 * radius + 1
            if kind == "line":
                if self.cols < need:
                    raise ValueError(
                        f"a wrapped 1D lattice needs at least {need} columns for "
                        f"{self.neighbourhood} (got {self.cols}); otherwise "
                        "neighbours wrap round and get counted twice")
            elif self.rows < need or self.cols < need:
                raise ValueError(
                    f"a wrapped lattice needs at least {need}x{need} sites for "
                    f"{self.neighbourhood} (got {self.rows}x{self.cols}); otherwise "
                    "neighbours wrap round onto each other -- with one row, a cell "
                    "even ends up as its own neighbour")

        for name, lo, hi in (("update_prob", 0.0, 1.0), ("mutation", 0.0, 1.0),
                             ("death_rate", 0.0, 1.0), ("selection", 0.0, 1.0),
                             ("beta", 0.0, np.inf)):
            v = float(getattr(self, name))
            if not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} (got {v})")
            setattr(self, name, v)

    # -- small conveniences -------------------------------------------------
    @property
    def k(self):
        """Number of strategies."""
        return self.A.shape[0]

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def is_1d(self):
        return self.rows == 1

    @property
    def n_neighbours(self):
        """Neighbours of an interior site (every site, if wrapped)."""
        return len(self.offsets)

    def max_payoff_gap(self):
        """Largest payoff difference two sites can show. Used to normalise the
        local-replicator switching probability so it stays in [0, 1]."""
        spread = float(self.A.max() - self.A.min())
        if spread <= 0:
            return 1.0
        n = self.n_neighbours + (1 if self.self_play else 0)
        return spread if self.payoff_mode == "average" else spread * n


class Sim:
    """A model plus its current state, its seeded generator, and its history."""

    def __init__(self, model, grid=None, seed=0, init="random", probs=None,
                 keep_grids=False, **init_kw):
        self.model = model
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.generation = 0
        # keep_grids stores every lattice, not just the summary numbers. Off by
        # default (it is the only thing here that grows with generations x sites),
        # but essential for 1D, where the interesting picture is the SPACE-TIME
        # diagram: one row per generation, stacked downwards.
        self.keep_grids = bool(keep_grids)
        self.grids = []
        if grid is None:
            grid = make_grid(model, self.rng, init=init, probs=probs, **init_kw)
        grid = np.asarray(grid, np.int64)
        # NOT reshape(): a 6x6 grid handed to a 4x9 model would reshape without
        # complaint and quietly scramble the configuration you painted
        if grid.shape != model.shape:
            raise ValueError(f"grid is {grid.shape}, model lattice is {model.shape}")
        if grid.min() < 0 or grid.max() >= model.k:
            raise ValueError(f"grid holds strategies outside 0..{model.k - 1}")
        self.grid = grid.copy()
        self.initial_grid = self.grid.copy()
        self.previous_grid = None
        self.history = [self.snapshot()]
        if self.keep_grids:
            self.grids = [self.grid.astype(np.int8)]

    # -- one generation -----------------------------------------------------
    def _propose(self, grid):
        """Every site's proposed next strategy, given `grid`."""
        m = self.model
        pf = payoff_field(grid, m.A, m.offsets, m.wrap, m.payoff_mode, m.self_play)
        return RULES[m.rule](grid, pf, m, self.rng)

    def step(self):
        """Advance one generation. Returns the new grid."""
        m = self.model
        self.previous_grid = self.grid.copy()
        if m.schedule == "synchronous":
            new = self._propose(self.grid)
        elif m.schedule == "subset":
            proposed = self._propose(self.grid)
            revising = self.rng.random(self.grid.shape) < m.update_prob
            new = np.where(revising, proposed, self.grid)
        else:  # "sequential": one site at a time, each seeing the updates so far
            new = self.grid.copy()
            n = new.size
            order = self.rng.permutation(n)
            for flat in order:
                r, c = divmod(int(flat), m.cols)
                # recompute from the CURRENT (partly updated) grid, then keep
                # only this one site's proposal -- same code path as above, so
                # the rules are implemented exactly once
                new[r, c] = self._propose(new)[r, c]
        self.grid = _mutate(new, m.k, m.mutation, self.rng)
        self.generation += 1
        self.history.append(self.snapshot())
        if self.keep_grids:
            self.grids.append(self.grid.astype(np.int8))   # int8: 16 strategies max
        return self.grid

    def run(self, generations, stop_when_frozen=False):
        """Advance `generations` generations.

        `stop_when_frozen` stops early the first time a generation changes
        nothing. Use it with deterministic rules; under a stochastic rule a
        single still generation can be luck rather than an absorbing state.
        """
        for _ in range(int(generations)):
            self.step()
            if stop_when_frozen and self.frozen:
                break
        return self

    def reset(self):
        """Back to generation 0 with the same seed -- an exactly repeatable run."""
        self.rng = np.random.default_rng(self.seed)
        self.grid = self.initial_grid.copy()
        self.previous_grid = None
        self.generation = 0
        self.history = [self.snapshot()]
        self.grids = [self.grid.astype(np.int8)] if self.keep_grids else []
        return self

    def at(self, generation):
        """A read-only view of this run as it stood at `generation`.

        Lets a front-end run the simulation ONCE and then scrub back and forth
        through it, instead of re-running from the seed every time the display
        moves. Needs keep_grids=True.
        """
        if not self.grids:
            raise ValueError("no stored lattices -- build the Sim with keep_grids=True")
        g = int(np.clip(generation, 0, len(self.grids) - 1))
        v = object.__new__(Sim)
        v.model, v.seed, v.rng = self.model, self.seed, None
        v.generation, v.keep_grids = g, True
        v.grids = self.grids[:g + 1]
        v.grid = np.asarray(self.grids[g], np.int64)
        v.initial_grid = np.asarray(self.grids[0], np.int64)
        v.previous_grid = np.asarray(self.grids[g - 1], np.int64) if g else None
        v.history = self.history[:g + 1]
        return v

    def spacetime(self):
        """Every generation stacked into one array, generation 0 on top.

        The standard way to look at a 1D lattice: space across, time down.
        Requires keep_grids=True.
        """
        if not self.grids:
            raise ValueError("no stored lattices -- build the Sim with keep_grids=True")
        return np.concatenate([g if self.model.is_1d else g.reshape(1, -1)
                               for g in self.grids], axis=0)

    # -- reporting ----------------------------------------------------------
    def snapshot(self):
        """The numbers worth recording each generation."""
        m = self.model
        _, payoff, _, _ = payoff_field(self.grid, m.A, m.offsets, m.wrap,
                                       m.payoff_mode, m.self_play)
        return {
            "generation": self.generation,
            "frequencies": frequencies(self.grid, m.k),
            "assortment": assortment(self.grid, m.offsets, m.wrap, m.k),
            "mean_payoff": float(payoff.mean()),
        }

    def series(self, key):
        """A recorded quantity across all generations so far, as an array."""
        return np.array([h[key] for h in self.history])

    @property
    def frozen(self):
        """True if the last generation left every site unchanged.

        This compares GRIDS, not frequencies. Constant frequencies do not mean a
        frozen lattice -- a travelling wave or a rotating Rock-Paper-Scissors
        spiral can hold its composition exactly while every site keeps changing.

        Only decisive for deterministic rules with no mutation; a stochastic rule
        can sit still for one generation by luck and move again the next.
        """
        return self.previous_grid is not None and np.array_equal(
            self.grid, self.previous_grid)


# ------------------------------------------------------------------------------
# 6. Measurement: frequencies, realized assortment, payoffs
# ------------------------------------------------------------------------------

def frequencies(grid, k):
    """Fraction of sites playing each strategy."""
    return np.bincount(np.asarray(grid).ravel(), minlength=k)[:k] / grid.size


def assortment(grid, offsets, wrap, k=None):
    """The REALIZED assortment index r of the current lattice.

    Take every (site, neighbour) pair on the lattice and ask how often the two
    play the same strategy. Call that P_same. Then compare it with the baseline
    you would get if the very same strategies were reshuffled at random over the
    lattice, P_random (a permutation null, so it accounts for the fact that a
    population that is 90% cooperators has plenty of like-pairs by chance):

        r = (P_same - P_random) / (1 - P_random)

    r = 0   the lattice is doing nothing: neighbours are as alike as chance
    r > 0   POSITIVE ASSORTMENT: like meets like. This is the lever behind
            altruism -- the same r as in the replicator explorer's slider
            (Eshel & Cavalli-Sforza 1982). Roughly, altruism needs r > c/b.
    r < 0   negative assortment: like avoids like (the mirror lever behind
            spite). Lattices with imitation rarely produce it; a checkerboard
            initial condition does.
    r = 1   perfect segregation into pure blocks.

    Returns nan when the population is monomorphic (P_random = 1): with one
    strategy left there is nothing for assortment to be *of*.

    Why you care: space has no direct effect on which strategies win. It works
    by generating assortment. Read r off the lattice, put it in the replicator
    explorer, and see whether a well-mixed population with that much correlation
    would go the same way. That comparison, not the pictures, is the result.
    """
    grid = np.asarray(grid)
    k = int(grid.max()) + 1 if k is None else k
    nb, valid = neighbour_stack(grid, offsets, wrap)
    n_pairs = valid.sum()
    if n_pairs == 0:
        return float("nan")
    p_same = float(((nb == grid[None]) & valid).sum()) / float(n_pairs)

    # permutation baseline: draw two DISTINCT sites, chance they match
    counts = np.bincount(grid.ravel(), minlength=k)[:k].astype(float)
    n = float(grid.size)
    if n < 2:
        return float("nan")
    p_random = float((counts * (counts - 1)).sum()) / (n * (n - 1.0))
    if p_random >= 1.0 - 1e-12:
        return float("nan")
    return (p_same - p_random) / (1.0 - p_random)


def payoff_by_strategy(grid, model):
    """Mean payoff earned by each strategy present (nan where absent)."""
    m = model
    _, payoff, _, _ = payoff_field(grid, m.A, m.offsets, m.wrap,
                                   m.payoff_mode, m.self_play)
    out = np.full(m.k, np.nan)
    for s in range(m.k):
        mask = grid == s
        if mask.any():
            out[s] = payoff[mask].mean()
    return out


# ------------------------------------------------------------------------------
# 7. Initial conditions
# ------------------------------------------------------------------------------
#
# The starting configuration is not a detail. A random 50/50 scatter and a single
# compact cluster of the same size are completely different experiments: the
# cluster already has the assortment that the scatter has to build. The classic
# demonstrations use both.

def make_grid(model, rng, init="random", probs=None, patch=3, background=0,
              foreground=1):
    """Build a starting lattice.

    init = "random"       each site drawn independently from `probs`
                          (defaults to equal shares of every strategy)
    init = "patch"        a `patch` x `patch` block of `foreground` at the
                          centre of a field of `background` -- the classic
                          "can a cluster of cooperators invade?" setup
    init = "mutant"       exactly one `foreground` site in a field of
                          `background`
    init = "stripes"      alternating bands of every strategy (maximal
                          assortment without full segregation)
    init = "checkerboard" alternating single sites (maximal ANTI-assortment;
                          the r < 0 starting point)
    init = "halves"       the lattice split into equal blocks, one per strategy
    """
    rows, cols, k = model.rows, model.cols, model.k
    if init == "random":
        if probs is None:
            probs = np.full(k, 1.0 / k)
        probs = np.asarray(probs, float)
        if probs.sum() <= 0:
            raise ValueError("initial proportions must not be all zero")
        probs = probs / probs.sum()
        return rng.choice(k, size=(rows, cols), p=probs)
    if init in ("patch", "mutant"):
        g = np.full((rows, cols), background, np.int64)
        side = 1 if init == "mutant" else int(patch)
        r0 = max(0, (rows - side) // 2)
        c0 = max(0, (cols - side) // 2)
        g[r0:r0 + side, c0:c0 + side] = foreground
        return g
    if init == "stripes":
        band = max(1, cols // k)
        return (np.arange(cols)[None, :] // band % k) * np.ones((rows, 1), np.int64)
    if init == "checkerboard":
        rr, cc = np.indices((rows, cols))
        return (rr + cc) % k
    if init == "halves":
        return np.minimum(np.arange(cols)[None, :] * k // cols, k - 1) * \
            np.ones((rows, 1), np.int64)
    raise ValueError(f"unknown init {init!r}")


INITIAL_CONDITIONS = ("random", "patch", "mutant", "stripes", "checkerboard", "halves")


# ------------------------------------------------------------------------------
# 7b. Saving and sharing a configuration
# ------------------------------------------------------------------------------
#
# A run is defined by (model, seed, initial condition), and all three are small.
# `encode_config` packs them into one URL-safe string so a whole experiment can
# travel as a link: a student sends back the exact setup that produced their
# figure, and it reproduces site for site rather than approximately.

_CONFIG_FIELDS = ("rows", "cols", "neighbourhood", "wrap", "payoff_mode",
                  "self_play", "rule", "schedule", "update_prob", "beta",
                  "death_rate", "selection", "mutation")


def config_to_dict(model, seed=0, init="random", probs=None, names=None,
                   label=None):
    """A plain-dict description of a complete, reproducible run."""
    d = {f: getattr(model, f) for f in _CONFIG_FIELDS}
    d["A"] = np.asarray(model.A, float).tolist()
    d["seed"] = int(seed)
    d["init"] = init
    if probs is not None:
        d["probs"] = [float(p) for p in probs]
    if names is not None:
        d["names"] = list(names)     # so a shared Lewis link still says "01|01"
    if label is not None:
        d["label"] = str(label)
    return d


def config_from_dict(d):
    """Rebuild (model, seed, init, probs, names, label) from `config_to_dict`."""
    d = dict(d)
    seed = int(d.pop("seed", 0))
    init = d.pop("init", "random")
    probs = d.pop("probs", None)
    names = d.pop("names", None)
    label = d.pop("label", None)
    model = Model(**d)
    if names is not None:
        names = tuple(names)[:model.k]
    return model, seed, init, probs, names, label


def encode_config(model, seed=0, init="random", probs=None, names=None,
                  label=None):
    """Pack a run into one URL-safe string."""
    import base64
    import json
    raw = json.dumps(config_to_dict(model, seed, init, probs, names, label),
                     separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_config(text):
    """Unpack `encode_config`. Returns (model, seed, init, probs, names, label)."""
    import base64
    import json
    pad = "=" * (-len(text) % 4)
    return config_from_dict(json.loads(base64.urlsafe_b64decode(text + pad)))


def paint(grid, cells, strategy):
    """Set a list of (row, col) sites to `strategy`. Returns a new grid.

    The programmatic version of clicking on the picture: build any starting
    configuration you like, then hand it to `Sim(model, grid=...)`.
    """
    g = np.array(grid, np.int64, copy=True)
    for r, c in cells:
        g[int(r), int(c)] = int(strategy)
    return g


# ------------------------------------------------------------------------------
# 8. A library of games
# ------------------------------------------------------------------------------
#
# Each entry is (payoff matrix, strategy names). Row = the focal player's
# strategy, column = the opponent's; the entry is what the ROW player earns.
# All games here are symmetric, which is what a single-population lattice needs.

def prisoners_dilemma(b=1.9, c=1.0):
    """Donation-game Prisoner's Dilemma: cooperating costs c and gives b."""
    return np.array([[b - c, -c],
                     [b, 0.0]])


def nowak_may(b=1.85):
    """The Nowak & May (1992) 'weak' Prisoner's Dilemma: T=b, R=1, P=S=0.

    Defection is dominant, yet on a lattice with imitate-the-best and
    synchronous updating, cooperators survive indefinitely in shifting fractal
    patterns. Nowak & May report a cooperator frequency settling near 0.318 at
    b = 1.85.

    THIS TOOL REPRODUCES THAT -- but only with the right settings, and finding
    out which settings is more instructive than the result. Measured here
    (torus, Moore-8, imitate_best, synchronous, random 50/50 start, generation
    200), cooperator frequency at b = 1.85:

        100x100, self_play=True   0.308  0.339  0.297  0.337   <- Nowak & May
        150x150, self_play=True   0.315  0.329  0.310  0.308
         60x60,  self_play=True   0.000  0.321  0.318  0.002
         40x40,  self_play=True   0.000  0.004  1.000  0.282
         40x40,  self_play=False  0.000  0.000  0.000  0.000

    Two lessons, both about the bundle rather than the game:

    1. SELF-INTERACTION IS LOAD-BEARING. Nowak & May have each cell play a round
       against itself as well as its neighbours. Turn `self_play` off and
       cooperation is extinct for every b >= 1.6 -- the result vanishes. It is
       an assumption doing real work, not a convention.
    2. THE LATTICE MUST BE BIG. At 40x40 and 60x60 individual runs fixate at 0
       or 1 and the average across seeds is meaningless; the 0.318 coexistence
       only stabilises around 100x100. A small lattice is not a fast version of
       a big one, it is a different model.

    And the third, which is someone else's: change `schedule` to "sequential"
    and much of the structure goes away. Huberman & Glance (PNAS 90, 1993)
    argued that the spatial chaos depends on every cell updating in lockstep --
    a shared global clock is a strong assumption to hang a result on. Test it
    rather than taking my word for it.
    """
    return np.array([[1.0, 0.0],
                     [b, 0.0]])


def mini_nash_demand(pie=6, demands=(2, 3, 4)):
    """Skyrms's mini demand game: demand a share, get it only if compatible.

    Two players each name a share of a pie. If the demands are compatible
    (they sum to no more than the pie) each gets what it asked for; otherwise
    both get nothing. With a pie of 6 and demands 2, 3, 4 this is the standard
    three-strategy miniature of the Nash demand game:

                 vs 2   vs 3   vs 4
        demand 2    2      2      2
        demand 3    3      3      0
        demand 4    4      0      0

    "Demand half" is the unique symmetric ESS -- Skyrms's dynamical argument for
    an egalitarian norm. But 2-and-4 is a stable polymorphic pair, a *fair* rule
    and an unfair pair of complementary roles, and which one you get depends on
    where you start. On a lattice you can get BOTH at once, in different regions,
    which a well-mixed population cannot do.
    """
    d = np.asarray(demands, float)
    A = np.where(d[:, None] + d[None, :] <= pie, d[:, None], 0.0)
    return A, tuple(f"demand {int(x)}" for x in d)


def lewis_signaling(n=2):
    """The Lewis sender-receiver game, symmetrised into one population.

    Nature picks one of `n` equiprobable states. A SENDER sees the state and
    emits one of `n` signals; a RECEIVER sees only the signal and picks one of
    `n` acts. Both get 1 if the act matches the state, 0 otherwise. Nothing
    connects any signal to any state in advance -- that is the whole point.
    Meaning is not assumed, it has to evolve (Lewis 1969; Skyrms, *Signals*).

    A single population needs one symmetric game, so each individual carries a
    strategy for BOTH roles -- a sender rule AND a receiver rule -- and any two
    players are equally likely to meet in either arrangement:

        A[i, j] = 1/2 * success(sender rule of i, receiver rule of j)
                + 1/2 * success(sender rule of j, receiver rule of i)

    With n = 2 there are n^n = 4 sender rules and 4 receiver rules, so
    2 x 2 x 2 gives **16 strategies**. Names are written `sender|receiver` as
    the map read off in order: `01|01` sends signal 0 in state 0 and signal 1 in
    state 1, then acts 0 on signal 0 and 1 on signal 1.

    Two of the sixteen are SIGNALLING SYSTEMS -- `01|01` and `10|10` -- where the
    sender's map is one-to-one and the receiver's inverts it. They score 1
    against themselves and are the only strict Nash equilibria. Everything else
    is partial or total miscommunication, including pooling strategies that
    ignore the state entirely and score 1/2. The interesting question is not
    whether a signalling system is good but whether the population finds one,
    and which -- the two are equally good and perfectly incompatible, so this is
    a convention in Lewis's sense. A lattice can settle into both at once and
    leave a boundary between them.
    """
    maps = list(product(range(n), repeat=n))     # every function {0..n-1} -> {0..n-1}
    strategies = [(s, r) for s in maps for r in maps]

    def success(send, recv):
        """Chance the act matches the state, averaged over equiprobable states."""
        return sum(1.0 for state in range(n) if recv[send[state]] == state) / n

    k = len(strategies)
    A = np.zeros((k, k))
    for i, (s_i, r_i) in enumerate(strategies):
        for j, (s_j, r_j) in enumerate(strategies):
            A[i, j] = 0.5 * success(s_i, r_j) + 0.5 * success(s_j, r_i)
    names = tuple("".join(map(str, s)) + "|" + "".join(map(str, r))
                  for s, r in strategies)
    return A, names


def random_game(k, rng=None, low=0, high=10):
    """A random symmetric game: every entry an integer drawn from [low, high].

    Presets are all games somebody chose because they illustrate something. That
    is a biased sample. Drawing a matrix at random is the cheap way to ask which
    of the behaviour you have been looking at is a fact about lattices and which
    is a fact about the handful of games in the menu.
    """
    rng = np.random.default_rng() if rng is None else rng
    A = rng.integers(int(low), int(high) + 1, size=(int(k), int(k))).astype(float)
    return A, tuple(f"S{i + 1}" for i in range(int(k)))


def resize_matrix(A, k, fill=0.0):
    """Grow or shrink a payoff matrix to k x k, keeping the overlapping corner."""
    A = np.asarray(A, float)
    out = np.full((int(k), int(k)), float(fill))
    m = min(len(A), int(k))
    out[:m, :m] = A[:m, :m]
    return out


_lewis_A, _lewis_names = lewis_signaling(2)

GAMES = {
    "Prisoner's Dilemma (b=1.9, c=1)": (prisoners_dilemma(), ("Cooperate", "Defect")),
    "Prisoner's Dilemma - Nowak & May (b=1.85)": (nowak_may(), ("Cooperate", "Defect")),
    "Stag Hunt": (np.array([[4.0, 0.0],
                            [3.0, 3.0]]), ("Stag", "Hare")),
    "Hawk-Dove (v=2, c=3)": (np.array([[-0.5, 2.0],
                                       [0.0, 1.0]]), ("Hawk", "Dove")),
    "Driving game (which side of the road?)": (np.array([[1.0, 0.0],
                                                         [0.0, 1.0]]),
                                               ("Drive left", "Drive right")),
    "Spite (cost c=1, harm h=3)": (np.array([[-4.0, -1.0],
                                             [-3.0, 0.0]]), ("Harm", "Refrain")),
    "Rock-Paper-Scissors": (np.array([[0.0, -1.0, 1.0],
                                      [1.0, 0.0, -1.0],
                                      [-1.0, 1.0, 0.0]]), ("Rock", "Paper", "Scissors")),
    "Coordination (3 conventions)": (np.eye(3), ("A", "B", "C")),
    "Mini Nash demand (2/3/4 of 6)": mini_nash_demand(),
    "Lewis signalling 2x2x2 (16 strategies)": (_lewis_A, _lewis_names),
}
"""Presets. `GAMES[name] -> (A, strategy_names)`.

Stag Hunt: Stag pays 4 against Stag but 0 against Hare; Hare pays 3 regardless.
Both pure profiles are equilibria; Stag is payoff-dominant and Hare is
risk-dominant. Skyrms's argument that the social contract is a stag hunt rather
than a prisoner's dilemma turns on exactly that gap -- the problem is not
temptation, it is trust.

Driving game: pure coordination. Nothing distinguishes left from right, both
conventions are equally good, and all that matters is doing what your
neighbours do. The cleanest case of a convention in Lewis's sense, and the one
where a lattice most obviously does something a well-mixed population cannot:
it can settle into different conventions in different regions and hold the
boundary between them for a very long time.

Spite: a Harm player pays c=1 to impose h=3 on its opponent, so
A[Harm, Refrain] = -1, A[Refrain, Harm] = -3, A[Harm, Harm] = -1-3 = -4. Every
payoff is negative and the game is not a Prisoner's Dilemma in disguise -- it is
worth watching under a rule that is shift-invariant (fermi, local replicator)
and under one that is not (proportional imitation), because the all-negative
payoffs are exactly where the difference shows up.
"""
