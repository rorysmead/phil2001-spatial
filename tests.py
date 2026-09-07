"""
tests.py -- correctness checks for the spatial core.

These check the model against results you can derive by hand or look up. Run:

    python3 tests.py

Every line should end in PASS. This is the safety net: if you edit spatial.py
and a fact about the model breaks, you find out here rather than in front of a
class.
"""
import numpy as np

import spatial as sp


def approx(a, b, tol=1e-9):
    return np.abs(np.asarray(a, float) - np.asarray(b, float)).max() < tol


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main():
    ok = True

    # ==========================================================================
    # 1. Geometry: neighbourhoods and the shift
    # ==========================================================================
    sizes = {name: len(sp.neighbour_offsets(*kr))
             for name, kr in sp.NEIGHBOURHOODS.items()}
    ok &= check("von Neumann r=1 has 4 neighbours", sizes["von Neumann (4)"] == 4)
    ok &= check("Moore r=1 has 8 neighbours", sizes["Moore (8)"] == 8)
    ok &= check("von Neumann r=2 has 12 neighbours", sizes["von Neumann r=2 (12)"] == 12)
    ok &= check("Moore r=2 has 24 neighbours (GenLab advertised it, never built it)",
                sizes["Moore r=2 (24)"] == 24)
    ok &= check("1D line r=1 has 2 neighbours", sizes["1D line, 2 neighbours"] == 2)
    ok &= check("no neighbourhood contains the cell itself",
                all((0, 0) not in sp.neighbour_offsets(*kr)
                    for kr in sp.NEIGHBOURHOODS.values()))

    a = np.arange(12).reshape(3, 4)
    up, valid = sp._shift(a, -1, 0, wrap=False)     # the neighbour ABOVE me
    ok &= check("shift reads the right site: row 1 sees row 0", up[1, 2] == a[0, 2])
    ok &= check("shift marks the top row invalid when not wrapping", not valid[0, 0])
    upw, validw = sp._shift(a, -1, 0, wrap=True)
    ok &= check("wrapped shift takes the top row from the bottom", upw[0, 2] == a[2, 2])
    ok &= check("wrapped shift has no invalid sites", validw.all())

    # neighbour counts on a BOUNDED Moore lattice: corner 3, edge 5, interior 8
    g = np.zeros((5, 5), np.int64)
    off = sp.neighbour_offsets("moore", 1)
    _, nb_valid = sp.neighbour_stack(g, off, wrap=False)
    counts = nb_valid.sum(axis=0)
    ok &= check("bounded lattice: corner has 3 neighbours", counts[0, 0] == 3)
    ok &= check("bounded lattice: edge has 5 neighbours", counts[0, 2] == 5)
    ok &= check("bounded lattice: interior has 8 neighbours", counts[2, 2] == 8)
    _, nb_valid_w = sp.neighbour_stack(g, off, wrap=True)
    ok &= check("wrapped lattice: every site has 8 neighbours",
                (nb_valid_w.sum(axis=0) == 8).all())

    # ==========================================================================
    # 2. Payoffs
    # ==========================================================================
    # Hand-checkable: 3x3 of all-strategy-0 with one strategy-1 at the centre.
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    grid = np.zeros((3, 3), np.int64)
    grid[1, 1] = 1
    m = sp.Model(A=A, rows=3, cols=3, neighbourhood="Moore (8)", wrap=False)
    pay_all, pay, _, _ = sp.payoff_field(grid, A, m.offsets, wrap=False)
    # centre plays strategy 1 against 8 neighbours all playing 0 -> 8 * A[1,0]
    ok &= check("centre payoff = 8 * A[1,0] = 24", approx(pay[1, 1], 24.0))
    # corner plays 0 against its 3 neighbours: two 0s and one 1 -> 2*A[0,0]+A[0,1]
    ok &= check("corner payoff = 2*A[0,0] + A[0,1] = 4", approx(pay[0, 0], 4.0))
    ok &= check("counterfactual payoffs match the actual one",
                approx(pay_all[grid, np.arange(3)[:, None], np.arange(3)[None, :]], pay))

    pay_all_avg, pay_avg, _, _ = sp.payoff_field(grid, A, m.offsets, wrap=False,
                                                 payoff_mode="average")
    ok &= check("average mode divides the corner by its 3 real neighbours",
                approx(pay_avg[0, 0], 4.0 / 3.0))
    ok &= check("average mode divides the centre by 8", approx(pay_avg[1, 1], 3.0))

    _, pay_self, _, _ = sp.payoff_field(grid, A, m.offsets, wrap=False, self_play=True)
    ok &= check("self-play adds one round against yourself (A[1,1]=4)",
                approx(pay_self[1, 1], 28.0))

    # ==========================================================================
    # 3. Update rules
    # ==========================================================================
    # A strictly dominant strategy must take over under BOTH deterministic rules.
    A_dom = np.array([[1.0, -10.0],
                      [10.0, 0.0]])                 # strategy 1 dominates hard
    for rule in ("imitate_best", "best_response"):
        mm = sp.Model(A=A_dom, rows=12, cols=12, rule=rule)
        s = sp.Sim(mm, seed=1).run(30)
        ok &= check(f"{rule}: a strictly dominant strategy takes the lattice",
                    approx(s.snapshot()["frequencies"], [0.0, 1.0]))

    # best response against a dominant strategy is immediate: one generation.
    mm = sp.Model(A=A_dom, rows=10, cols=10, rule="best_response")
    s = sp.Sim(mm, seed=2)
    s.step()
    ok &= check("best response reaches all-dominant in ONE generation",
                (s.grid == 1).all())

    # Monomorphic states are absorbing for every copying rule (nobody to copy).
    for rule in ("imitate_best", "imitate_prob", "fermi", "proportional_imit",
                 "death_birth", "death_selection"):
        mm = sp.Model(A=sp.GAMES["Stag Hunt"][0], rows=8, cols=8, rule=rule, beta=2.0)
        s = sp.Sim(mm, grid=np.zeros((8, 8), np.int64), seed=3).run(10)
        ok &= check(f"{rule}: a monomorphic lattice never changes", (s.grid == 0).all())

    # ... but myopic best response is INNOVATIVE: it can revive a dead strategy.
    mm = sp.Model(A=np.array([[0.0, 0.0], [1.0, 1.0]]), rows=6, cols=6,
                  rule="best_response")
    s = sp.Sim(mm, grid=np.zeros((6, 6), np.int64), seed=4)
    s.step()
    ok &= check("best response revives a strategy nobody was playing",
                (s.grid == 1).all())

    # The local replicator never copies someone who did no better than you, so a
    # site whose whole neighbourhood is tied with it cannot move. Two solid
    # blocks of a coordination game: everything except the two seams is tied.
    A_c = sp.GAMES["Driving game (which side of the road?)"][0]
    mm = sp.Model(A=A_c, rows=20, cols=20, rule="proportional_imit")
    s = sp.Sim(mm, init="halves", seed=5)
    before = s.grid.copy()
    s.step()
    interior = np.zeros((20, 20), bool)          # columns >=2 away from either seam
    interior[:, [3, 4, 5, 6, 13, 14, 15, 16]] = True
    ok &= check("local replicator: a site tied with all its neighbours never moves",
                np.array_equal(s.grid[interior], before[interior]))

    # --- the two birth-death rules ---
    # death_birth uses the OPEN neighbourhood: the dead occupant does not compete
    # to replace itself, so a site whose neighbours all play 1 must become 1 --
    # whatever its own payoff was.
    g = np.ones((5, 5), np.int64)
    g[2, 2] = 0
    mm = sp.Model(A=A_dom, rows=5, cols=5, rule="death_birth")
    s = sp.Sim(mm, grid=g, seed=6)
    s.step()
    ok &= check("death-birth: the occupant does not compete to replace itself",
                s.grid[2, 2] == 1)
    # imitate_prob keeps the incumbent in the running, so with a big enough
    # payoff advantage that same site can hold on -- the contrast is the point
    mm2 = sp.Model(A=np.array([[100.0, 100.0], [0.0, 0.0]]), rows=5, cols=5,
                   rule="imitate_prob")
    s2 = sp.Sim(mm2, grid=g.copy(), seed=6)
    s2.step()
    ok &= check("proportional imitation: the incumbent can hold its own site",
                s2.grid[2, 2] == 0)

    # death_selection: nobody dies at rate 0; the locally worst always dies at rate 1
    mm = sp.Model(A=A_dom, rows=10, cols=10, rule="death_selection", death_rate=0.0)
    s = sp.Sim(mm, seed=7)
    s.step()
    ok &= check("death-selection: death_rate 0 means nothing ever changes",
                np.array_equal(s.grid, s.initial_grid))
    # with a dominant strategy and death_rate 1 the worst performers turn over fast
    mm = sp.Model(A=A_dom, rows=16, cols=16, rule="death_selection", death_rate=1.0)
    s = sp.Sim(mm, seed=8).run(60)
    ok &= check("death-selection: a strictly dominant strategy still takes over",
                s.snapshot()["frequencies"][1] > 0.95)

    # selection strength w: at w=0 payoffs are ignored entirely, so BOTH birth-death
    # rules must become pure drift -- the payoff matrix cannot matter at all.
    for rule in ("death_birth", "death_selection"):
        g0 = sp.Sim(sp.Model(A=A_dom, rows=20, cols=20, rule=rule, selection=0.0,
                             death_rate=1.0), seed=9).run(5).grid
        g1 = sp.Sim(sp.Model(A=A_dom * -3.0 + 7.0, rows=20, cols=20, rule=rule,
                             selection=0.0, death_rate=1.0), seed=9).run(5).grid
        ok &= check(f"{rule}: selection=0 is pure drift (the game stops mattering)",
                    np.array_equal(g0, g1))
    # ... and drift alone does not systematically favour the dominant strategy
    drift = [sp.Sim(sp.Model(A=A_dom, rows=20, cols=20, rule="death_birth",
                             selection=0.0), seed=sd).run(20).snapshot()["frequencies"][1]
             for sd in range(6)]
    ok &= check("death-birth under drift: no systematic push to the dominant strategy",
                0.2 < float(np.mean(drift)) < 0.8)
    # weak selection lets cooperation survive in a donation game where full
    # selection wipes it out (see rule_death_birth's docstring for the table)
    A_don = sp.prisoners_dilemma(b=12.0, c=1.0)
    weak = [sp.Sim(sp.Model(A=A_don, rows=40, cols=40, rule="death_birth",
                            selection=0.02), seed=sd).run(200).snapshot()["frequencies"][0]
            for sd in range(3)]
    strong = [sp.Sim(sp.Model(A=A_don, rows=40, cols=40, rule="death_birth",
                              selection=1.0), seed=sd).run(200).snapshot()["frequencies"][0]
              for sd in range(3)]
    ok &= check("death-birth: weak selection sustains cooperation where full selection "
                f"does not (weak {np.mean(weak):.2f} vs full {np.mean(strong):.2f})",
                np.mean(weak) > 0.05 and np.mean(strong) < 0.01)

    # ==========================================================================
    # 4. Scheduling and mutation
    # ==========================================================================
    base = sp.Model(A=sp.GAMES["Prisoner's Dilemma (b=1.9, c=1)"][0], rows=10, cols=10)
    s1 = sp.Sim(base, seed=7)
    s1.step()
    s2 = sp.Sim(sp.Model(**{**_fields(base), "schedule": "subset", "update_prob": 1.0}),
                seed=7)
    s2.step()
    ok &= check("subset schedule with p=1 reproduces synchronous updating",
                np.array_equal(s1.grid, s2.grid))

    s3 = sp.Sim(sp.Model(**{**_fields(base), "schedule": "subset", "update_prob": 0.0}),
                seed=7)
    s3.step()
    ok &= check("subset schedule with p=0 changes nothing",
                np.array_equal(s3.grid, s3.initial_grid))

    s4 = sp.Sim(sp.Model(**{**_fields(base), "rows": 6, "cols": 6,
                            "schedule": "sequential"}), seed=7)
    s4.run(3)
    ok &= check("sequential schedule runs and keeps the lattice shape",
                s4.grid.shape == (6, 6) and s4.generation == 3)

    s5 = sp.Sim(sp.Model(**{**_fields(base), "mutation": 1.0}), seed=8)
    s5.step()
    ok &= check("mutation rate 1 randomises the whole lattice",
                0.2 < float((s5.grid == 1).mean()) < 0.8)

    # ==========================================================================
    # 5. Reproducibility
    # ==========================================================================
    r1 = sp.Sim(base, seed=42).run(15)
    r2 = sp.Sim(base, seed=42).run(15)
    r3 = sp.Sim(base, seed=43).run(15)
    ok &= check("same seed gives an identical run", np.array_equal(r1.grid, r2.grid))
    # compare TRAJECTORIES, not endpoints: this PD fixates at all-Defect from
    # every seed, so identical endpoints would prove nothing
    ok &= check("a different seed gives a different run",
                not np.array_equal(r1.series("frequencies"), r3.series("frequencies")))
    r4 = sp.Sim(base, seed=42).run(15)
    r4.reset().run(15)
    ok &= check("reset() replays the same run exactly", np.array_equal(r1.grid, r4.grid))

    # ==========================================================================
    # 6. Meaningfulness: payoffs are an interval scale
    # ==========================================================================
    # Only differences carry meaning, so every rule must survive a positive
    # affine change of payoffs, A -> alpha*A + c. On a WRAPPED lattice every site
    # has the same number of neighbours, so the constant shifts every total
    # equally. GenLab's probabilistic rule used a RATIO of payoffs and failed
    # this; ours must not. The stated exception is fermi, whose beta carries
    # units of 1/payoff, so rescaling payoffs is the same as rescaling beta.
    A0 = sp.GAMES["Stag Hunt"][0]
    all_rules = ("imitate_best", "best_response", "imitate_prob", "fermi",
                 "proportional_imit", "death_birth", "death_selection")

    # beta=0.05 keeps the Fermi logistic well away from saturation -- at the old
    # default of 1.0 with 8 neighbours it sits at 0.9997, i.e. deterministic, and
    # the scale dependence below would be invisible
    def run_with(A, rule):
        return sp.Sim(sp.Model(A=A, rows=14, cols=14, rule=rule, beta=0.05,
                               death_rate=0.5), seed=11).run(8).grid

    for rule in all_rules:
        ok &= check(f"{rule}: unchanged by adding 100 to every payoff",
                    np.array_equal(run_with(A0, rule), run_with(A0 + 100.0, rule)))

    for rule in all_rules:
        invariant = rule != "fermi"
        same = np.array_equal(run_with(A0, rule), run_with(A0 * 10.0, rule))
        ok &= check(
            f"{rule}: {'unchanged by' if invariant else 'SCALE-dependent (beta has units)'}"
            f"{' multiplying every payoff by 10' if invariant else ''}",
            same == invariant)

    # ==========================================================================
    # 7. Assortment: the number that connects this tool to the replicator explorer
    # ==========================================================================
    off_vn = sp.neighbour_offsets("vn", 1)
    # The baseline is a permutation null on a FINITE lattice, so the extremes are
    # +-1 only up to O(1/N): a 20x20 checkerboard gives -0.995, a 60x60 gives
    # -0.9994. That is the correction working, not an error.
    r_checker = sp.assortment(np.indices((20, 20)).sum(axis=0) % 2, off_vn, True, 2)
    r_checker_big = sp.assortment(np.indices((60, 60)).sum(axis=0) % 2, off_vn, True, 2)
    ok &= check("checkerboard has r = -1 up to the finite-lattice correction",
                r_checker < -0.99 and r_checker_big < r_checker)

    blocks = np.zeros((20, 20), np.int64)
    blocks[:, 10:] = 1
    r_blocks = sp.assortment(blocks, off_vn, wrap=True, k=2)
    ok &= check("two solid blocks have r close to 1", 0.85 < r_blocks < 1.0)

    rng = np.random.default_rng(0)
    r_rand = sp.assortment(rng.integers(0, 2, (60, 60)), off_vn, wrap=True, k=2)
    ok &= check("a random scatter has r near 0 (no assortment)", abs(r_rand) < 0.05)

    mono = np.zeros((10, 10), np.int64)
    ok &= check("a monomorphic lattice has undefined assortment (nan)",
                np.isnan(sp.assortment(mono, off_vn, wrap=True, k=2)))

    # Assortment is what space BUILDS: start random, watch r rise under imitation.
    m_pd = sp.Model(A=sp.nowak_may(1.85), rows=40, cols=40, rule="imitate_best",
                    self_play=True)
    s_pd = sp.Sim(m_pd, seed=17).run(10)
    r_series = s_pd.series("assortment")
    ok &= check("imitation builds assortment from a random start (r rises)",
                r_series[0] < 0.05 and r_series[-1] > 0.6)

    # ==========================================================================
    # 7b. Replicating a published result
    # ==========================================================================
    # Nowak & May (Nature 359, 1992) report the cooperator frequency settling
    # near 0.318 at b = 1.85. It only appears with self-interaction ON and a
    # lattice of about 100x100 or more -- below that, single runs fixate at 0
    # or 1. Both of those are documented in nowak_may()'s docstring; this is the
    # check that the headline number is actually reproduced.
    m_nm = sp.Model(A=sp.nowak_may(1.85), rows=100, cols=100, rule="imitate_best",
                    self_play=True)
    freqs = [sp.Sim(m_nm, seed=sd).run(200).snapshot()["frequencies"][0]
             for sd in (0, 1)]
    ok &= check(f"Nowak & May 1992: cooperators settle near 0.318 "
                f"(got {freqs[0]:.3f}, {freqs[1]:.3f})",
                all(0.25 < f < 0.40 for f in freqs))

    m_small = sp.Model(A=sp.nowak_may(1.85), rows=40, cols=40, rule="imitate_best",
                       self_play=False)
    ok &= check("...and it disappears entirely without self-interaction",
                all(sp.Sim(m_small, seed=sd).run(60).snapshot()["frequencies"][0] == 0.0
                    for sd in (0, 1, 2)))

    # ==========================================================================
    # 8. Bookkeeping
    # ==========================================================================
    ok &= check("frequencies sum to 1", approx(s_pd.snapshot()["frequencies"].sum(), 1.0))
    ok &= check("history has one entry per generation plus the start",
                len(s_pd.history) == 11)
    m1d = sp.Model(A=A0, cols=50, neighbourhood="1D line, 2 neighbours")
    ok &= check("a 1D neighbourhood forces a one-row lattice",
                m1d.rows == 1 and m1d.is_1d)
    s1d = sp.Sim(m1d, seed=3).run(5)
    ok &= check("1D lattices run", s1d.grid.shape == (1, 50))

    prob_grid = sp.make_grid(sp.Model(A=A0, rows=100, cols=100),
                             np.random.default_rng(1), init="random", probs=[0.8, 0.2])
    ok &= check("initial proportions are respected",
                abs(float((prob_grid == 1).mean()) - 0.2) < 0.02)
    patch = sp.make_grid(sp.Model(A=A0, rows=11, cols=11),
                         np.random.default_rng(1), init="patch", patch=3)
    ok &= check("patch init plants a 3x3 block", int((patch == 1).sum()) == 9)
    mut = sp.make_grid(sp.Model(A=A0, rows=11, cols=11),
                       np.random.default_rng(1), init="mutant")
    ok &= check("mutant init plants exactly one cell", int((mut == 1).sum()) == 1)

    painted = sp.paint(np.zeros((5, 5), np.int64), [(0, 0), (4, 4)], 1)
    ok &= check("paint() sets the cells it is given", painted[0, 0] == 1
                and painted[4, 4] == 1 and painted.sum() == 2)

    ok &= check("every preset game is square and named",
                all(np.asarray(A).shape[0] == np.asarray(A).shape[1] == len(names)
                    for A, names in sp.GAMES.values()))
    ok &= check("every rule in the label menu exists",
                all(v in sp.RULES for v in sp.RULE_LABELS.values()))

    # ==========================================================================
    # 9. Refusing models that are quietly wrong
    # ==========================================================================
    # Each of these ran happily before the 2026-08-08 bug check and produced a
    # model nobody intended. They must now raise.
    def raises(name, **kw):
        base_kw = dict(A=A0, rows=8, cols=8)
        base_kw.update(kw)
        try:
            sp.Model(**base_kw)
            return False
        except ValueError:
            return True

    ok &= check("a one-row wrapped lattice with a 2D neighbourhood is refused "
                "(the cell would be its own neighbour, twice)",
                raises("selfnbr", rows=1, cols=8, neighbourhood="Moore (8)"))
    ok &= check("a wrapped lattice too small for its radius is refused "
                "(neighbours would be counted twice)",
                raises("alias", cols=6, neighbourhood="1D line, 8 neighbours"))
    ok &= check("a 1x1 lattice is refused", raises("tiny", rows=1, cols=1,
                                                   neighbourhood="von Neumann (4)"))
    for bad in ({"mutation": 5.0}, {"selection": -2.0}, {"beta": -1.0},
                {"update_prob": 3.0}, {"death_rate": 9.0}):
        ok &= check(f"out-of-range {list(bad)[0]} is refused", raises("range", **bad))
    ok &= check("an unknown neighbourhood name is refused",
                raises("kind", neighbourhood="Moore (9)"))
    ok &= check("an unknown rule is refused", raises("rule", rule="copy_the_worst"))
    ok &= check("an unknown schedule is refused", raises("sched", schedule="whenever"))
    ok &= check("a 1x1 payoff matrix is refused", raises("k1", A=np.array([[1.0]])))

    # the same lattice UNWRAPPED is fine: no aliasing without a torus
    try:
        sp.Model(A=A0, rows=1, cols=8, neighbourhood="Moore (8)", wrap=False)
        bounded_ok = True
    except ValueError:
        bounded_ok = False
    ok &= check("...but the same small lattice is fine when it is not wrapped",
                bounded_ok)

    # a mis-shaped or out-of-range starting grid must be refused, not reshaped
    m_shape = sp.Model(A=A0, rows=4, cols=9)
    for bad_grid, why in ((np.zeros((6, 6), np.int64), "wrong shape"),
                          (np.full((4, 9), 7, np.int64), "strategy out of range")):
        try:
            sp.Sim(m_shape, grid=bad_grid, seed=0)
            caught = False
        except ValueError:
            caught = True
        ok &= check(f"a starting grid with the {why} is refused", caught)

    # frozen must compare GRIDS: constant frequencies are not a frozen lattice
    m_move = sp.Model(A=A0, rows=10, cols=10)
    s_move = sp.Sim(m_move, grid=np.zeros((10, 10), np.int64), seed=0)
    ok &= check("frozen is False before anything has run", not s_move.frozen)
    s_move.step()
    ok &= check("a monomorphic lattice reports frozen", s_move.frozen)
    # a hand-built travelling pattern: composition constant, every site changing
    shift_grid = np.tile(np.array([0, 0, 1, 1]), (8, 2))
    s_shift = sp.Sim(sp.Model(A=A0, rows=8, cols=8), grid=shift_grid, seed=0)
    s_shift.previous_grid = np.roll(shift_grid, 1, axis=1)
    ok &= check("constant frequencies with a changed grid is NOT frozen",
                approx(sp.frequencies(s_shift.grid, 2),
                       sp.frequencies(s_shift.previous_grid, 2))
                and not s_shift.frozen)

    # ==========================================================================
    # 10. The game library and matrix editing
    # ==========================================================================
    A_lew, n_lew = sp.lewis_signaling(2)
    ok &= check("Lewis 2x2x2 has 16 strategies", A_lew.shape == (16, 16)
                and len(n_lew) == 16)
    ok &= check("Lewis payoffs are symmetric (one population, either role)",
                approx(A_lew, A_lew.T))
    # a signalling system: sender's map one-to-one, receiver inverts it
    _sys = [n_lew.index("01|01"), n_lew.index("10|10")]
    ok &= check("the two signalling systems score a perfect 1 against themselves",
                all(approx(A_lew[i, i], 1.0) for i in _sys))
    ok &= check("nothing else scores 1 against itself",
                sorted(i for i in range(16) if approx(A_lew[i, i], 1.0)) == sorted(_sys))
    _strict = [i for i in range(16)
               if all(A_lew[i, i] > A_lew[j, i] for j in range(16) if j != i)]
    ok &= check("the signalling systems are the only strict Nash equilibria",
                sorted(_strict) == sorted(_sys))
    ok &= check("a pooling sender (ignores the state) scores only 1/2",
                approx(A_lew[n_lew.index("00|01"), n_lew.index("00|01")], 0.5))
    # and it actually runs on a lattice
    _m_lew = sp.Model(A=A_lew, rows=30, cols=30, rule="imitate_best")
    _s_lew = sp.Sim(_m_lew, seed=1).run(40)
    ok &= check("the 16-strategy Lewis game runs on a lattice",
                approx(_s_lew.snapshot()["frequencies"].sum(), 1.0))

    A_dem, n_dem = sp.mini_nash_demand()
    ok &= check("mini demand game: compatible demands are paid, greedy pairs get 0",
                approx(A_dem, [[2, 2, 2], [3, 3, 0], [4, 0, 0]]))
    ok &= check("mini demand: 'demand half' is a symmetric Nash equilibrium",
                all(A_dem[1, 1] >= A_dem[j, 1] for j in range(3)))

    ok &= check("driving game is pure coordination",
                approx(sp.GAMES["Driving game (which side of the road?)"][0],
                       np.eye(2)))
    ok &= check("Stag Hunt is a stag hunt (both pure profiles are equilibria)",
                sp.GAMES["Stag Hunt"][0][0][0] > sp.GAMES["Stag Hunt"][0][1][0]
                and sp.GAMES["Stag Hunt"][0][1][1] > sp.GAMES["Stag Hunt"][0][0][1])

    _rg, _rn = sp.random_game(5, np.random.default_rng(0))
    ok &= check("random_game returns k x k integers inside [0, 10]",
                _rg.shape == (5, 5) and _rg.min() >= 0 and _rg.max() <= 10
                and approx(_rg, np.round(_rg)))
    _spread = [sp.random_game(6, np.random.default_rng(s))[0] for s in range(3)]
    ok &= check("random_game actually varies with the seed",
                not approx(_spread[0], _spread[1]))

    _base = np.arange(9, dtype=float).reshape(3, 3)
    ok &= check("shrinking a matrix keeps the top-left corner",
                approx(sp.resize_matrix(_base, 2), [[0, 1], [3, 4]]))
    ok &= check("growing a matrix pads with zeros and keeps the old entries",
                approx(sp.resize_matrix(_base, 4)[:3, :3], _base)
                and approx(sp.resize_matrix(_base, 4)[3], [0, 0, 0, 0]))
    ok &= check("every preset survives a round trip through resize at its own k",
                all(approx(sp.resize_matrix(A, len(A)), A) for A, _ in sp.GAMES.values()))

    # every preset must actually run, including the 16-strategy one
    _ran = []
    for _name, (_A, _nm) in sp.GAMES.items():
        _m = sp.Model(A=_A, rows=24, cols=24, rule="imitate_best")
        _ran.append(len(sp.Sim(_m, seed=0).run(15).snapshot()["frequencies"]) == len(_nm))
    ok &= check(f"all {len(sp.GAMES)} preset games run and report one frequency "
                "per strategy", all(_ran))

    # ==========================================================================
    # 11. Sharing a configuration
    # ==========================================================================
    _m_cfg = sp.Model(A=A_lew, rows=30, cols=30, rule="fermi", beta=0.3,
                      schedule="subset", update_prob=0.4, mutation=0.01)
    _txt = sp.encode_config(_m_cfg, seed=99, init="patch",
                            probs=[1.0] * 16, names=n_lew, label="Lewis")
    _m2, _s2, _i2, _p2, _n2, _l2 = sp.decode_config(_txt)
    ok &= check("a shared config round-trips the model, seed and initial condition",
                approx(_m2.A, _m_cfg.A) and _m2.rule == "fermi"
                and approx(_m2.beta, 0.3) and _s2 == 99 and _i2 == "patch")
    ok &= check("a shared config round-trips strategy names and proportions",
                _n2 == n_lew and len(_p2) == 16 and _l2 == "Lewis")
    ok &= check("the two runs really are identical after a round trip",
                np.array_equal(sp.Sim(_m_cfg, seed=99).run(10).grid,
                               sp.Sim(_m2, seed=99).run(10).grid))

    print()
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return 0 if ok else 1


def _fields(model):
    """The constructor arguments of a Model (offsets is derived, not passed)."""
    return {k: v for k, v in vars(model).items() if k != "offsets"}


if __name__ == "__main__":
    raise SystemExit(main())
