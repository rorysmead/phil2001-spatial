# Spatial Evolution Explorer (PHIL 2001)

A lattice-based evolutionary game simulator, built as a teaching resource for *Ethics and
Evolutionary Games*. Two front-ends, **one shared model core**, so the model is defined in
exactly one place.

Replaces the GenLab web app used in previous years (Christopher Myers, a final project for
this course). What changed and why is at the bottom.

## For students (nothing to install)

- **Explore the model:** open the widget in your browser → **https://OWNER.github.io/REPO/**
- **Read / edit the code:** open the notebook in Colab →
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebook.ipynb)

*(Instructor: replace `OWNER`/`REPO` throughout this repo with your GitHub username and
repository name — one command in the "Deploying" section below.)*

## The two tracks

| For students who... | Use | File |
|---|---|---|
| just want to **explore** with controls | the marimo widget | `app.py` |
| want to **read and edit** the model | the Colab notebook | `notebook.ipynb` |

Both import the same core:

- **`spatial.py`** — the model. Lattices, neighbourhoods, payoffs, seven update rules,
  scheduling, assortment. Heavily commented; this is the file to read first.
- **`render.py`** — the drawing layer (Matplotlib). No model here.
- **`tests.py`** — 96 correctness checks. Run `python3 tests.py`.

## What it does

- **Lattices**: 1D or 2D, wrapped (torus) or bounded, von Neumann (4), Moore (8), von
  Neumann r=2 (12), Moore r=2 (24).
- **Any symmetric game**: choose k from 2 to 16 and edit every entry of the k×k matrix.
  Presets:

  | preset | k | what it is for |
  |---|---|---|
  | Prisoner's Dilemma (donation, b/c) | 2 | altruism with a tunable benefit-to-cost ratio |
  | Prisoner's Dilemma — Nowak & May | 2 | the published spatial-chaos result |
  | Stag Hunt | 2 | Skyrms's social contract: trust, not temptation |
  | Hawk–Dove | 2 | conflict with a polymorphic equilibrium |
  | Driving game | 2 | pure convention — two equally good answers |
  | Spite (cost c, harm h) | 2 | harm as first-class; every payoff negative |
  | Rock-Paper-Scissors | 3 | cyclic dominance, spirals |
  | Coordination (3 conventions) | 3 | more than two conventions |
  | Mini Nash demand (2/3/4 of 6) | 3 | Skyrms's egalitarian ESS vs the 2–4 polymorphism |
  | **Lewis signalling 2×2×2** | **16** | meaning from nothing (see below) |

- **Random payoffs** — one button fills the matrix with integers drawn from [0, 10].
  Presets are games somebody chose because they illustrate something, which is a biased
  sample; this is the cheap way to ask what survives a game nobody designed.
- **Starting proportions**: a percentage box per strategy (equal shares by default,
  rescaled to sum to 100), or structured starts — single mutant, patch, stripes,
  checkerboard, halves — or any array you build yourself.
- **Runs up to 1000 generations**, with a one-click random seed.
- **Seven update rules**, grouped by what they assume about the players — the grouping is
  itself the lesson, because a rule is a behavioural hypothesis, not a technicality:
  - *imitation, deterministic*: imitate the best (the Nowak–May rule);
  - *optimisation*: myopic best response — the only **innovative** rule, able to introduce a
    strategy nobody nearby is playing;
  - *imitation, stochastic*: proportional imitation, pairwise comparison (Fermi, intensity
    `beta`), local replicator — whose well-mixed limit **is** the replicator dynamic, which
    is what makes this tool directly comparable with the replicator explorer;
  - *birth and death*: **death–birth** (random death, birth proportional to payoff —
    selection acts on birth) and **death–selection** (doing badly kills you, birth from a
    random neighbour — selection acts on death). Both take a selection strength `w` from
    full strength down to neutral drift.
- **Scheduling**: synchronous, random subset (probability `p`), or strictly sequential —
  and for the birth–death rules the schedule is also *who dies*.
- **Realized assortment `r`**, measured every generation (see below).
- **Reproducibility**: one seed drives every random choice; the whole configuration encodes
  into a URL so a run can be shared and reproduced site for site.
- **Export**: the run as CSV, the dashboard as PNG.
- Mutation, self-interaction, total-vs-average payoff, and structured or hand-built starting
  lattices (random, single mutant, patch, stripes, checkerboard, halves, or any array).

## The point of it: assortment

Space has no direct effect on which strategies win. It works by producing **assortment** —
making like meet like — and assortment is the lever. So the tool measures it:

```
r = (P_same - P_random) / (1 - P_random)
```

`P_same` is how often a site and a neighbour play the same strategy; `P_random` is what you
would get by reshuffling the very same strategies over the lattice. `r = 0` means the lattice
is doing nothing; `r > 0` is the altruism lever (Eshel & Cavalli-Sforza); `r < 0` is the
mirror lever behind spite.

**This is the same `r` as the replicator explorer's slider.** Read `r` off a lattice run, put
that number into the well-mixed model, and ask whether it predicts what the lattice did. If it
does, space mattered *only* through assortment and you have explained the run rather than
watched it. If it does not, that gap is the finding. That cross-check is the exercise this
tool exists for.

## Two results it reproduces, and what they rest on

**Nowak & May (1992)** report cooperators settling near 0.318 at b = 1.85. Measured here
(torus, Moore-8, imitate-the-best, synchronous, random 50/50 start, generation 200):

| setting | cooperator frequency, 4 seeds |
|---|---|
| 100×100, self-interaction on | 0.308  0.339  0.297  0.337 |
| 150×150, self-interaction on | 0.315  0.329  0.310  0.308 |
| 60×60, self-interaction on | 0.000  0.321  0.318  0.002 |
| 40×40, self-interaction on | 0.000  0.004  1.000  0.282 |
| 40×40, self-interaction **off** | 0.000 every run |

Two assumptions are carrying it. **Self-interaction**: Nowak & May have each cell play its own
site as well as its neighbours (a 3×3 "territory"), which adds `A[i,i]` to everyone — and in
this game that is a flat bonus paid to cooperators only, worth about one extra neighbour.
Switch it off and cooperation is extinct for every b ≥ 1.6. **Lattice size**: below about
100×100 single runs fixate at 0 or 1 and the average across seeds is meaningless. A small
lattice is not a fast version of a big one.

*(The first is my measurement, not a citation. The published robustness critique of Nowak &
May targets the update rule — Roca, Cuesta & Sánchez find the lattice advantage largely
disappears for rules other than unconditional imitation — and Huberman & Glance's 1993 PNAS
objection targets synchronous updating. Both are testable here in about a minute.)*

**Lewis signalling** is the one to show a class. Nature picks one of two equiprobable states;
a sender sees it and emits one of two signals; a receiver sees only the signal and picks an
act; both score 1 if the act matches the state. Nothing connects any signal to any state in
advance. Each individual carries a rule for *both* roles, so there are 4 × 4 = **16
strategies**, named `sender|receiver` — `01|01` sends signal 0 in state 0, signal 1 in state 1,
then acts 0 on signal 0 and 1 on signal 1.

Exactly two of the sixteen are **signalling systems** (`01|01` and `10|10`), and the tool
confirms they are the only strict Nash equilibria. Starting from a random spread over all
sixteen (60×60 torus, Moore-8, imitate-the-best, 200 generations):

| seed | signalling systems | split | assortment r |
|---|---|---|---|
| 0 | 100% | 0.59 / 0.41 | +0.73 |
| 1 | 100% | 0.52 / 0.48 | +0.76 |
| 2 | 100% | 0.55 / 0.45 | +0.73 |
| 3 | 100% | 0.52 / 0.48 | +0.73 |
| 4 | 100% | 0.53 / 0.47 | +0.72 |

Meaning evolves *every time* — and the population settles into **both** systems at once, in
roughly equal shares, with a frozen boundary between them. That is the lesson in one picture:
that signals acquire meaning is not arbitrary, but *which* meaning they acquire is, and a
lattice can sustain two incompatible conventions side by side where a well-mixed population
has to choose one.

**Ohtsuki et al. (2006)** give b/c > k for cooperation under death–birth updating on a regular
graph of degree k. That is a **weak-selection** result and it does not survive being run at
full strength: at `selection = 1` the highest-paid neighbour nearly always wins the vacancy,
the rule collapses towards "imitate the best", and defection takes everything at every b/c.
Turn selection down and cooperation appears and rises with b/c. See `rule_death_birth`'s
docstring for the table — and note it is *consistent with* the condition, not a test of it,
since the published claim is about fixation probability and the table is a frequency.

## Running it

```bash
pip install numpy matplotlib marimo

python3 tests.py            # verify the model (should end "ALL TESTS PASSED")
marimo run app.py           # the student widget
marimo edit app.py          # edit the widget
jupyter notebook notebook.ipynb   # the coder walkthrough (or open in Colab)
```

## Deploying (one-time, instructor)

A GitHub Actions workflow (`.github/workflows/deploy.yml`) builds the WASM widget and
publishes it to Pages on every push.

```bash
# 1. From this folder, fill in your GitHub username + repo name everywhere:
OWNER=your-github-username   REPO=spatial-evolution-phil2001
grep -rl 'OWNER/REPO\|OWNER.github.io' . --include='*.md' --include='*.ipynb' \
  | xargs sed -i '' "s|OWNER/REPO|$OWNER/$REPO|g; s|OWNER.github.io/REPO|$OWNER.github.io/$REPO|g"

# 2. Install + authenticate the GitHub CLI (one-time browser login):
brew install gh
gh auth login

# 3. Create the public repo and push (this folder becomes the repo root):
git init && git add -A && git commit -m "Spatial evolution teaching tool"
gh repo create "$REPO" --public --source=. --push

# 4. Turn on Pages via Actions, then trigger the first deploy:
gh api -X POST "repos/$OWNER/$REPO/pages" -f build_type=workflow || true
git commit --allow-empty -m "trigger pages" && git push
```

Preview the exact hosted page locally before publishing:

```bash
python3 -m marimo export html-wasm app.py -o build --mode run
cp spatial.py render.py build/          # REQUIRED -- see below
python3 -m http.server --directory build
```

### ⚠ The one non-obvious deployment step

**`marimo export html-wasm` bundles the notebook, not the files it imports.** `spatial.py` and
`render.py` are *not* copied into the build, so in the browser `import spatial` fails, and the
failure is silent — you get a blank page and nothing in the console but an RPC timeout. The
two files must be copied next to `index.html`, which the workflow and the preview command
above both do.

`app.py` also has to fetch them at runtime using the **absolute** URL from
`mo.notebook_location()`; a relative path resolves against marimo's virtual filesystem and
quietly returns a few hundred bytes of something else. And it loads them through `importlib`
rather than a plain `import`, because marimo's WASM loader scans the source for imports and
tries to install anything it finds from PyPI — a bare `import spatial` sends it hunting for a
package called "spatial". The same scan is why `app.py` contains an otherwise pointless
`import matplotlib`: that is the only way matplotlib gets installed, since the real import of
it lives in `render.py` where the scanner never looks.

Verified working end to end in a browser on 2026-08-08: Pyodide boots, the modules load, the
simulation runs and draws, and the controls respond.

## Design decisions (and why they differ from GenLab)

1. **One model core in Python.** GenLab put the model in ~900 lines of Angular/TypeScript
   with a build chain, which made it effectively uneditable. Here the model is one commented
   module and the front-ends are thin.
2. **Seeded and shareable.** GenLab had no seed, so no run could be reproduced or handed in.
   Here one seed drives everything and the whole configuration encodes into a URL.
3. **Absent neighbours are absent.** On a bounded lattice GenLab scored off-grid neighbours
   as payoff 0, so edge cells were compared on *total* payoff against interior cells with
   more neighbours — and with negative payoffs the edge became the best place on the board.
   Here a missing neighbour is missing, and `payoff_mode="average"` divides by the neighbours
   you actually have.
4. **Rules that are meaningful on an interval scale.** Payoffs carry no absolute zero or
   unit, so a rule must give the same answer under `payoff -> alpha*payoff + c`. GenLab's
   probabilistic imitation drew a neighbour with probability `payoff_j / sum(payoffs)` — a
   ratio of interval-scale quantities, which divides by zero when payoffs sum to zero and
   inverts itself when they are negative. Every rule here is invariant under positive affine
   rescaling, with one documented exception (`fermi`, whose `beta` carries units of 1/payoff).
   `tests.py` checks this for all seven rules.
5. **Models that would be quietly wrong are refused.** A wrapped lattice smaller than its
   neighbourhood aliases neighbours onto each other — and a one-row wrapped lattice with a 2D
   neighbourhood makes a cell its own neighbour, twice. These now raise instead of running.
6. **Moore r=2 actually exists.** GenLab offered it in the menu and silently fell back to
   Moore-8.
7. **Canvas-style image rendering.** GenLab drew an HTML table of `<td>` elements, which does
   not scale; a 150×150 lattice is fine here.
8. **Asynchronous updating is a first-class option**, because whether a result survives a
   change of schedule is a question worth asking rather than a setting worth hiding.

## Provenance

Built 2026-08-08 for PHIL 2001 as a replacement for
[GenLab](https://chessmyers.github.io/GenLab/) (Christopher Myers), at Rory's request: same
job, cleaner model, editable. The birth–death and death–selection rules were added at his
suggestion to give the tool biologically grounded dynamics alongside the imitation rules.
