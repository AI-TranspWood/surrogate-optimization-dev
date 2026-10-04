# AITW surrogate-based material optimization

Optimize microscopic biocomposite parameters using the bundled stiffness surrogate
and batched differential evolution. Match selected output targets, or minimize and
maximize selected quantities subject to confidence and uncertainty constraints.

The package follows the structure of `biocomposite-surrogate` and installs independently.
It does not require MATLAB or modify the surrogate project.

## Installation

Python **3.10** is required by the pinned TensorFlow stack. A Conda environment can
be created as follows (run the installation commands in this repository):

```bash
conda create -n aitw-material-optimization python=3.10 pip
conda activate aitw-material-optimization
python -m pip install .
```

Alternatively, use a Python 3.10 virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The runtime versions are declared in `pyproject.toml` and `requirements.txt`.
TensorFlow 2.10.0, Keras 2.10.0, TensorFlow Probability 0.18.0, NumPy 1.26.4,
and SciPy 1.11.4 are pinned. TensorFlow uses an available supported GPU without
requiring a separate optimization implementation; GPU drivers/runtime setup follows
the installed TensorFlow distribution. CPU execution is also supported.

## CLI

The CLI takes two JSON specification files and one result path:

```bash
aitw-material-optimization example-input/matching-material.json example-input/matching-outputs.json result-matching.json --mode matching
aitw-material-optimization example-input/report-material-2.json example-input/minimum-stiffness.json result-constrained.json --mode constrained
```

Use `--maxiter`, `--popsize`, `--tol`, and `--seed` to control differential evolution.
`popsize` is a multiplier per free variable, rather than a total particle count.
Defaults are `maxiter=20`, `popsize=200` for matching and `maxiter=10`,
`popsize=1000` for constrained optimization; both use `tol=1e-4`. Evaluation is
vectorized across the population, uses deferred updates, and disables polishing.
For a quick smoke run, append `--maxiter 1 --popsize 5 --seed 1`.

`--help` does not load TensorFlow. Invalid specifications produce a CLI error.
An infeasible returned candidate is written to the result file and produces a nonzero
exit code. A feasible candidate can still have optimizer status `notConverged` when
the iteration limit is reached; inspect both feasibility and convergence.

## Basic examples

Run these commands from the repository root after installing the package. Each uses
a complete input file in `example-input`; the input snippets below show only the
parameters selected for optimization. Keep all 18 parameters in your own input file.

### Match stiffness targets

Choose fiber volume fraction and microfibril angle to bring the predicted median
`E_T` towards 10 GPa and median `mu_LT` towards 4 GPa. The relevant entries in
`matching-material.json` are:

```json
{"microfibrilAngle": "optimize", "fiberVolumeFraction": "optimize"}
```

`matching-outputs.json` supplies the targets:

```json
{"outputs": {"E_T": {"target": 10}, "mu_LT": {"target": 4}}}
```

```bash
aitw-material-optimization example-input/matching-material.json example-input/matching-outputs.json result-matching.json --mode matching --seed 1
```

Both selected inputs can change; all other supplied inputs stay fixed before
preprocessing. Targets are fitted using the matching loss, rather than enforced as
exact equalities. No objective weights are needed.

### Minimize fiber volume with a stiffness requirement

Find the smallest fiber volume fraction whose predicted `E_T` interval stays above
10 GPa. In `report-material-2.json`, the selected input is:

```json
{"fiberVolumeFraction": "minimize"}
```

`minimum-stiffness.json` defines the condition:

```json
{"E_T": {"min": 10, "max": "", "confidence": 0.5}}
```

```bash
aitw-material-optimization example-input/report-material-2.json example-input/minimum-stiffness.json result-minimum-fiber.json --mode constrained --seed 1
```

The loss is simply `fiberVolumeFraction`, with default weight 1. At confidence 0.5,
the interval runs from the 25th to the 75th percentile; its lower endpoint must be
at least 10 GPa. The empty `max` imposes no upper bound. This is a constraint on
the interval, rather than a stiffness reward added to the loss.

### Maximize stiffness with an uncertainty limit

Choose fiber volume fraction to maximize median `E_T`, while keeping the width of
its central 50% interval at most 5 GPa. In `maximum-material.json`, use:

```json
{"fiberVolumeFraction": "optimize"}
```

`maximum-outputs.json` supplies the output objective and constraint:

```json
{"outputs": {"E_T": {"objective": "maximize", "maxUncertainty": 5, "confidence": 0.5}}}
```

```bash
aitw-material-optimization example-input/maximum-material.json example-input/maximum-outputs.json result-maximum-stiffness.json --mode constrained --seed 1
```

The loss is `-median(E_T)`: minimizing its negative maximizes stiffness. Fiber
volume can change but is not itself rewarded or penalized. Check `feasible` and
`optimizer.status` in each result file before using the returned parameters.

## Input specification

The first file must contain all 18 named parameters, using the names in
`example-input/report-material-1.json`. Each continuous parameter is either:

- A number: keep the supplied physical value fixed before model preprocessing.
- `"optimize"`: choose its value during optimization.
- `"minimize"` or `"maximize"`: choose its value and include it as a directed
  objective in constrained mode.

The orientation distribution remains fixed: use `0` or `"vonMises"`, or `1` or
`"vonMisesFisher"`. All other parameters can be selected, including composition
contents. With no free parameters, the material is evaluated once without a search.

| Parameter | Minimum | Maximum |
| --- | ---: | ---: |
| fiberAspectRatio | 1.2 | 20000 |
| orientationConcentration | 0 | 100 |
| microfibrilAngle | 0 | 45 |
| lumenPorosity | 0 | 0.99 |
| fiberVolumeFraction | 0 | 1 |
| celluloseContent | 0.05 | 0.9 |
| hemicelluloseContent | 0.01 | 0.5 |
| ligninContent | 0.01 | 0.6 |
| pectinContent | 0 | 0.15 |
| extractivesContent | 0 | 0.15 |
| ashContent | 0 | 0.20 |
| celluloseCrystallinity | 0.20 | 1 |
| matrixYoungsModulus | 0 | 100 |
| matrixPoissonRatio | 0 | 0.5 |
| airPorosity | 0 | 0.5 |
| tangentialInterfaceCompliance | 0 | 100000 |
| longitudinalInterfaceCompliance | 0 | 100000 |

Young's and shear moduli use GPa, angles use degrees, and interface compliances
use 1/GPa. Aspect ratio, orientation concentration, and the two interface compliances
are scaled logarithmically; the others use linear scaling. Physical zero is accepted
for the three zero-bounded logarithmic inputs and mapped to `1e-10` for scaling.
Positive values below `1e-10` use the same practical floor. Free logarithmic variables
search from that floor, so their reported minimum is `1e-10`.

## Output specification and loss

The second file selects any subset of the model's five outputs: `E_T`, `E_L`,
`nu_T`, `nu_LT`, and `mu_LT`. Use an `outputs` object. In **matching** mode, set
`target` on one or more selected outputs, as in the basic matching example above.

The matching loss is the mean of squared residuals for selected targets, using natural
logarithms for moduli and linear residuals for Poisson's ratios, retaining the original
optimizer's transformation convention. Modulus targets must be positive. Directed
objectives and weights are not allowed in matching mode.

In **constrained** mode, specify at least one directed input or output objective.
Selected outputs use `"objective": "minimize"` or `"maximize"`. The minimized
loss is a weighted sum of physical input values and output medians, with maximization
terms negated. Targets are not allowed in this mode. Both modes accept constraints.

A flat output map is also accepted:

```json
{"E_T": {"min": 10, "max": "", "confidence": 0.5}}
```

Empty strings mean unspecified bounds. Omit unused outputs or give them empty objects.
Unknown names are errors; the five supported names include shear modulus `mu_LT`.

### Multiple objectives and weights

Use weights when you explicitly want to trade one quantity against another. For example,
to minimize fiber volume and maximize stiffness at the same time, use `"minimize"`
for `fiberVolumeFraction` in the input file and set both objective weights:

```json
{
  "outputs": {
    "E_T": {"objective": "maximize", "weight": 1, "maxUncertainty": 5, "confidence": 0.5}
  },
  "inputWeights": {"fiberVolumeFraction": 20}
}
```

This is the specification in `weighted-outputs.json`. Run it with the input file
whose fiber volume fraction is marked `"minimize"`:

```bash
aitw-material-optimization example-input/report-material-2.json example-input/weighted-outputs.json result-weighted.json --mode constrained --seed 1
```

Its loss is `20 * fiberVolumeFraction - median(E_T)`. The weight 20 applies only
to fiber fraction; stiffness has weight 1. Increasing fiber fraction by 0.01 must
increase stiffness by 0.2 GPa to keep this loss unchanged. These weights are arbitrary
example values, not recommended material-design settings. Since quantities have
different units, choose weights to reflect their reference scales and your intended
tradeoff. The software does not normalize the objectives automatically.

Writing `20 * (fiberVolumeFraction - median(E_T))` would give both terms weight 20;
that overall multiplier does not change the minimizing solution. Multiple directed
objectives require an explicit positive weight for every objective. A single directed
objective defaults to weight 1, as in the basic examples.

## Confidence and uncertainty constraints

Each output can specify `min`, `max`, `confidence`, and `maxUncertainty`. At confidence
`c`, use its **central prediction interval**, with quantiles `(1-c)/2` and `(1+c)/2`.
Confidence must be strictly between 0 and 1 and defaults to 0.5. Consequently:

- `min`: the interval's lower endpoint must be at least this value.
- `max`: the interval's upper endpoint must be at most this value.
- `maxUncertainty`: the interval's width must not exceed this value.

All limits and widths use physical output units. Both endpoints must satisfy supplied
bounds. Modulus distributions are exponentiated from the model's log space; Poisson
ratios use their own linear distributions. Infeasible or nonfinite candidates receive
infinite internal loss. The uncertainty width has a precise meaning here; it is not
a one-sided probability constraint.

## Python API and results

```python
from aitw_material_optimization import optimize
from aitw_material_optimization import dataLoaders as dl

result = optimize(
    dl.loadInput("example-input/report-material-2.json"),
    dl.loadInput("example-input/minimum-stiffness.json"),
    mode="constrained",
    settings={"maxiter": 10, "popsize": 1000, "tol": 1e-4, "seed": 1},
)
dl.saveResult(result, "result.json")
```

The API accepts dictionaries, raises `ValueError` for invalid specifications, and
returns a JSON-compatible dictionary. An optional `model` argument lets callers reuse
an already-loaded model. Results contain:

- `parameters`: physical candidate values before preprocessing; orientation is a name.
- `effectiveModelInput`: the centered 18-by-1 tensor actually evaluated.
- `predictions`: all five medians, lower/upper bounds, confidence, and quantiles.
- `constraints`: each supplied limit and whether its condition passed.
- `objectiveValue`, `feasible`, and `optimizer`: status, convergence, message,
  iterations, evaluation calls, and the search objective before final re-evaluation.

Evaluation counts refer to model batches, including the final evaluation, rather than
the number of candidate materials. Nonfinite numerical results are serialized as `null`.
`noFeasibleSolution` means the returned candidate failed the final checks; it is not
proof that the problem has no feasible solution.

The Flipout layer is stochastic. Reported objective, predictions, and feasibility come
from a single final evaluation and may differ from the last search evaluation.
`seed` controls differential evolution and seeds TensorFlow when constructing a bundled
model; it does not guarantee reproducibility across devices or reseed a supplied model.

### Preserved preprocessing

The original loader normalizes the **scaled** cellulose-through-extractives entries
(five contents, excluding ash). The optimizer then normalizes all six scaled contents,
including ash, before centering by 0.5. This convention is preserved and applied to
each candidate. The effective composition tensor can therefore differ from fixed
physical contents and is not a direct physical fraction representation. For newly free
composition entries, both stages are applied after assembling each physical candidate.
A candidate with a zero normalization denominator is infeasible. The interface does
not impose new physical composition constraints or retrain the surrogate.

## Examples and validation

`report-material-1.json` and `report-material-2.json` reproduce the report's two input
specifications, with fiber volume minimized and the `minimum-stiffness.json` constraint.
`matching-*` demonstrates partial target matching including shear modulus; `maximum-*`
demonstrates maximizing stiffness with an uncertainty limit. Pair `weighted-outputs.json`
with `report-material-1.json` for a weighted input/output tradeoff.

The report's fiber-volume estimates 0.3205 and 0.5897 are illustrative, not exact
regression values: its confidence interpretation is unspecified and the surrogate
evaluations are stochastic. Here, confidence follows the central-interval definition above.
The report's comparisons with the multiscale model are not reproduced by this package.

```bash
python -m pip install '.[test,release]'
python -m pytest -q
python -m flit build
```

Tests exercise objectives, distribution mappings, confidence constraints, validation,
serialization, and CLI/API agreement. Tests requiring SciPy or TensorFlow are skipped
when those dependencies are absent; passing the reduced suite alone does not verify
real differential evolution or the bundled model.

Verification on 2026-10-03 with Python 3.10.20 and the pinned numerical stack passed
all 57 tests, including real differential evolution and the bundled model. Wheel and
source distributions were built, and the installed CLI was exercised outside the
repository on CPU. With `--maxiter 10 --popsize 100 --seed 1`, the second report
example returned a feasible fiber volume fraction of approximately 0.65357 and an
`E_T` lower bound of 10.00005 GPa, while reaching the iteration limit. The first report
example returned no feasible candidate with those settings. These are observed runs,
not fixed regression expectations or proof of feasibility/infeasibility for other runs.
