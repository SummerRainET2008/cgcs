# Exact compute-matched gate and search-family study

## Main confirmation result

Thirty committed confirmation MDPs were evaluated at K=2 and K=4 under expected-candidate budgets 1.5 and 2.0. Every gate used exactly matched realized expansion under its own induced state occupancy. Uniform and Value selectors were both retained so gate allocation was not confounded with candidate selection.

At K=2, budget 2.0 means expansion probability one; all gates are therefore intentionally identical. The three non-degenerate conditions show:

| Setting | Entropy vs random | Value vs random | Value-and-entropy vs random | Oracle vs random |
|---|---:|---:|---:|---:|
| K2, B1.5 | +0.01034 [0.00101, 0.02030] | -0.00646 [-0.01765, 0.00421] | +0.00057 [-0.00813, 0.00966] | +0.04656 [0.03872, 0.05487] |
| K4, B1.5 | +0.01048 [0.00249, 0.01908] | +0.00698 [-0.00302, 0.01657] | +0.00790 [-0.00033, 0.01654] | +0.05194 [0.04257, 0.06163] |
| K4, B2.0 | +0.01575 [0.00298, 0.02944] | +0.00079 [-0.01553, 0.01639] | +0.00594 [-0.00670, 0.01874] | +0.07886 [0.06614, 0.09221] |

Values are paired composite-return differences with 95% problem-clustered bootstrap intervals. Entropy gating is the only deployable simple gate whose interval is positive in all three non-degenerate settings. Low Value alone, the direct Value-and-entropy heuristic, and the preregistered four-feature learned gate do not show stable confirmation gains.

The oracle gate wins 30/30 MDPs in every non-degenerate setting, demonstrating substantial allocation headroom. Thus the negative result is feature insufficiency, not absence of value-of-computation structure.

## Primary learned gate

The frozen linear features `[1, low_value, entropy, low_value*entropy]` had development-test Pearson -0.0078 for K=2 and -0.0125 for K=4. Its confirmation return was essentially random gating. The result is retained unchanged; the intuitive four-quadrant framing is not, by itself, a sufficient predictor of exact expansion gain.

## Frozen feature-sufficiency diagnostic

The protocol's four development-only richer groups isolate why the primary gate fails:

| Candidate count | Feature group | Development-test Pearson | Mean gate AUC | Return vs random | Wins |
|---:|---|---:|---:|---:|---:|
| 2 | Value/entropy degree 3 | -0.0470 | 0.5245 | -0.00009 | 5/10 |
| 2 | Actor confidence | 0.0424 | 0.5825 | +0.00013 | 6/10 |
| 2 | Critic shape | **0.9942** | **0.9880** | **+0.04061** | **10/10** |
| 4 | Value/entropy degree 3 | -0.0451 | 0.5097 | -0.00013 | 7/10 |
| 4 | Actor confidence | 0.0412 | 0.5350 | +0.00036 | 6/10 |
| 4 | Critic shape | **0.9867** | **0.9929** | **+0.10438** | **10/10** |

The useful variables are policy-weighted dispersion and shape of the per-Action Critic values, not the scalar state Value. These are diagnostic features rather than free deployment signals: an LLM implementation must obtain them from already generated candidates, an uncertainty head, or an ensemble, and must charge that cost to the compute budget.

## Diversity and isolation controls

- Expected unique candidates were about 1.75 for K=2 and 2.81--2.91 for K=4.
- A uniform selector exactly recovers the raw Actor executed-action distribution under every gate. Consequently, gate-only rows have no return gain; they isolate allocation and compute without silently attributing selector gains to the gate.
- The random-expansion/Value-selection row is a compute-matched stochastic
  allocation baseline.  A legacy `best_of_n` alias in the frozen JSON copies this
  row, but it is noncanonical because the protocol reserves best-of-N for
  always-expanded search.  It is excluded from best-of-N claims; see
  `SEARCH_FAMILY_CONFORMANCE_AUDIT.md`.

## Artifacts

- `GATE_DEVELOPMENT.json` — SHA-256 `81a3c99a74127ceec8f6851cc29aa010c0a9ccc0aedc10fe4781c23f223a9990`
- `GATE_CONFIRMATION.json` — SHA-256 `1018ef234ea1b3b2010cb60b1046f4b2ee4593ad1b0af7eebfadf90e76b578c7`
- `GATE_FEATURE_SUFFICIENCY.json` — SHA-256 `b4c9c680429d95b70c01bcb6fe14db482bb4e6767d968860cd5b0d2afe0566a4`
