# S4-Y DROP retained-composite segmentation two-seed 2x2 protocol

Status: **frozen before execution; exploratory and non-confirmatory**

Date: 2026-11-08

## Question

This experiment tests whether inference-time action segmentation changes the
effect of the retained best-of-N E4 composite (Actor + value head + best-of-N
search) relative to its matched E0 composite. It is independent of the cancelled H16-W96 formal condition and must not
enter Paper A version-one results.

## Frozen matrix

Seeds 99803 and 99802 each contain four cells:

1. retained best-of-N/E0 composite with H8-W96-C768;
2. retained best-of-N/E0 composite with H16-W48-C768;
3. retained best-of-N/E4 composite with H8-W96-C768;
4. retained best-of-N/E4 composite with H16-W48-C768.

All eight cells use Qwen3-4B-Instruct-2507, the same 512 DROP
open-development examples in the same order, stochastic candidate generation with temperature 1.3, two candidates, epsilon
0.10, beta 1.0, expansion rate 1.0, maximum prompt length 1536, and head length
256. The evaluation seed equals the Actor-training seed.

Every state expands to two candidates and uses the retained value head for
selection. H16-W48 changes only max_steps and step_tokens; both settings retain
the same 768-token max_steps x step_tokens envelope. There is no new training.

## Frozen inputs

- Data SHA-256:
  `fab647e3a708c18702a7ddcfceed4a3cc58568d0ffaeaba23015b3a3dc458010`.
- Evaluator SHA-256:
  `d803271fa170ba8432333a2f2aacdbf7e52a787519b3cd6015f0e3724bff4d44`.
- DROP parser SHA-256:
  `af5a2b8ddcde681809bbcd2aa1da20d89f17838db86a8acb9ae2f3247cc6fd0a`.
- seed-99803 best-of-N/E0 adapter aggregate SHA-256:
  `558c92ee6643f4b49b8504aa5639ab643349dde99a20dc492e562cb51bbeffeb`.
- seed-99803 best-of-N/E4 adapter aggregate SHA-256:
  `a34d12e5563973ad6bdf104a71cb5eba4181c0476bb998d33bcbb7c9ab6f3c13`.
- seed-99802 best-of-N/E0 adapter aggregate SHA-256:
  `ce909d111be64ca07a909a6dd5c82698b16e2eecddd808211c59557d25e29c75`.
- seed-99802 best-of-N/E4 adapter aggregate SHA-256:
  `ce898138278d11cb99678a2565a5bc6199de337f6e7a634e32fbde97d1f4044e`.
- seed-99803 E0 value-head SHA-256: 5a16a9caf3431e8f5471e639f8b5fe6f24186637e58e67b181806873281a640e.
- seed-99803 E4 value-head SHA-256: 2d9ec88c662c2f9dec9992b012dbdeb2623c697d83ca9219231e9db258df19a9.
- seed-99802 E0 value-head SHA-256: a4973a6cec0436b2b7709c0bc9a7a2b3429160a4ff98202d690af09bd83c8fbd.
- seed-99802 E4 value-head SHA-256: 385d0d15981c94407cc957fe4da220138b8d6f8c7a3bda59b1874858f3f6c534.

## Frozen estimands

Report per seed and pooled across seeds:

1. E4 segmentation effect: E4(H16-W48) minus E4(H8-W96).
2. E0 segmentation effect: E0(H16-W48) minus E0(H8-W96).
3. Interaction: [E4 minus E0] under H16-W48 minus [E4 minus E0] under
   H8-W96.

Report exact match, F1, answer rate, conditional metrics, Actions, realized
completion tokens, termination reasons, token-cap hits, paired state
transitions, elapsed time, and GPU time. Use 10,000 problem-clustered bootstrap
samples with seed 20261108, keeping both seed observations for a problem
together. Report every result regardless of direction.

## Frozen scheduling and interpretation

All eight cells run concurrently across the two identical four-GPU machines:
seed 99803 on the first machine and seed 99802 on the second. Timing pressure
must not change any frozen setting.

This is a retained-composite inference-time segmentation interaction test. Because all Actors
were trained with the original 96-token Action structure, a negative result does
not rule out matched W48 training. It does rule against claiming that
inference-time resegmentation alone improves E4.
