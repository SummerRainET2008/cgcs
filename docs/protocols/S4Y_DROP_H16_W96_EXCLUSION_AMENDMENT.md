# S4-Y DROP H16-W96 exclusion amendment

Date: 2026-11-08

## Decision

The H16-W96 condition is cancelled and excluded from Paper A because it is a
design error for this DROP evaluation. Increasing only the maximum number of
96-token Actions does not create the intended higher-frequency decision
process: observed trajectories generally terminate well before the H8 ceiling.

Paper A therefore uses only the complete H8-W96 matrix:

- 3 seeds;
- 4 Actors;
- 9,536 examples per cell;
- 12 complete cells and 114,432 evaluations.

No H16-W96 numerical result may enter the paper body, appendix, tables, figures,
statistical tests, claims, or conclusions. The paper analysis entry point is
restricted to H8-W96 and must ignore H16-W96 files even when they are present.

## Provenance retention

H16-W96 artifacts are retained only for internal audit provenance and must not
be deleted or promoted into the paper-analysis input:

- six complete H16-W96 cells remain covered by
  `experiments/s4y_drop_full_live/SHA256SUMS.complete_units`;
- two cancelled partial cells from the first machine are isolated under
  `experiments/s4y_drop_full_live/cancelled_partial_h16_w96_first_20261108T1934Z`;
- four cancelled partial cells plus logs and launch controls from the second
  machine are isolated under
  `experiments/s4y_drop_full_live/cancelled_partial_h16_w96_second_20261108T1931Z`.

The independent H8-W96-C768 versus H16-W48-C768 open-development diagnostic is
not H16-W96 and is governed by its separate exploratory protocol. It must also
remain outside the primary Paper A confirmatory table unless separately
