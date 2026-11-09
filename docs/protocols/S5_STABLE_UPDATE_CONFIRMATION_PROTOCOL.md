# S5 Prospectively Frozen Stable-Update Confirmation

Frozen before S5 updates or sealed evaluation. Stage 1 Qwen3-4B only; no 14B or DROP rerun.

Question: after reducing update magnitude enough to pass prospectively defined clipping diagnostics, do E1/E3 still lack advantage, and does this persist from H8 to H16?

Fixed: Qwen/Qwen3-4B-Instruct-2507; GSM8K 48 examples/round at offsets 6548/6596; MATH-500 sealed offset 48 count 200 seed 100901; best_of_n with 2 candidates, temperature 1.3, epsilon .10, beta 1, expansion 1; byte-identical S3 burn-in and round-1 rollouts; PPO clip .2, KL .02, value .5, accumulation 32, LoRA 32/64, max grad norm 1. Arms are K1-E0, search-E0, E1 trajectory propensity, E3 step-cumulative lambda .75. E4 is excluded because this confirmation isolates correction estimators.

Open calibration: seed 99801 only, no sealed evaluation. Run all four arms at LR 1e-6 and 2e-6. A rate passes only if every arm has finite losses, positive updates, sequence clip fraction <.20, and selected raw-Actor token clip fraction <.20. Select the largest common passing rate. If neither passes, freeze 1e-6, mark gate failed, and continue without more tuning. Original S3 remains FAIL.

Sealed confirmation: after selection is written and hashed, run seeds 99802/99803. Each arm runs two adaptive rounds; round 1 reuses inherited rollout and round 2 recollects offset 6596. Evaluate final raw Actors at H8-W96 and H16-W96; no retained composite.

Report sequence/token clip and saturation, KL, ESS, max weight, updates, tokens and time; strict accuracy, answer rate, conditional accuracy, steps and termination. Paired bootstrap: E1-searchE0, E3-searchE0, E3-E1, H16-H8. Report all outcomes. No more LR search, seed replacement, estimator replacement, or eval expansion.
