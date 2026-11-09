"""Exact, CPU-only mechanism study; not an LLM or tau-bench evaluation.

An unmasked tool workflow exposes state-distribution and continuation-value
mismatch. Enumeration integrates candidate randomness, so results have no
Monte Carlo error. The critic is an oracle plus controlled noise, not learned.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np


def workflow(horizon: int, seed: int):
    # States: no evidence, order known, product known, mutation done,
    # successful termination, failed termination. Every tool remains available.
    # Actions: order lookup, product lookup, mutation, respond, think.
    rng = np.random.default_rng(seed)
    next_state = np.tile(np.arange(6)[:, None], (1, 5))
    for stage in range(3):
        next_state[stage, stage] = stage + 1
    next_state[:3, 3] = 5
    next_state[3, 3] = 4
    rewards = np.zeros((6, 5))
    rewards[3, 3] = 1.0
    logits = rng.normal(0, 0.25, (horizon, 6, 5))
    logits[:, :, 4] += 0.6
    for stage in range(4):
        logits[:, stage, stage] += 1.0
    actor = np.exp(logits - logits.max(axis=-1, keepdims=True))
    actor /= actor.sum(axis=-1, keepdims=True)
    return actor, next_state, rewards


def evaluate(policy, next_state, rewards):
    horizon, states, _ = policy.shape
    values = np.zeros((horizon + 1, states))
    q = np.zeros_like(policy)
    occupancy = np.zeros((horizon + 1, states))
    occupancy[0, 0] = 1.0
    for t in range(horizon - 1, -1, -1):
        q[t] = rewards + values[t + 1, next_state]
        values[t] = (policy[t] * q[t]).sum(axis=-1)
    for t in range(horizon):
        for state in range(states):
            np.add.at(occupancy[t + 1], next_state[state],
                      occupancy[t, state] * policy[t, state])
    return values, q, occupancy


def candidate_moments(actor, scores, k=3, beta=4.0, epsilon=0.2):
    """Return E[q], E[q*w] and E[q*w^2], grouped by executed action."""
    tuples = np.asarray(list(product(range(actor.shape[-1]), repeat=k)))
    measures = [np.zeros_like(actor) for _ in range(3)]
    for t in range(actor.shape[0]):
        for state in range(6):
            # Absorbing states involve no further model/tool calls or weights.
            if state >= 4:
                for measure in measures:
                    measure[t, state] = actor[t, state]
                continue
            p = np.prod(actor[t, state, tuples], axis=1)
            logits = beta * scores[t, state, tuples]
            guided = np.exp(logits - logits.max(axis=1, keepdims=True))
            guided /= guided.sum(axis=1, keepdims=True)
            selector = (1 - epsilon) * guided + epsilon / k
            weight = 1.0 / (k * selector)
            for power, measure in enumerate(measures):
                for i in range(k):
                    np.add.at(measure[t, state], tuples[:, i],
                              p * selector[:, i] * weight[:, i] ** power)
    return measures


def return_moment(measure, next_state, rewards, power):
    """Moment of trajectory-IS terminal success; reward occurs at most once.

    Weights stop at absorption. For this binary terminal reward, success^2
    equals success and the recursion integrates the appropriate weight power.
    """
    value = np.zeros(6)
    for t in range(len(measure) - 1, -1, -1):
        value = (measure[t] * (rewards ** power + value[next_state])).sum(-1)
    return float(value[0])


def gradient(actor, occupancy, action_measure, advantage):
    # Exact mean of score-function updates for independent tabular logits.
    terms = action_measure * advantage
    result = occupancy[:-1, :, None] * (
        terms - actor * terms.sum(-1, keepdims=True))
    result[:, 4:] = 0
    return result


def run_case(horizon, seed, noise, family, k=3, beta=4.0, epsilon=0.2):
    actor, nxt, reward = workflow(horizon, seed)
    v_pi, q_pi, d_pi = evaluate(actor, nxt, reward)
    advantage = q_pi - v_pi[:-1, :, None]
    perturbation = np.random.default_rng(seed + 10000).normal(size=actor.shape)
    estimate = q_pi + noise * perturbation
    scores = estimate if family == "return" else np.abs(
        estimate - (actor * estimate).sum(-1, keepdims=True))
    mu, corrected, second = candidate_moments(actor, scores, k, beta, epsilon)
    v_mu, q_mu, d_mu = evaluate(mu, nxt, reward)
    assert np.allclose(corrected, actor, atol=1e-12)
    oracle_gradient = gradient(actor, d_pi, actor, advantage)
    norm = np.linalg.norm(oracle_gradient)
    methods = {
        "uniform": (actor, actor, d_pi, actor, advantage),
        "guided_uncorrected": (mu, mu, d_mu, mu, advantage),
        "guided_current_only": (corrected, second, d_mu, corrected, advantage),
        "guided_full_prefix": (corrected, second, d_pi, corrected, advantage),
        "guided_full_prefix_wrong_q_mu": (
            corrected, second, d_pi, corrected, q_mu - v_mu[:-1, :, None]),
    }
    outputs = {}
    for name, (first, second_measure, occupancy, action_measure, adv) in methods.items():
        g = gradient(actor, occupancy, action_measure, adv)
        if name == "guided_current_only":
            mean = float(np.sum(d_mu[:-1, :, None] * first * reward))
            moment2 = float(np.sum(d_mu[:-1, :, None] * second_measure * reward))
        else:
            mean = return_moment(first, nxt, reward, 1)
            moment2 = return_moment(second_measure, nxt, reward, 2)
        outputs[name] = {
            "estimated_success_expectation": mean,
            "success_bias": mean - float(v_pi[0, 0]),
            "return_estimator_variance": max(0.0, moment2 - mean * mean),
            "gradient_relative_bias": float(np.linalg.norm(g - oracle_gradient) / norm),
        }
    return {
        "horizon": horizon, "seed": seed, "critic_noise_std": noise,
        "selector_family": family, "actor_success": float(v_pi[0, 0]),
        "guided_success": float(v_mu[0, 0]),
        "q_mu_minus_q_pi_rms_under_actor": float(np.sqrt(
            np.sum(d_pi[:-1, :, None] * actor * (q_mu - q_pi)**2) / horizon)),
        "methods": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[91001, 91002, 91003])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.horizons) < 4 or not 1 <= args.candidates <= 5:
        parser.error("horizons must be >=4 and candidates between 1 and 5")
    if not 0 < args.epsilon <= 1 or args.beta < 0 or min(args.noise) < 0:
        parser.error("require epsilon in (0,1], beta >=0, noise >=0")
    cases = [run_case(h, s, n, f, args.candidates, args.beta, args.epsilon)
             for h in args.horizons for s in args.seeds for n in args.noise
             for f in ("return", "absolute_advantage")]
    payload = {"kind": "exact_synthetic_tool_workflow_not_llm_benchmark",
               "config": {k: v for k, v in vars(args).items() if k != "output"},
               "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    report = [
        "# Exact multistep tool-workflow mechanism study", "",
        "CPU-only synthetic experiment, not an LLM/Agent benchmark result. "
        "All expectations are enumerated exactly; no sampling confidence interval is needed.", "",
        "Workflow: order lookup -> product lookup -> mutation -> successful response. "
        "All five actions remain available at all nonterminal states; invalid calls "
        "do nothing and premature responses fail. Reward is terminal binary success, gamma=1.", "",
        f"Grid: {len(cases)} cases, horizons {args.horizons}, actor seeds {args.seeds}, "
        f"critic noise {args.noise}, K={args.candidates}, beta={args.beta}, epsilon={args.epsilon}. "
        "Critics are exact Q^pi plus controlled noise, not learned models.", "",
        "## Oracle return-oriented selector (mean across actor seeds)", "",
        "| Horizon | Actor success | Guided success | Current-only return bias (pp) | "
        "Current-only relative gradient bias | Full-prefix relative gradient bias | "
        "Full-prefix with wrong Q^mu relative gradient bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for h in args.horizons:
        subset = [c for c in cases if c["horizon"] == h and c["critic_noise_std"] == 0
                  and c["selector_family"] == "return"]
        if not subset:
            continue
        def avg(key, method=None):
            return float(np.mean([c["methods"][method][key] if method else c[key]
                                  for c in subset]))
        report.append(
            f"| {h} | {100*avg('actor_success'):.2f}% | {100*avg('guided_success'):.2f}% | "
            f"{100*avg('success_bias', 'guided_current_only'):+.2f} | "
            f"{avg('gradient_relative_bias', 'guided_current_only'):.3f} | "
            f"{avg('gradient_relative_bias', 'guided_full_prefix'):.2e} | "
            f"{avg('gradient_relative_bias', 'guided_full_prefix_wrong_q_mu'):.3f} |")
    report.extend([
        "", "## Estimator definitions and limits", "",
        "The true actor gradient is computed by exact occupancy and Q^pi. "
        "Current-only updates weight each score-times-advantage by w_t but retain "
        "behavior state occupancy. Full-prefix updates use product(w_0,...,w_t) "
        "with oracle A^pi. The wrong-Q diagnostic changes only that advantage to A^mu. "
        "Its return-estimation fields stay unchanged; only its gradient is wrong.", "",
        "Success estimators compare unweighted terminal reward, reward weighted "
        "only at its occurrence, and reward weighted by the full prefix. "
        "Reported return variance is NOT full gradient variance. Relative gradient "
        "bias is ||E[g_hat]-grad J(pi)|| / ||grad J(pi)||.", "",
        "Candidate enumeration verifies E[q*w] grouped by action equals pi. "
        "This experiment establishes identities and failure modes, not compute "
        "savings, learned-critic quality, PPO improvement, or transfer to real tools. "
        "The tabular policy changes with time; absorbing states incur no more calls.", "",
        "Reproduce: `PYTHONPATH=src .venv/bin/python -m acbpo.multistep_tool_study "
        "--output results/multistep_agent_mechanism/summary.json`", "",
    ])
    args.output.with_suffix(".md").write_text("\n".join(report))
    print(json.dumps({"cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
