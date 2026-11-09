from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .tau_retail import DirectRetailEnv


MUTATION_ACTIONS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _gold_tools_by_task(
    tau_root: Path, split: str, task_ids: Iterable[int]
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for task_id in sorted(set(task_ids)):
        result[task_id] = set(
            DirectRetailEnv(tau_root, split, task_id).gold_action_names()
        ) - {"respond"}
    return result


def analyze_selector(
    episodes: list[dict[str, Any]],
    gold_tools: dict[int, set[str]] | None = None,
) -> dict[str, Any]:
    steps = [step for episode in episodes for step in episode["trajectory"]]
    action_counts = Counter(step["selected"] for step in steps)
    action_rows: dict[str, dict[str, Any]] = {}
    for action, count in sorted(action_counts.items(), key=lambda item: (-item[1], item[0])):
        action_steps = [step for step in steps if step["selected"] == action]
        action_rows[action] = {
            "count": count,
            "share": _safe_rate(count, len(steps)),
            "tool_error_rate": mean(float(step["tool_error"]) for step in action_steps),
            "repeat_rate": mean(float(step["repeated_action"]) for step in action_steps),
            "positive_progress_rate": mean(
                float(step["progress_after"] > step["progress_before"])
                for step in action_steps
            ),
        }

    actor_probabilities = [
        float(step["actor_probability"])
        for step in steps
        if "actor_probability" in step
    ]
    order_id_steps = [
        step
        for step in steps
        if isinstance(step.get("arguments"), dict)
        and "order_id" in step["arguments"]
    ]
    valid_order_id_steps = [
        step
        for step in order_id_steps
        if re.fullmatch(r"#W\d+", str(step["arguments"]["order_id"]))
    ]
    missing_hash_order_id_steps = [
        step
        for step in order_id_steps
        if re.fullmatch(r"W\d+", str(step["arguments"]["order_id"]))
    ]
    summary: dict[str, Any] = {
        "episodes": len(episodes),
        "steps": len(steps),
        "success_rate": mean(float(episode["reward"]) for episode in episodes),
        "timeout_rate": mean(
            float(not episode["completed_naturally"]) for episode in episodes
        ),
        "zero_database_progress_rate": mean(
            float(episode["database_progress"] == 0.0) for episode in episodes
        ),
        "tool_error_rate": mean(float(step["tool_error"]) for step in steps),
        "repeat_rate": mean(float(step["repeated_action"]) for step in steps),
        "positive_progress_rate": mean(
            float(step["progress_after"] > step["progress_before"]) for step in steps
        ),
        "all_candidates_same_rate": mean(
            float(step["candidate_count_unique"] == 1) for step in steps
        ),
        "diverse_candidates_rate": mean(
            float(step["candidate_count_unique"] > 1) for step in steps
        ),
        "think_share": _safe_rate(action_counts["think"], len(steps)),
        "mutation_share": _safe_rate(
            sum(action_counts[action] for action in MUTATION_ACTIONS), len(steps)
        ),
        "order_id_argument_count": len(order_id_steps),
        "valid_order_id_format_rate": _safe_rate(
            len(valid_order_id_steps), len(order_id_steps)
        ),
        "missing_hash_order_id_rate": _safe_rate(
            len(missing_hash_order_id_steps), len(order_id_steps)
        ),
        "missing_hash_order_id_error_rate": _safe_rate(
            sum(step["tool_error"] for step in missing_hash_order_id_steps),
            len(missing_hash_order_id_steps),
        ),
        "action_counts": dict(action_counts.most_common()),
        "actions": action_rows,
    }
    if actor_probabilities:
        summary["selected_actor_probability_mean"] = mean(actor_probabilities)
        summary["selected_actor_probability_median"] = median(actor_probabilities)

    if gold_tools is not None:
        gold_step_count = 0
        successful_gold_step_count = 0
        candidate_gold_opportunities = 0
        selected_gold_from_candidate_opportunities = 0
        coverage: list[float] = []
        successful_coverage: list[float] = []
        for episode in episodes:
            required = gold_tools[episode["task_id"]]
            selected = {
                step["selected"]
                for step in episode["trajectory"]
                if step["selected"] != "respond"
            }
            selected_successfully = {
                step["selected"]
                for step in episode["trajectory"]
                if step["selected"] != "respond" and not step["tool_error"]
            }
            if required:
                coverage.append(len(required & selected) / len(required))
                successful_coverage.append(
                    len(required & selected_successfully) / len(required)
                )
            for step in episode["trajectory"]:
                if step["selected"] in required:
                    gold_step_count += 1
                    successful_gold_step_count += int(not step["tool_error"])
                candidates = step.get("candidate_actions")
                if candidates is not None and required.intersection(candidates):
                    candidate_gold_opportunities += 1
                    selected_gold_from_candidate_opportunities += int(
                        step["selected"] in required
                    )
        summary.update(
            {
                # This is a tool-name membership proxy, not a next-action oracle:
                # it ignores ordering and arguments.
                "gold_tool_step_rate": _safe_rate(gold_step_count, len(steps)),
                "gold_tool_success_rate": _safe_rate(
                    successful_gold_step_count, gold_step_count
                ),
                "mean_gold_tool_coverage": mean(coverage) if coverage else 0.0,
                "mean_successful_gold_tool_coverage": (
                    mean(successful_coverage) if successful_coverage else 0.0
                ),
                "gold_tool_candidate_step_rate": _safe_rate(
                    candidate_gold_opportunities,
                    sum(
                        "candidate_actions" in step
                        for episode in episodes
                        for step in episode["trajectory"]
                    ),
                ),
                "selected_gold_tool_given_candidate_rate": _safe_rate(
                    selected_gold_from_candidate_opportunities,
                    candidate_gold_opportunities,
                ),
            }
        )
    return summary


def shared_prefix_disagreements(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    gold_tools: dict[int, set[str]] | None = None,
) -> dict[str, Any]:
    baseline_map = {
        (episode["seed"], episode["task_id"]): episode for episode in baseline
    }
    treatment_map = {
        (episode["seed"], episode["task_id"]): episode for episode in treatment
    }
    disagreements: list[dict[str, Any]] = []
    identical = 0
    for key in sorted(set(baseline_map) & set(treatment_map)):
        left = baseline_map[key]
        right = treatment_map[key]
        found = False
        for index, (left_step, right_step) in enumerate(
            zip(left["trajectory"], right["trajectory"])
        ):
            left_choice = (left_step["selected"], left_step["arguments"])
            right_choice = (right_step["selected"], right_step["arguments"])
            if left_choice == right_choice:
                continue
            required = gold_tools[left["task_id"]] if gold_tools is not None else set()
            disagreements.append(
                {
                    "seed": key[0],
                    "task_id": key[1],
                    "step": index,
                    "baseline_action": left_step["selected"],
                    "treatment_action": right_step["selected"],
                    "baseline_tool_error": bool(left_step["tool_error"]),
                    "treatment_tool_error": bool(right_step["tool_error"]),
                    "baseline_positive_progress": bool(
                        left_step["progress_after"] > left_step["progress_before"]
                    ),
                    "treatment_positive_progress": bool(
                        right_step["progress_after"] > right_step["progress_before"]
                    ),
                    "baseline_is_gold_tool": left_step["selected"] in required,
                    "treatment_is_gold_tool": right_step["selected"] in required,
                }
            )
            found = True
            break
        if not found and len(left["trajectory"]) == len(right["trajectory"]):
            identical += 1

    result: dict[str, Any] = {
        "paired_episodes": len(set(baseline_map) & set(treatment_map)),
        "first_choice_disagreements": len(disagreements),
        "identical_trajectories": identical,
        "treatment_think_at_disagreement_rate": _safe_rate(
            sum(item["treatment_action"] == "think" for item in disagreements),
            len(disagreements),
        ),
        "baseline_think_at_disagreement_rate": _safe_rate(
            sum(item["baseline_action"] == "think" for item in disagreements),
            len(disagreements),
        ),
        "treatment_error_minus_baseline": (
            mean(float(item["treatment_tool_error"]) for item in disagreements)
            - mean(float(item["baseline_tool_error"]) for item in disagreements)
            if disagreements
            else 0.0
        ),
        "treatment_progress_minus_baseline": (
            mean(float(item["treatment_positive_progress"]) for item in disagreements)
            - mean(float(item["baseline_positive_progress"]) for item in disagreements)
            if disagreements
            else 0.0
        ),
        "examples": disagreements[:20],
    }
    if gold_tools is not None and disagreements:
        result["treatment_gold_tool_minus_baseline"] = mean(
            float(item["treatment_is_gold_tool"]) for item in disagreements
        ) - mean(float(item["baseline_is_gold_tool"]) for item in disagreements)
    return result


def diagnose(
    payload: dict[str, Any], tau_root: Path | None = None
) -> dict[str, Any]:
    evaluations = payload["evaluation_tasks"]
    split = payload["config"]["eval_split"]
    task_ids = [
        episode["task_id"]
        for episodes in evaluations.values()
        for episode in episodes
    ]
    gold_tools = (
        _gold_tools_by_task(tau_root, split, task_ids)
        if tau_root is not None
        else None
    )
    selectors = {
        name: analyze_selector(episodes, gold_tools)
        for name, episodes in evaluations.items()
    }
    comparisons = {}
    if "mask" in evaluations:
        for name, episodes in evaluations.items():
            if name != "mask":
                comparisons[f"{name}_vs_mask_shared_prefix"] = (
                    shared_prefix_disagreements(
                        evaluations["mask"], episodes, gold_tools
                    )
                )
    has_candidate_actions = any(
        "candidate_actions" in step
        for episodes in evaluations.values()
        for episode in episodes
        for step in episode["trajectory"]
    )
    return {
        "source_config": payload["config"],
        "candidate_actions_available": has_candidate_actions,
        "candidate_oracle_note": (
            "Exact oracle@K is computable from candidate_actions."
            if has_candidate_actions
            else "Exact oracle@K is unavailable because this run did not log candidate_actions."
        ),
        "selectors": selectors,
        "comparisons": comparisons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for name, item in report["selectors"].items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    _percent(item["success_rate"]),
                    _percent(item["zero_database_progress_rate"]),
                    _percent(item["tool_error_rate"]),
                    _percent(item["repeat_rate"]),
                    _percent(item["all_candidates_same_rate"]),
                    _percent(item["think_share"]),
                    _percent(item["mutation_share"]),
                    _percent(item["valid_order_id_format_rate"]),
                    _percent(item.get("mean_successful_gold_tool_coverage", 0.0)),
                ]
            )
            + " |"
        )
    lines = [
        "# Q4 actor trace diagnostics",
        "",
        report["candidate_oracle_note"],
        "",
        "Gold-tool coverage is only a task-level tool-name proxy; it does not validate next-action order or arguments.",
        "",
        "| Selector | Success | Zero DB progress | Tool error | Exact repeat | All 4 candidates same | Think share | Mutation share | Valid order ID | Successful gold-tool coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Most frequent actions",
        "",
    ]
    for name, item in report["selectors"].items():
        top = list(item["action_counts"].items())[:6]
        lines.append(
            f"- {name}: " + ", ".join(f"{action}={count}" for action, count in top)
        )
        if "gold_tool_candidate_step_rate" in item:
            lines.append(
                f"  Logged-candidate gold-tool proxy: present in "
                f"{_percent(item['gold_tool_candidate_step_rate'])} of steps; "
                f"selected in {_percent(item['selected_gold_tool_given_candidate_rate'])} "
                "of those opportunities."
            )
    lines.extend(["", "## Shared-prefix selector disagreements", ""])
    for name, item in report["comparisons"].items():
        lines.append(
            f"- {name}: {item['first_choice_disagreements']}/{item['paired_episodes']} first-choice disagreements; "
            f"treatment chose `think` in {_percent(item['treatment_think_at_disagreement_rate'])}."
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = diagnose(payload, args.tau_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
