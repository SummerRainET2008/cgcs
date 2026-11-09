from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


_MISSING = object()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _data_hash(data: dict[str, Any]) -> str:
    return sha256(str(_freeze(data)).encode("utf-8")).hexdigest()


def _flatten_leaves(value: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    """Map a nested JSON-like object to leaf paths for progress comparison."""
    if isinstance(value, dict):
        result: dict[tuple[Any, ...], Any] = {}
        for key, item in value.items():
            result.update(_flatten_leaves(item, path + (key,)))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten_leaves(item, path + (index,)))
        return result
    return {path: value}


def _load_tau(tau_root: Path, split: str) -> tuple[Any, Any, Any, str]:
    root = str(tau_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    # The upstream package imports its cloud-backed user simulator at package
    # import time. This deterministic adapter never calls it, so provide the
    # narrow symbol it expects instead of installing every cloud SDK.
    if "litellm" not in sys.modules:
        litellm = types.ModuleType("litellm")

        def unavailable_completion(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("litellm is unavailable in direct-user mode")

        litellm.completion = unavailable_completion  # type: ignore[attr-defined]
        sys.modules["litellm"] = litellm
    from tau_bench.envs.retail.data import load_data
    from tau_bench.envs.retail.tools import ALL_TOOLS
    from tau_bench.envs.retail.wiki import WIKI

    if split == "train":
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
    elif split == "dev":
        from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
    elif split == "test":
        from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
    else:
        raise ValueError(f"unknown Retail split: {split}")
    return load_data, ALL_TOOLS, tasks, WIKI


@dataclass(frozen=True)
class StepResult:
    observation: str
    done: bool
    reward: float
    error: bool


class DirectRetailEnv:
    """Deterministic τ-bench Retail adapter exposing the full task instruction.

    It preserves the executable tools and official terminal database/output
    reward, while removing the external LLM user simulator for algorithmic
    debugging.
    """

    def __init__(self, tau_root: Path, split: str, task_index: int):
        load_data, tool_types, tasks, wiki = _load_tau(tau_root, split)
        self._load_data = load_data
        self.task = tasks[task_index]
        self.task_index = task_index
        self.wiki = wiki
        self.tool_types = tool_types
        self.tools = {
            tool.get_info()["function"]["name"]: tool for tool in tool_types
        }
        self.tool_schemas = [tool.get_info() for tool in tool_types]
        self.reset()

    @property
    def instruction(self) -> str:
        return self.task.instruction

    @property
    def action_names(self) -> list[str]:
        # Retail policy explicitly forbids transfer; excluding it also avoids
        # treating an early hand-off as a successful direct-user trajectory.
        return [name for name in self.tools if name != "transfer_to_human_agents"] + [
            "respond"
        ]

    def reset(self) -> str:
        self.data = self._load_data()
        initial_data = deepcopy(self.data)
        gold_data = self._load_data()
        for action in self.task.actions:
            if action.name in self.tools and action.name != "transfer_to_human_agents":
                self.tools[action.name].invoke(data=gold_data, **action.kwargs)
        initial_flat = _flatten_leaves(initial_data)
        self._gold_flat = _flatten_leaves(gold_data)
        all_paths = set(initial_flat) | set(self._gold_flat)
        self._changed_paths = {
            path
            for path in all_paths
            if initial_flat.get(path, _MISSING) != self._gold_flat.get(path, _MISSING)
        }
        self._gold_hash = _data_hash(gold_data)
        self.actions: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[str] = []
        self.done = False
        return self.instruction

    def schema_for(self, name: str) -> dict[str, Any]:
        if name == "respond":
            return {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            }
        return self.tools[name].get_info()["function"]["parameters"]

    def step(self, name: str, arguments: dict[str, Any]) -> StepResult:
        if self.done:
            raise RuntimeError("cannot step a completed environment")
        self.actions.append((name, arguments))
        if name == "respond":
            content = str(arguments.get("content", ""))
            self.responses.append(content)
            self.done = True
            reward = self.calculate_reward()
            return StepResult(content, True, reward, False)
        try:
            observation = self.tools[name].invoke(data=self.data, **arguments)
            observation_text = str(observation)
            return StepResult(
                observation_text,
                False,
                0.0,
                observation_text.lstrip().lower().startswith("error"),
            )
        except Exception as exc:
            return StepResult(f"Error: {exc}", False, 0.0, True)

    def finish(self, content: str = "") -> StepResult:
        return self.step("respond", {"content": content})

    def calculate_reward(self) -> float:
        actual_hash = _data_hash(self.data)
        if actual_hash != self._gold_hash:
            return 0.0
        joined = "\n".join(self.responses).lower().replace(",", "")
        if any(str(output).lower() not in joined for output in self.task.outputs):
            return 0.0
        return 1.0

    def database_progress(self) -> float:
        """Fraction of task-relevant database leaves that match the gold state."""
        if not self._changed_paths:
            return 0.0
        current = _flatten_leaves(self.data)
        matches = sum(
            current.get(path, _MISSING) == self._gold_flat.get(path, _MISSING)
            for path in self._changed_paths
        )
        return float(matches / len(self._changed_paths))

    def gold_action_names(self) -> list[str]:
        names = [
            action.name
            for action in self.task.actions
            if action.name != "transfer_to_human_agents"
        ]
        names.append("respond")
        return names

    def describe(self) -> str:
        return json.dumps(
            {
                "task_index": self.task_index,
                "instruction": self.instruction,
                "gold_actions": self.gold_action_names(),
                "outputs": list(self.task.outputs),
            },
            ensure_ascii=False,
        )
