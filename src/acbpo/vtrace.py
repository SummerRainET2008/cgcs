from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def discounted_return(rewards: FloatArray, gamma: float) -> float:
    powers = gamma ** np.arange(rewards.size)
    return float(np.dot(powers, rewards))


def per_decision_is_return(
    rewards: FloatArray, ratios: FloatArray, gamma: float
) -> float:
    cumulative_ratio = np.cumprod(ratios)
    powers = gamma ** np.arange(rewards.size)
    return float(np.dot(powers * cumulative_ratio, rewards))


def vtrace_values(
    *,
    states: IntArray,
    rewards: FloatArray,
    ratios: FloatArray,
    baseline_values: FloatArray,
    gamma: float,
    rho_bar: float = 10.0,
    c_bar: float = 1.0,
) -> FloatArray:
    """Compute finite-horizon V-trace targets for every visited state."""
    horizon = rewards.size
    output = np.empty(horizon, dtype=np.float64)
    next_vtrace = 0.0
    for t in range(horizon - 1, -1, -1):
        value = baseline_values[t, states[t]]
        next_value = (
            baseline_values[t + 1, states[t + 1]] if t + 1 < horizon else 0.0
        )
        rho = min(rho_bar, ratios[t])
        c = min(c_bar, ratios[t])
        delta = rho * (rewards[t] + gamma * next_value - value)
        output[t] = value + delta + gamma * c * (next_vtrace - next_value)
        next_vtrace = output[t]
    return output


def batch_vtrace_initial(
    *,
    states: IntArray,
    rewards: FloatArray,
    ratios: FloatArray,
    baseline_values: FloatArray,
    gamma: float,
    rho_bar: float = 10.0,
    c_bar: float = 1.0,
) -> FloatArray:
    """Vectorized initial-state V-trace targets for a trajectory batch."""
    if states.shape != rewards.shape or ratios.shape != rewards.shape:
        raise ValueError("states, rewards, and ratios must share shape [batch, horizon]")
    batch_size, horizon = rewards.shape
    next_vtrace = np.zeros(batch_size, dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        value = baseline_values[t, states[:, t]]
        next_value = (
            baseline_values[t + 1, states[:, t + 1]]
            if t + 1 < horizon
            else np.zeros(batch_size, dtype=np.float64)
        )
        rho = np.minimum(rho_bar, ratios[:, t])
        c = np.minimum(c_bar, ratios[:, t])
        delta = rho * (rewards[:, t] + gamma * next_value - value)
        current = value + delta + gamma * c * (next_vtrace - next_value)
        next_vtrace = current
    return next_vtrace
