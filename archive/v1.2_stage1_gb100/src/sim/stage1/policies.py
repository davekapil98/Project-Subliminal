"""Deterministic, joint-safe exploration policies for synthetic data collection."""

from __future__ import annotations

import numpy as np


class SafeJointPolicy:
    """Stateful policy that emits absolute SO-101 position targets."""

    def __init__(
        self,
        name: str,
        joint_limits: np.ndarray,
        default_position: np.ndarray,
        num_envs: int,
        steps: int,
        control_hz: int,
        seed: int,
    ) -> None:
        self.name = name
        self.num_envs = num_envs
        self.steps = steps
        self.control_hz = control_hz
        self.rng = np.random.default_rng(seed)
        limits = np.asarray(joint_limits, dtype=np.float32)
        if limits.shape != (6, 2):
            raise ValueError(f"joint_limits must be [6,2], got {limits.shape}")
        default = np.asarray(default_position, dtype=np.float32).reshape(6)
        limits = limits.copy()
        for index in range(6):
            if not np.isfinite(limits[index]).all() or limits[index, 1] <= limits[index, 0]:
                limits[index] = (default[index] - np.pi, default[index] + np.pi)
        span = limits[:, 1] - limits[:, 0]
        self.low = limits[:, 0] + 0.08 * span
        self.high = limits[:, 1] - 0.08 * span
        self.default = np.clip(default, self.low, self.high)
        self.command = np.repeat(self.default[None], num_envs, axis=0)
        self.goal = self.command.copy()
        self.last_delta = np.zeros_like(self.command)
        self.phase = self.rng.uniform(0.0, 2.0 * np.pi, size=(num_envs, 6)).astype(np.float32)
        self.frequency = self.rng.uniform(0.05, 0.22, size=(num_envs, 6)).astype(np.float32)
        self.amplitude = self.rng.uniform(0.08, 0.28, size=(num_envs, 6)).astype(np.float32) * span
        self.waypoint_noise = self.rng.uniform(-0.12, 0.12, size=(num_envs, 6)).astype(np.float32) * span

        # Conservative software bounds; physics still applies its own articulation limits.
        max_velocity = np.array([1.2, 1.0, 1.1, 1.5, 1.8, 1.2], dtype=np.float32)
        max_acceleration = np.array([4.0, 3.5, 4.0, 5.0, 6.0, 4.0], dtype=np.float32)
        self.max_delta = max_velocity / float(control_hz)
        self.max_delta_change = max_acceleration / float(control_hz**2)

    def _random_goal(self) -> np.ndarray:
        return self.rng.uniform(self.low, self.high, size=(self.num_envs, 6)).astype(np.float32)

    def _task_goal(self, step: int) -> np.ndarray:
        fraction = step / max(self.steps - 1, 1)
        target = np.repeat(self.default[None], self.num_envs, axis=0)
        if fraction < 0.2:  # open and approach
            target += self.waypoint_noise * np.array([0.4, 0.7, 0.7, 0.4, 0.25, 0.0], dtype=np.float32)
            target[:, 5] = self.high[5]
        elif fraction < 0.42:  # descend
            target += self.waypoint_noise
            target[:, 1] -= 0.12
            target[:, 2] += 0.12
            target[:, 5] = self.high[5]
        elif fraction < 0.58:  # grasp
            target += self.waypoint_noise
            target[:, 5] = self.low[5]
        elif fraction < 0.82:  # lift and move laterally toward rack
            target += self.waypoint_noise * 0.6
            target[:, 0] += 0.22
            target[:, 1] += 0.12
            target[:, 2] -= 0.1
            target[:, 5] = self.low[5]
        else:  # release
            target[:, 0] += 0.22
            target[:, 5] = self.high[5]
        return np.clip(target, self.low, self.high)

    def _recovery_goal(self, step: int) -> np.ndarray:
        fraction = step / max(self.steps - 1, 1)
        if fraction < 0.3:
            target = self.default[None] + 1.25 * self.waypoint_noise
            target[:, 5] = self.rng.choice([self.low[5], self.high[5]], size=self.num_envs)
            return np.clip(target, self.low, self.high)
        if fraction < 0.55:
            target = np.repeat(self.default[None], self.num_envs, axis=0)
            target[:, 5] = self.high[5]
            return target
        return np.repeat(self.default[None], self.num_envs, axis=0)

    def step(self, observation_q: np.ndarray, step: int) -> np.ndarray:
        """Return a velocity/acceleration-limited absolute target."""

        if step == 0:
            self.command = np.clip(np.asarray(observation_q, dtype=np.float32), self.low, self.high)
        if self.name == "smooth_random":
            if step % max(8, self.control_hz // 2) == 0:
                self.goal = self._random_goal()
        elif self.name == "motor_sweep":
            time_s = step / float(self.control_hz)
            midpoint = (self.low + self.high) * 0.5
            self.goal = midpoint + self.amplitude * np.sin(self.phase + 2.0 * np.pi * self.frequency * time_s)
        elif self.name == "task_attempt":
            self.goal = self._task_goal(step)
        elif self.name == "failure_recovery":
            self.goal = self._recovery_goal(step)
        else:
            raise ValueError(f"unknown policy: {self.name}")

        desired_delta = np.clip(self.goal - self.command, -self.max_delta, self.max_delta)
        lower_delta = self.last_delta - self.max_delta_change
        upper_delta = self.last_delta + self.max_delta_change
        delta = np.clip(desired_delta, lower_delta, upper_delta)
        self.command = np.clip(self.command + delta, self.low, self.high)
        self.last_delta = delta
        return self.command.astype(np.float32, copy=True)


class RandomizedActuation:
    """Per-environment command delay, quantization and deadband model."""

    def __init__(
        self,
        initial_command: np.ndarray,
        delay_range: tuple[int, int],
        deadband_rad: float,
        seed: int,
    ) -> None:
        initial = np.asarray(initial_command, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.delays = self.rng.integers(delay_range[0], delay_range[1] + 1, size=initial.shape[0])
        self.deadband = float(deadband_rad)
        self.applied = initial.copy()
        self.history = [initial.copy() for _ in range(int(self.delays.max(initial=0)) + 1)]
        self.quantization_rad = self.rng.uniform(0.0005, 0.002, size=(initial.shape[0], 1)).astype(np.float32)

    def apply(self, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(requested, dtype=np.float32)
        self.history.append(requested.copy())
        delayed = np.empty_like(requested)
        for env_index, delay in enumerate(self.delays):
            delayed[env_index] = self.history[-1 - int(delay)][env_index]
        quantized = np.round(delayed / self.quantization_rad) * self.quantization_rad
        held = np.abs(quantized - self.applied) < self.deadband
        self.applied = np.where(held, self.applied, quantized).astype(np.float32)
        max_history = int(self.delays.max(initial=0)) + 1
        if len(self.history) > max_history:
            self.history.pop(0)
        return self.applied.copy(), self.delays.astype(np.int16, copy=True)
