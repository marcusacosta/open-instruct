"""Utilities for the Sample Optimal Length (SOL) reward."""

import math
from numbers import Integral, Real


def _validate_sol_coefficients(alpha: float, beta: float) -> None:
    if not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    if not isinstance(beta, Real) or not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be a finite non-negative number")


def _validate_sol_inputs(correctness: list[bool], lengths: list[int], alpha: float, beta: float) -> None:
    if not correctness or len(correctness) != len(lengths):
        raise ValueError("correctness and lengths must be non-empty lists of equal length")
    if any(not isinstance(value, bool) for value in correctness):
        raise ValueError("correctness must contain booleans")
    if any(not isinstance(length, Integral) or length < 0 for length in lengths):
        raise ValueError("lengths must contain non-negative integers")
    _validate_sol_coefficients(alpha, beta)


def compute_sol_rewards(
    correctness: list[bool], lengths: list[int], alpha: float = 2.0, beta: float = 0.001
) -> list[float]:
    """Compute ShorterBetter's Sample Optimal Length reward for one prompt group.

    Lengths count generated response tokens, including tokens in truncated responses.
    If no response is correct, the target length is the group's mean response length.
    """
    _validate_sol_inputs(correctness, lengths, alpha, beta)

    correct_lengths = [length for length, is_correct in zip(lengths, correctness) if is_correct]
    target_length = min(correct_lengths) if correct_lengths else sum(lengths) / len(lengths)
    return [alpha * is_correct - beta * abs(length - target_length) for length, is_correct in zip(lengths, correctness)]
