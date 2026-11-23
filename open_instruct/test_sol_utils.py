import pytest

from open_instruct.sol_utils import compute_sol_rewards


@pytest.mark.parametrize(
    ("correctness", "lengths", "expected"),
    [
        ([True, True], [10, 10], [2.0, 2.0]),
        ([True, True], [10, 20], [2.0, 1.99]),
        ([True, False], [10, 20], [2.0, -0.01]),
        ([False, False], [10, 20], [-0.005, -0.005]),
        ([False, False], [10, 10], [0.0, 0.0]),
        ([True, False], [0, 0], [2.0, 0.0]),
        ([False, False], [0, 0], [0.0, 0.0]),
        ([True, False], [5, 100_000], [2.0, -99.995]),
    ],
)
def test_sol_reward_groups(correctness, lengths, expected):
    assert compute_sol_rewards(correctness, lengths) == expected


def test_numerical_equivalence_with_direct_sol_equation():
    cases = [
        ([True, True, False], [4, 9, 17], 2.0, 0.001),
        ([False, False], [5, 20], 1.5, 0.02),
        ([True, False], [0, 100_000], 3.0, 0.0001),
    ]
    for correctness, lengths, alpha, beta in cases:
        correct_lengths = [length for length, correct in zip(lengths, correctness) if correct]
        target = min(correct_lengths) if correct_lengths else sum(lengths) / len(lengths)
        expected = [alpha * correct - beta * abs(length - target) for correct, length in zip(correctness, lengths)]
        assert compute_sol_rewards(correctness, lengths, alpha, beta) == expected


def test_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="correctness and lengths must be non-empty"):
        compute_sol_rewards([], [])
    with pytest.raises(ValueError, match="correctness and lengths must be non-empty"):
        compute_sol_rewards([True], [1, 2])


def test_rejects_invalid_correctness_values():
    with pytest.raises(ValueError, match="correctness must contain booleans"):
        compute_sol_rewards([1], [1])


@pytest.mark.parametrize("length", [-1, 1.5])
def test_rejects_invalid_lengths(length):
    with pytest.raises(ValueError, match="lengths must contain non-negative integers"):
        compute_sol_rewards([False], [length])


@pytest.mark.parametrize("alpha", [-1.0, float("inf"), float("nan"), "2"])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha must be a finite non-negative number"):
        compute_sol_rewards([True], [1], alpha=alpha)


@pytest.mark.parametrize("beta", [-1.0, float("inf"), float("nan"), "0.1"])
def test_rejects_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta must be a finite non-negative number"):
        compute_sol_rewards([True], [1], beta=beta)
