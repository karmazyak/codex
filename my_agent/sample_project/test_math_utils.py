import pytest

from my_agent.sample_project.math_utils import fibonacci, squares


def test_fibonacci_basic():
    assert fibonacci(5) == [0, 1, 1, 2, 3]


def test_squares_basic():
    assert squares(5) == [0, 1, 4, 9, 16]


def test_negative_input():
    with pytest.raises(ValueError):
        fibonacci(-1)
    with pytest.raises(ValueError):
        squares(-1)
