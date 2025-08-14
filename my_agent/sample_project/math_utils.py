"""Simple math utilities: Fibonacci and squares.

Run this module as a script to print the first N Fibonacci numbers and squares.
"""
from __future__ import annotations

import argparse
from typing import List


def fibonacci(n: int) -> List[int]:
    """Return a list with the first ``n`` Fibonacci numbers."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seq = []
    a, b = 0, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


def squares(n: int) -> List[int]:
    """Return a list of squares for numbers 0..n-1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return [i * i for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Fibonacci numbers and squares")
    parser.add_argument("n", type=int, help="how many numbers to generate")
    args = parser.parse_args()

    print("Fibonacci:", fibonacci(args.n))
    print("Squares:", squares(args.n))


if __name__ == "__main__":
    main()
