"""Minimum-tracking stack stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List


class MinStack:
    """A stack that also reports its smallest value in constant time.

    _values holds every value, bottom at index 0, so the top is the last
    element. _mins is the auxiliary stack that keeps min() O(1), and how
    you fill it is yours to decide - push every value onto it, or only the
    ones that set a new minimum. Its length is not asserted anywhere; only
    what sits on top of it is.

    Structural invariants every mutating method must preserve:
      1. _values holds the stack's values, bottom first.
      2. len(_values) is the size of the stack.
      3. The top of _mins is the smallest value in _values, which is what
         min() returns.
      4. _mins is empty exactly when _values is.

    Values must be comparable with <. Duplicates are kept, so two equal
    minima are two separate entries: popping one leaves the other as the
    minimum. None is a legitimate value only if nothing else is pushed
    alongside it, since it cannot be compared.

    to_list() and iteration both run top first, in pop order, matching the
    other two stacks.
    """

    def __init__(self) -> None:
        self._values: List[Any] = []
        self._mins: List[Any] = []

    # --- adding ----------------------------------------------------------

    def push(self, value: Any) -> None:
        """Add value to the top. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- removing --------------------------------------------------------

    def pop(self) -> Any:
        """Remove and return the top; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def clear(self) -> None:
        """Empty both stacks, returning to the start. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- reading ---------------------------------------------------------

    def peek(self) -> Any:
        """Return the top; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def min(self) -> Any:
        """Return the smallest value; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stack holds nothing. O(1) time, O(1) space."""
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        """Return True if value is in the stack. O(n) time, O(1) space."""
        raise NotImplementedError

    def to_list(self) -> List[Any]:
        """Return the values as a list, top first. O(n) time, O(n) space."""
        raise NotImplementedError

    def print_stack(self) -> None:
        """Print the values from top to bottom. O(n) time, O(1) space."""
        raise NotImplementedError

    # --- validation ------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if every structural invariant holds. O(n) time, O(1) space."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[Any]:
        """Yield the values top first, reading _values directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as MinStack([3, 2, 1]), top first."""
        raise NotImplementedError


if __name__ == "__main__":
    stack = MinStack()
    for n in [5, 3, 7, 3, 9]:
        stack.push(n)
        print(f"pushed {n}, min is {stack.min()}")
    print(stack)
    while not stack.is_empty():
        print(f"min is {stack.min()}, popping {stack.pop()}")
