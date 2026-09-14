"""Array-backed stack stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List


class ArrayStack:
    """A last-in-first-out stack over an array you size yourself.

    _items is a fixed-size array, not a growing Python list: its length is
    the capacity, and push() is the method responsible for replacing it
    with a larger one when it fills. The list operations that would do that
    work for you - append(), pop(), insert() and slice assignment - are out
    of bounds here, because the amortized O(1) push is the thing being
    practiced.

    Structural invariants every mutating method must preserve:
      1. len(_items) is the capacity, and is never zero.
      2. _count is how many values the stack holds, and never exceeds the
         capacity.
      3. The values sit in _items[0.._count - 1], bottom at index 0, so the
         top is _items[_count - 1].
      4. Every slot from _count onwards is None, so a popped value is not
         left behind in the array.

    Values are unconstrained and duplicates are kept. None is a legitimate
    value, which is why is_empty() exists: peek() returning None does not
    mean the stack is empty.

    to_list() and iteration both run top first, in pop order, so this class
    and LinkedStack look the same from the outside.

    The stack grows but never shrinks: pop() and clear() leave the capacity
    where it is.
    """

    def __init__(self, capacity: int = 8) -> None:
        self._items: List[Any] = [None] * capacity
        self._count: int = 0

    # --- adding ----------------------------------------------------------

    def push(self, value: Any) -> None:
        """Add value to the top. O(1) amortized time, O(1) space."""
        raise NotImplementedError

    # --- removing --------------------------------------------------------

    def pop(self) -> Any:
        """Remove and return the top; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def clear(self) -> None:
        """Empty the stack, keeping the capacity. O(n) time, O(1) space."""
        raise NotImplementedError

    # --- reading ---------------------------------------------------------

    def peek(self) -> Any:
        """Return the top; IndexError if empty. O(1) time, O(1) space."""
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

    # --- sizing ----------------------------------------------------------

    def capacity(self) -> int:
        """Return the length of the backing array. O(1) time, O(1) space."""
        raise NotImplementedError

    def _grow(self, new_capacity: int) -> None:
        """Rebuild into an array of new_capacity slots. O(n) time, O(n) space."""
        raise NotImplementedError

    # --- validation ------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if every structural invariant holds. O(n) time, O(1) space."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return self._count

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[Any]:
        """Yield the values top first, reading the array directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as ArrayStack([3, 2, 1]), top first."""
        raise NotImplementedError


if __name__ == "__main__":
    stack = ArrayStack(capacity=2)
    for n in [10, 20, 30, 40]:
        stack.push(n)
    stack.print_stack()
    print(stack)
    print("capacity:", stack.capacity(), "size:", len(stack))
    print("top:", stack.peek())
    print("popped:", stack.pop())
    print("30 in stack:", 30 in stack)
