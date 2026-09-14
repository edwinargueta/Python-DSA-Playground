"""Node-backed stack stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from node import Node


class LinkedStack:
    """A last-in-first-out stack over a chain of nodes.

    Every operation happens at the head of the chain, which is what makes
    push and pop O(1) with no array to resize and no capacity to manage.

    Structural invariants every mutating method must preserve:
      1. _top is the most recently pushed node, or None when the stack is
         empty.
      2. Following .next from _top reaches every value, bottom last, and
         terminates at None.
      3. _size equals the number of reachable nodes.
      4. A popped node is unlinked - its .next is cleared, so it no longer
         holds on to the rest of the stack.

    Values are unconstrained and duplicates are kept. None is a legitimate
    value, which is why is_empty() exists: peek() returning None does not
    mean the stack is empty.

    to_list() and iteration both run top first, in pop order, so this class
    and ArrayStack look the same from the outside.
    """

    def __init__(self) -> None:
        self._top: Optional[Node] = None
        self._size: int = 0

    # --- adding ----------------------------------------------------------

    def push(self, value: Any) -> None:
        """Add value to the top. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- removing --------------------------------------------------------

    def pop(self) -> Any:
        """Remove and return the top; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def clear(self) -> None:
        """Drop every node, returning to the empty state. O(1) time, O(1) space."""
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

    # --- validation ------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if every structural invariant holds. O(n) time, O(1) space."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[Any]:
        """Yield the values top first, walking the nodes directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as LinkedStack([3, 2, 1]), top first."""
        raise NotImplementedError


if __name__ == "__main__":
    stack = LinkedStack()
    for n in [10, 20, 30]:
        stack.push(n)
    stack.print_stack()
    print(stack)
    print("top:", stack.peek())
    print("popped:", stack.pop())
    print("size:", len(stack))
