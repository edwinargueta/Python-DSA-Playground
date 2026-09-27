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
        new_node = Node(value)
        new_node.next = self._top
        self._top = new_node
        self._size += 1

    # --- removing --------------------------------------------------------

    def pop(self) -> Any:
        """Remove and return the top; IndexError if empty. O(1) time, O(1) space."""
        if self._size == 0:
            raise IndexError("pop from empty stack")
        temp = self._top
        self._top = temp.next
        temp.next = None
        self._size -= 1
        return temp.value

    def clear(self) -> None:
        """Drop every node, returning to the empty state. O(1) time, O(1) space."""
        self._top = None
        self._size = 0

    # --- reading ---------------------------------------------------------

    def peek(self) -> Any:
        """Return the top; IndexError if empty. O(1) time, O(1) space."""
        if self._size == 0:
            raise IndexError("peek from empty stack")
        return self._top.value

    def is_empty(self) -> bool:
        """Return True if the stack holds nothing. O(1) time, O(1) space."""
        return self._size == 0

    def contains(self, value: Any) -> bool:
        """Return True if value is in the stack. O(n) time, O(1) space."""
        temp = self._top
        while temp is not None:
            if temp.value == value:
                return True
            temp = temp.next
        return False

    def to_list(self) -> List[Any]:
        """Return the values as a list, top first. O(n) time, O(n) space."""
        values = []
        temp = self._top
        while temp is not None:
            values.append(temp.value)
            temp = temp.next
        return values

    def print_stack(self) -> None:
        """Print the values from top to bottom. O(n) time, O(1) space."""
        temp = self._top
        while temp is not None:
            print(temp.value)
            temp = temp.next

    # --- validation ------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if every structural invariant holds. O(n) time, O(1) space."""
        count = 0
        temp = self._top
        while temp is not None:
            count += 1
            if count > self._size:
                return False
            temp = temp.next
        return count == self._size

    # Dunder Helpers
    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[Any]:
        """Yield the values top first, walking the nodes directly."""
        temp = self._top
        while temp is not None:
            yield temp.value
            temp = temp.next

    def __repr__(self) -> str:
        """Render as LinkedStack([3, 2, 1]), top first."""
        return f"LinkedStack({self.to_list()!r})"


if __name__ == "__main__":
    stack = LinkedStack()
    for n in [10, 20, 30]:
        stack.push(n)
    stack.print_stack()
    print(stack)
    print("top:", stack.peek())
    print("popped:", stack.pop())
    print("size:", len(stack))
