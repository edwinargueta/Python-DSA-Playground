"""Singly linked list stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from node import Node


class SinglyLinkedList:
    """A singly linked list with head and tail pointers.

    Structural invariants every mutating method must preserve:
      1. head is the first node, tail is the last, and tail.next is None.
      2. An empty list has head is None, tail is None, and _size == 0.
      3. A one-element list has head is tail.
      4. _size always equals the number of reachable nodes.
      5. The list is acyclic - has_cycle() exists to detect a list that a
         caller has deliberately corrupted, not one this class produced.

    Unlike the BST, duplicate values are allowed and stored. Methods that act
    on a value operate on the first match from the head.

    Indices are zero-based, and negative indices are not supported.
    """

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size: int = 0

    # --- adding ----------------------------------------------------------

    def append(self, value: Any) -> None:
        """Add value to the end of the list. O(1) time, O(1) space."""
        raise NotImplementedError

    def prepend(self, value: Any) -> None:
        """Add value to the front of the list. O(1) time, O(1) space."""
        raise NotImplementedError

    def insert(self, index: int, value: Any) -> bool:
        """Insert value at index 0..len; False if invalid. O(n) time, O(1) space."""
        raise NotImplementedError

    # --- removing --------------------------------------------------------

    def pop_first(self) -> Any:
        """Remove and return the head; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def pop(self) -> Any:
        """Remove and return the tail; IndexError if empty. O(n) time, O(1) space."""
        raise NotImplementedError

    def remove(self, index: int) -> Any:
        """Remove the node at index and return its value. O(n) time, O(1) space."""
        raise NotImplementedError

    def remove_value(self, value: Any) -> bool:
        """Remove the first match; True if one was removed. O(n) time, O(1) space."""
        raise NotImplementedError

    def clear(self) -> None:
        """Remove every node, returning to the empty state. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- reading ---------------------------------------------------------

    def get(self, index: int) -> Any:
        """Return the value at index; IndexError if invalid. O(n) time, O(1) space."""
        raise NotImplementedError

    def set_value(self, index: int, value: Any) -> bool:
        """Overwrite the value at index; False if invalid. O(n) time, O(1) space."""
        raise NotImplementedError

    def index_of(self, value: Any) -> int:
        """Return the first index holding value, or -1. O(n) time, O(1) space."""
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        """Return True if value appears in the list. O(n) time, O(1) space."""
        raise NotImplementedError

    def to_list(self) -> List[Any]:
        """Return the values as a Python list, head first. O(n) time, O(n) space."""
        raise NotImplementedError

    # --- classic interview operations ------------------------------------

    def reverse(self) -> None:
        """Reverse in place, swapping head and tail. O(n) time, O(1) space."""
        raise NotImplementedError

    def find_middle(self) -> Any:
        """Return the middle value, later one when even. O(n) time, O(1) space."""
        raise NotImplementedError

    def nth_from_end(self, n: int) -> Any:
        """Return the nth value from the end, 1 = tail. O(n) time, O(1) space."""
        raise NotImplementedError

    def has_cycle(self) -> bool:
        """Return True if following .next loops forever. O(n) time, O(1) space."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __iter__(self) -> Iterator[Any]:
        """Yield values head first, walking the nodes directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as SinglyLinkedList([1, 2, 3])."""
        raise NotImplementedError


if __name__ == "__main__":
    sll = SinglyLinkedList()
    for n in [10, 20, 30, 40, 50]:
        sll.append(n)
    print(sll)
    sll.reverse()
    print(sll)
