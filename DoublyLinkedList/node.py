"""Node used by the DoublyLinkedList."""

from __future__ import annotations

from typing import Any, Optional


class Node:
    """A single node in a doubly linked list.

    Holds a value and two pointers. A node with next set to None is the last
    node in its list; a node with prev set to None is the first.
    """

    def __init__(self, value: Any) -> None:
        self.value: Any = value
        self.next: Optional["Node"] = None
        self.prev: Optional["Node"] = None

    # String Representation of the Node
    def __repr__(self) -> str:
        return f"Node({self.value!r})"
