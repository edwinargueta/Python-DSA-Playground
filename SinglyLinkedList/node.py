"""Node used by the SinglyLinkedList."""

from __future__ import annotations

from typing import Any, Optional


class Node:
    """A single node in a singly linked list.

    Holds a value and one forward pointer. A node with next set to None is
    the last node in its list.
    """

    def __init__(self, value: Any) -> None:
        self.value: Any = value
        self.next: Optional["Node"] = None

    # String Representation of the Node
    def __repr__(self) -> str:
        return f"Node({self.value!r})"
