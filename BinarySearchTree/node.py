"""Node used by the BinarySearchTree."""

from __future__ import annotations

from typing import Any, Optional


class Node:
    """A single node in a binary search tree."""

    def __init__(self, value: Any) -> None:
        self.value: Any = value
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None

    # String Representation of the Node
    def __repr__(self) -> str:
        return f"Node({self.value!r})"
