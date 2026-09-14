"""Node used by the HashTable bucket chains."""

from __future__ import annotations

from typing import Any, Optional


class Node:
    """A single key/value entry in a hash table bucket chain.

    Holds the key alongside the value, because two different keys can land in
    the same bucket and a lookup has to tell them apart. A node with next set
    to None is the last entry in its chain.
    """

    def __init__(self, key: Any, value: Any) -> None:
        self.key: Any = key
        self.value: Any = value
        self.next: Optional["Node"] = None

    # String Representation of the Node
    def __repr__(self) -> str:
        return f"Node({self.key!r}: {self.value!r})"
