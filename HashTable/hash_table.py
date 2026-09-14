"""Hash table stub, separate chaining - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple

from node import Node


class HashTable:
    """A hash table that resolves collisions with a linked chain per bucket.

    Structural invariants every mutating method must preserve:
      1. _buckets is a list of chain heads, one slot per bucket, and its
         length is the capacity - never zero.
      2. Every entry sits in the bucket _bucket_index() picks for its key,
         reachable by following .next from that bucket's head.
      3. A key appears at most once in the whole table. put() on a key that
         is already present overwrites the value and leaves _size alone.
      4. _size equals the number of entries reachable across every chain.
      5. Chains terminate at None - nothing here creates a cycle.
      6. load_factor() is at or below _max_load once put() returns, which is
         what forces a resize.

    Keys may be any hashable value, and two keys that compare equal must
    hash equal. Values are unconstrained, so None is a real value: get()
    returning None does not mean the key is missing, and only contains()
    can tell the difference.

    Where a new entry goes in its chain is yours to choose - nothing here
    depends on chain order. Iteration order is unspecified for the same
    reason, and changes when the table resizes.

    The table grows but never shrinks: remove() and clear() leave the
    capacity where it is.
    """

    def __init__(self, capacity: int = 8, max_load: float = 0.75) -> None:
        self._buckets: List[Optional[Node]] = [None] * capacity
        self._size: int = 0
        self._max_load: float = max_load

    # --- hashing ---------------------------------------------------------

    def _hash(self, key: Any) -> int:
        """Return a non-negative int; equal keys agree. O(1) time, O(1) space."""
        my_hash = 0
        for letter in key:
            my_hash = (my_hash + ord(letter) * 23) % len(self._buckets) 
        return my_hash

    def _bucket_index(self, key: Any) -> int:
        """Return the bucket for key, always in range. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- writing ---------------------------------------------------------

    def put(self, key: Any, value: Any) -> None:
        """Insert key, or overwrite it if present. O(1) time, O(1) space."""
        index = self._hash(key)
        if self._buckets[index] == None:
            self._buckets[index] = []
        self._buckets[index].append([key, value])

    # --- reading ---------------------------------------------------------

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for key, else default. O(1) time, O(1) space."""
        index = self._hash(key)
        if self._buckets[index] is not None:
            for i in range(len(self._buckets[index])):
                if self._buckets[index][i][0] == key:
                    return self._buckets[index][i][1]
        return default
    

    def contains(self, key: Any) -> bool:
        """Return True if key is stored. O(1) time, O(1) space."""
        raise NotImplementedError

    def keys(self) -> List[Any]:
        """Return every key, in any order. O(n) time, O(n) space."""
        all_keys = []
        for i in range(len(self._buckets)):
            if self._buckets[i] is not None:
                for j in range(len(self._buckets[i])):
                    all_keys.append(self._buckets[i][j][0])
        return all_keys

    def values(self) -> List[Any]:
        """Return every value, in any order. O(n) time, O(n) space."""
        raise NotImplementedError

    def items(self) -> List[Tuple[Any, Any]]:
        """Return every key/value pair, in any order. O(n) time, O(n) space."""
        raise NotImplementedError

    def print_table(self) -> None:
        """Print each bucket and the chain hanging off it. O(n) time, O(1) space."""
        for i, val in enumerate(self._buckets):
            print(i, ": ", val)

    # --- removing --------------------------------------------------------

    def remove(self, key: Any) -> bool:
        """Remove key; True if it was there. O(1) time, O(1) space."""
        raise NotImplementedError

    def pop(self, key: Any) -> Any:
        """Remove key, return its value; KeyError if absent. O(1) time, O(1) space."""
        raise NotImplementedError

    def clear(self) -> None:
        """Drop every entry, keeping the capacity. O(n) time, O(1) space."""
        raise NotImplementedError

    # --- sizing ----------------------------------------------------------

    def capacity(self) -> int:
        """Return the number of buckets. O(1) time, O(1) space."""
        raise NotImplementedError

    def load_factor(self) -> float:
        """Return entries divided by buckets. O(1) time, O(1) space."""
        raise NotImplementedError

    def bucket_sizes(self) -> List[int]:
        """Return each bucket's chain length, in order. O(n) time, O(n) space."""
        raise NotImplementedError

    def _resize(self, new_capacity: int) -> None:
        """Rehome every entry into new_capacity buckets. O(n) time, O(n) space."""
        raise NotImplementedError

    # --- validation ------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if every structural invariant holds. O(n) time, O(n) space."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self.contains(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    def __getitem__(self, key: Any) -> Any:
        """Return the value for key; KeyError if absent. O(1) time, O(1) space."""
        raise NotImplementedError

    def __delitem__(self, key: Any) -> None:
        """Remove key; KeyError if absent. O(1) time, O(1) space."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        """Yield every key, walking the buckets directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as HashTable({'a': 1})."""
        raise NotImplementedError


if __name__ == "__main__":
    table = HashTable(capacity=4)
    for word in ["apple", "banana", "cherry", "date", "elderberry"]:
        table[word] = len(word)
    table.print_table()
    print(table)
    print("capacity:", table.capacity(), "load:", table.load_factor())
    print("chains:", table.bucket_sizes())
    print("cherry ->", table["cherry"])
    del table["cherry"]
    print("cherry in table:", "cherry" in table)
