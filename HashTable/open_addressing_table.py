"""Hash table stub, open addressing - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List, Tuple


class _Slot:
    """A marker for a slot that holds no live entry.

    Compared by identity, so no real key can ever be mistaken for one.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


EMPTY = _Slot("EMPTY")
DELETED = _Slot("DELETED")


class OpenAddressingTable:
    """A hash table that resolves collisions by probing for another slot.

    Entries live in two parallel lists rather than in chains: _keys[i] holds
    a key, EMPTY, or DELETED, and _values[i] holds that key's value. A slot
    is EMPTY if nothing has ever occupied it and DELETED once an entry has
    been removed from it - the tombstone that keeps probe runs intact.

    Structural invariants every mutating method must preserve:
      1. _keys and _values are the same length, and that length is the
         capacity - never zero.
      2. A key appears in at most one slot. put() on a key that is already
         present overwrites the value and leaves _size alone.
      3. Every key is reachable from its home slot by following the probe
         sequence without crossing an EMPTY slot.
      4. _size is the number of live keys and _tombstones the number of
         DELETED slots.
      5. _size + _tombstones is always below the capacity, so at least one
         EMPTY slot remains and no probe can run forever.
      6. occupancy() is at or below _max_load once put() returns, which is
         what forces a resize.

    Tombstones are why occupancy(), not load_factor(), drives that resize:
    a table whose live entries are few but whose slots are all DELETED has
    to be rebuilt, or every lookup walks the whole table. For the same
    reason put() fills the first tombstone on the key's probe run rather
    than taking a fresh EMPTY slot further along it.

    Keys may be any hashable value, and two keys that compare equal must
    hash equal. Values are unconstrained, so None is a real value: get()
    returning None does not mean the key is missing, and only contains()
    can tell the difference.

    probe_length() counts every slot examined, the one that ends the search
    included - the slot holding the key, or the EMPTY slot that proves it
    is absent.

    Iteration order is unspecified and changes when the table resizes. The
    table grows but never shrinks: remove() and clear() leave the capacity
    where it is.
    """

    def __init__(self, capacity: int = 8, max_load: float = 0.5) -> None:
        self._keys: List[Any] = [EMPTY] * capacity
        self._values: List[Any] = [None] * capacity
        self._size: int = 0
        self._tombstones: int = 0
        self._max_load: float = max_load

    # --- hashing and probing ---------------------------------------------

    def _hash(self, key: Any) -> int:
        """Return a non-negative int; equal keys agree. O(1) time, O(1) space."""
        raise NotImplementedError

    def _slot_index(self, key: Any) -> int:
        """Return the home slot for key, always in range. O(1) time, O(1) space."""
        raise NotImplementedError

    def _next_index(self, index: int) -> int:
        """Return the slot probed after index, wrapping. O(1) time, O(1) space."""
        raise NotImplementedError

    def _find_slot(self, key: Any) -> int:
        """Return the slot holding key, or -1 if absent. O(1) time, O(1) space."""
        raise NotImplementedError

    def _insert_slot(self, key: Any) -> int:
        """Return the slot put() should write key into. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- writing ---------------------------------------------------------

    def put(self, key: Any, value: Any) -> None:
        """Insert key, or overwrite it if present. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- reading ---------------------------------------------------------

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for key, else default. O(1) time, O(1) space."""
        raise NotImplementedError

    def contains(self, key: Any) -> bool:
        """Return True if key is stored. O(1) time, O(1) space."""
        raise NotImplementedError

    def keys(self) -> List[Any]:
        """Return every key, in any order. O(n) time, O(n) space."""
        raise NotImplementedError

    def values(self) -> List[Any]:
        """Return every value, in any order. O(n) time, O(n) space."""
        raise NotImplementedError

    def items(self) -> List[Tuple[Any, Any]]:
        """Return every key/value pair, in any order. O(n) time, O(n) space."""
        raise NotImplementedError

    def print_table(self) -> None:
        """Print every slot, tombstones included. O(n) time, O(1) space."""
        raise NotImplementedError

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
        """Return the number of slots. O(1) time, O(1) space."""
        raise NotImplementedError

    def load_factor(self) -> float:
        """Return live entries divided by slots. O(1) time, O(1) space."""
        raise NotImplementedError

    def occupancy(self) -> float:
        """Return live plus deleted, divided by slots. O(1) time, O(1) space."""
        raise NotImplementedError

    def tombstone_count(self) -> int:
        """Return how many slots are DELETED. O(1) time, O(1) space."""
        raise NotImplementedError

    def probe_length(self, key: Any) -> int:
        """Return how many slots a lookup of key reads. O(1) time, O(1) space."""
        raise NotImplementedError

    def _resize(self, new_capacity: int) -> None:
        """Rebuild into new_capacity slots, no tombstones. O(n) time, O(n) space."""
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
        """Yield every key, walking the slots directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as OpenAddressingTable({'a': 1})."""
        raise NotImplementedError


if __name__ == "__main__":
    table = OpenAddressingTable(capacity=8)
    for word in ["apple", "banana", "cherry", "date"]:
        table[word] = len(word)
    table.print_table()
    print(table)
    print("load:", table.load_factor(), "occupancy:", table.occupancy())
    print("probes for 'cherry':", table.probe_length("cherry"))
    del table["banana"]
    print("tombstones:", table.tombstone_count())
    print("probes for 'cherry':", table.probe_length("cherry"))
