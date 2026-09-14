"""Ring-buffer queue stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List


class ArrayQueue:
    """A first-in-first-out queue over a ring buffer you size yourself.

    _items is a fixed-size array whose length is the capacity. The queue
    wraps around it rather than shifting: _head is the index of the front
    value and _count is how many values follow it, so the value i places
    back sits at (_head + i) % capacity, and the next free slot is at
    (_head + _count) % capacity. list.pop(0) and insert(0, x) are out of
    bounds here - walking the indices instead is what keeps dequeue O(1)
    rather than O(n).

    Structural invariants every mutating method must preserve:
      1. len(_items) is the capacity, and is never zero.
      2. _head is always a valid index into _items, empty queue included.
      3. _count is how many values the queue holds, and never exceeds the
         capacity.
      4. The values are _items[(_head + i) % capacity] for every i below
         _count, front first.
      5. Every slot outside that window is None, so a dequeued value is not
         left behind in the array.

    Values are unconstrained and duplicates are kept. None is a legitimate
    value, which is why is_empty() exists: peek() returning None does not
    mean the queue is empty.

    The queue grows but never shrinks: dequeue() and clear() leave the
    capacity where it is.
    """

    def __init__(self, capacity: int = 8) -> None:
        self._items: List[Any] = [None] * capacity
        self._head: int = 0
        self._count: int = 0

    # --- adding ----------------------------------------------------------

    def enqueue(self, value: Any) -> None:
        """Add value to the back. O(1) amortized time, O(1) space."""
        raise NotImplementedError

    # --- removing --------------------------------------------------------

    def dequeue(self) -> Any:
        """Remove and return the front; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def clear(self) -> None:
        """Empty the queue, keeping the capacity. O(n) time, O(1) space."""
        raise NotImplementedError

    # --- reading ---------------------------------------------------------

    def peek(self) -> Any:
        """Return the front; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def peek_back(self) -> Any:
        """Return the back; IndexError if empty. O(1) time, O(1) space."""
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the queue holds nothing. O(1) time, O(1) space."""
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        """Return True if value is in the queue. O(n) time, O(1) space."""
        raise NotImplementedError

    def to_list(self) -> List[Any]:
        """Return the values as a list, front first. O(n) time, O(n) space."""
        raise NotImplementedError

    def print_queue(self) -> None:
        """Print the values from front to back. O(n) time, O(1) space."""
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
        """Yield the values front first, reading the array directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as ArrayQueue([1, 2, 3]), front first."""
        raise NotImplementedError


if __name__ == "__main__":
    queue = ArrayQueue(capacity=4)
    for n in [10, 20, 30]:
        queue.enqueue(n)
    print("served:", queue.dequeue(), queue.dequeue())
    for n in [40, 50, 60]:
        queue.enqueue(n)
    queue.print_queue()
    print(queue)
    print("capacity:", queue.capacity(), "head at:", queue._head)
    print("front:", queue.peek(), "back:", queue.peek_back())
