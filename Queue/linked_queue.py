"""Node-backed queue stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from node import Node


class LinkedQueue:
    """A first-in-first-out queue over a chain of nodes.

    Values leave at the head and arrive at the tail, which is the reason
    for the tail pointer: without it every enqueue would walk the chain to
    find the end, and an O(1) operation would become O(n).

    Structural invariants every mutating method must preserve:
      1. _head is the front node and _tail the back one; following .next
         from _head reaches every value, front first, ending at None.
      2. An empty queue has _head None, _tail None and _size 0.
      3. A one-value queue has _head and _tail as the same node.
      4. _size equals the number of reachable nodes.
      5. A dequeued node is unlinked - its .next is cleared, so it no
         longer holds on to the rest of the queue.

    Values are unconstrained and duplicates are kept. None is a legitimate
    value, which is why is_empty() exists: peek() returning None does not
    mean the queue is empty.
    """

    def __init__(self) -> None:
        self._head: Optional[Node] = None
        self._tail: Optional[Node] = None
        self._size: int = 0

    # --- adding ----------------------------------------------------------

    def enqueue(self, value: Any) -> None:
        """Add value to the back. O(1) time, O(1) space."""
        new_node = Node(value)
        if self._head is None:
            self._head = new_node
            self._tail = new_node
        else:
            self._tail.next = new_node
            self._tail = new_node
        self._size += 1

    # --- removing --------------------------------------------------------

    def dequeue(self) -> Any:
        """Remove and return the front; IndexError if empty. O(1) time, O(1) space."""
        if self._size == 0:
            raise IndexError("dequeue from empty queue")
        temp = self._head
        if self._size == 1:
            self._head = None
            self._tail = None
        else:
            self._head = self._head.next
            temp.next = None
        self._size -= 1
        return temp.value

    def clear(self) -> None:
        """Drop every node, returning to the empty state. O(1) time, O(1) space."""
        self._head = None
        self._tail = None
        self._size = 0

    # --- reading ---------------------------------------------------------

    def peek(self) -> Any:
        """Return the front; IndexError if empty. O(1) time, O(1) space."""
        if self._size == 0:
            raise IndexError("peek from empty queue")
        return self._head.value

    def peek_back(self) -> Any:
        """Return the back; IndexError if empty. O(1) time, O(1) space."""
        if self._size == 0:
            raise IndexError("peek_back from empty queue")
        return self._tail.value

    def is_empty(self) -> bool:
        """Return True if the queue holds nothing. O(1) time, O(1) space."""
        return self._size == 0

    def contains(self, value: Any) -> bool:
        """Return True if value is in the queue. O(n) time, O(1) space."""
        temp = self._head
        while temp is not None:
            if temp.value == value:
                return True
            temp = temp.next
        return False

    def to_list(self) -> List[Any]:
        """Return the values as a list, front first. O(n) time, O(n) space."""
        values = []
        temp = self._head
        while temp is not None:
            values.append(temp.value)
            temp = temp.next
        return values

    def print_queue(self) -> None:
        """Print the values from front to back. O(n) time, O(1) space."""
        temp = self._head
        while temp is not None:
            print(temp.value)
            temp = temp.next

    # --- validation ------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if every structural invariant holds. O(n) time, O(1) space."""
        if self._head is None or self._tail is None:
            return self._head is None and self._tail is None and self._size == 0
        count = 0
        last = None
        temp = self._head
        while temp is not None:
            count += 1
            if count > self._size:
                return False
            last = temp
            temp = temp.next
        return count == self._size and last is self._tail

    # Dunder Helpers
    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[Any]:
        """Yield the values front first, walking the nodes directly."""
        temp = self._head
        while temp is not None:
            yield temp.value
            temp = temp.next

    def __repr__(self) -> str:
        """Render as LinkedQueue([1, 2, 3]), front first."""
        return f"LinkedQueue({self.to_list()!r})"


if __name__ == "__main__":
    queue = LinkedQueue()
    for n in [10, 20, 30]:
        queue.enqueue(n)
    queue.print_queue()
    print(queue)
    print("front:", queue.peek(), "back:", queue.peek_back())
    print("served:", queue.dequeue())
    print("size:", len(queue))
