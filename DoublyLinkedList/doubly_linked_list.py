"""Doubly linked list stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from node import Node


class DoublyLinkedList:
    """A doubly linked list with head and tail pointers.

    Structural invariants every mutating method must preserve:
      1. head is the first node and head.prev is None; tail is the last node
         and tail.next is None.
      2. The links mirror each other - for every node, node.next.prev is node
         and node.prev.next is node.
      3. An empty list has head is None, tail is None, and _size == 0.
      4. A one-element list has head is tail, with both pointers None.
      5. _size always equals the number of reachable nodes.
      6. The list is acyclic - has_cycle() exists to detect a list that a
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
        new_Node = Node(value)
        if self.head is None:
            self.head = new_Node
            self.tail = new_Node
        else:
            self.tail.next = new_Node
            new_Node.prev = self.tail
            self.tail = new_Node
        self._size += 1
        return True


    def prepend(self, value: Any) -> None:
        """Add value to the front of the list. O(1) time, O(1) space."""
        new_Node = Node(value)
        if self._size == 0:
            self.head = new_Node
            self.tail = new_Node
        else:
            new_Node.next = self.head
            self.head.prev = new_Node
            self.head = new_Node
        self._size += 1
        return True

    def insert(self, index: int, value: Any) -> bool:
        """Insert value at index 0..len; False if invalid. O(n) time, O(1) space."""
        if index < 0 or index > self._size:
            return False
        if index == 0:
            return self.prepend(value)
        if index == self._size:
            return self.append(value)

        new_Node = Node(value)
        before = self.get(index - 1)
        after = before.next

        new_Node.prev = before
        new_Node.next = after
        before.next = new_Node
        after.prev = new_Node

        self._size += 1 
        return True

    def print_list(self) -> None:
        """Print the values from head to tail. O(n) time, O(1) space."""
        raise NotImplementedError

    def print_reverse(self) -> None:
        """Print the values from tail to head. O(n) time, O(1) space."""
        raise NotImplementedError

    # --- removing --------------------------------------------------------

    def pop_first(self) -> Any:
        """Remove and return the head; IndexError if empty. O(1) time, O(1) space."""
        temp = self.head
        if self._size == 0:
            return None
        elif self._size == 1:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
            temp.next = None
        self._size -= 1
        return temp

    def pop(self) -> Any:
        """Remove and return the tail; IndexError if empty. O(1) time, O(1) space."""
        if self._size == 0:
            return None
        temp = self.tail
        if self._size == 1:
            self.head = None
            self.tail = None
        else:
            self.tail = temp.prev
            self.tail.next = None
            temp.prev = None
        self._size -= 1
        return temp

    def remove(self, index: int) -> Any:
        """Remove the node at index and return its value. O(n) time, O(1) space."""
        if index < 0 or index >= self._size:
            return None
        if index == 0:
            return self.pop_first()
        if index == self._size - 1:
            return self.pop()

        temp = self.get(index)

        temp.next.prev = temp.prev
        temp.prev.next = temp.next
        temp.next = None
        temp.prev = None

        self.length -= 1
        return temp

    def remove_value(self, value: Any) -> bool:
        """Remove the first match; True if one was removed. O(n) time, O(1) space."""
        raise NotImplementedError

    def remove_node(self, node: Node) -> Any:
        """Unlink a node you hold and return its value. O(1) time, O(1) space."""
        raise NotImplementedError

    def clear(self) -> None:
        """Remove every node, returning to the empty state. O(1) time, O(1) space."""
        raise NotImplementedError

    # --- reading ---------------------------------------------------------

    def get(self, index: int) -> Any:
        """Return the value at index; IndexError if invalid. O(n) time, O(1) space."""
        if index < 0 or index >= self._size:
            return None
        temp = self.head
        if index < self._size/2:
            for _ in range(index):
                temp = temp.next
        else:
            temp = self.tail
            for _ in range(self._size - 1, index, -1):
                temp = temp.prev
        return temp

    def get_node(self, index: int) -> Node:
        """Return the node at index; IndexError if invalid. O(n) time, O(1) space."""
        raise NotImplementedError

    def set_value(self, index: int, value: Any) -> bool:
        """Overwrite the value at index; False if invalid. O(n) time, O(1) space."""
        temp = self.get(index)
        if temp:
            temp.value = value
            return True
        return False

    def index_of(self, value: Any) -> int:
        """Return the first index holding value, or -1. O(n) time, O(1) space."""
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        """Return True if value appears in the list. O(n) time, O(1) space."""
        raise NotImplementedError

    def to_list(self) -> List[Any]:
        """Return the values as a Python list, head first. O(n) time, O(n) space."""
        raise NotImplementedError

    def to_list_reversed(self) -> List[Any]:
        """Return the values as a Python list, tail first. O(n) time, O(n) space."""
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

    def is_palindrome(self) -> bool:
        """Return True if the values read the same reversed. O(n) time, O(1) space."""
        raise NotImplementedError

    def swap_first_last(self) -> None:
        """Swap the head and tail nodes, not their values. O(1) time, O(1) space."""
        raise NotImplementedError

    def has_cycle(self) -> bool:
        """Return True if following .next loops forever. O(n) time, O(1) space."""
        raise NotImplementedError

    def is_valid(self) -> bool:
        """Return True if every structural invariant holds. O(n) time, O(1) space."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __setitem__(self, index: int, value: Any) -> None:
        """Assign by index; IndexError if invalid. O(n) time, O(1) space."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        """Yield values head first, walking the nodes directly."""
        raise NotImplementedError

    def __reversed__(self) -> Iterator[Any]:
        """Yield values tail first, walking .prev directly."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as DoublyLinkedList([1, 2, 3])."""
        raise NotImplementedError


if __name__ == "__main__":
    dll = DoublyLinkedList()
    for n in [10, 20, 30, 40, 50]:
        dll.append(n)
    print(dll)
    dll.reverse()
    print(dll)
