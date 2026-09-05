"""Binary search tree stub - implement the methods below."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from node import Node


class BinarySearchTree:
    """An unbalanced binary search tree."""

    def __init__(self) -> None:
        self.root: Optional[Node] = None
        self._size: int = 0

    def insert(self, value: Any) -> bool:
        """Insert value. Return True if inserted, False if it already exists."""
        new_node = Node(value)
        if self.root is None:
            self.root = new_node
            self._size += 1
            return True

        curr = self.root
        while True:
            if value < curr.value:
                if curr.left is None:
                    curr.left = new_node
                    self._size += 1
                    return True
                curr = curr.left
            elif value > curr.value:
                if curr.right is None:
                    curr.right = new_node
                    self._size += 1
                    return True
                curr = curr.right
            else:
                return False

    def contains(self, value: Any) -> bool:
        """Return True if value is in the tree."""
        curr = self.root
        while curr is not None:
            if value < curr.value:
                curr = curr.left
            elif value > curr.value:
                curr = curr.right
            else:
                return True
        return False

    def delete(self, value: Any) -> bool:
        """Remove value. Return True if it was removed, False if not found."""
        raise NotImplementedError

    # Queries ------------------------------------------------------------

    def find_min(self) -> Any:
        """Return the smallest value in the tree."""
        raise NotImplementedError

    def find_max(self) -> Any:
        """Return the largest value in the tree."""
        raise NotImplementedError

    def height(self) -> int:
        """Return the height in edges. Empty tree is -1, single node is 0."""
        raise NotImplementedError

    def is_valid_bst(self) -> bool:
        """Return True if the tree satisfies the BST ordering property."""
        raise NotImplementedError

    # Traversals ----------------------------------------------------------

    def in_order(self) -> List[Any]:
        """Left, node, right - yields sorted order for a valid BST."""
        raise NotImplementedError

    def pre_order(self) -> List[Any]:
        """Node, left, right."""
        raise NotImplementedError

    def post_order(self) -> List[Any]:
        """Left, right, node."""
        raise NotImplementedError

    def level_order(self) -> List[Any]:
        """Breadth-first, top to bottom, left to right."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.in_order())


if __name__ == "__main__":
    bst = BinarySearchTree()
    for n in [50, 30, 70, 20, 40, 60, 80]:
        bst.insert(n)
    print(bst.in_order())
