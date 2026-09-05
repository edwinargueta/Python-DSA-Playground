"""Binary search tree stub - implement the methods below.

Complexity notes use n for the number of nodes and h for the tree's height.
This tree does not self-balance, so h is O(log n) only when insertion order is
reasonably random. Sorted input degrades the tree into a linked list, making
h = n and every search-based operation O(n). That gap is the entire motivation
for AVL and red-black trees.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from node import Node


class BinarySearchTree:
    """An unbalanced binary search tree.

    Ordering invariant: for any node, every value in its left subtree is
    strictly less than the node's value, and every value in its right subtree
    is strictly greater. Duplicates are rejected rather than stored.
    """

    def __init__(self) -> None:
        self.root: Optional[Node] = None
        self._size: int = 0

    # --- core operations -------------------------------------------------

    def insert(self, value: Any) -> bool:
        """Insert value, keeping the BST ordering invariant.

        Returns:
            True if the value was inserted, False if it was already present.

        Remember to increment self._size on a successful insert.

        Time:  O(h) - O(log n) average, O(n) worst case.
        Space: O(1) iterative, O(h) recursive (call stack).
        """
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        """Return True if value is in the tree.

        Time:  O(h) - O(log n) average, O(n) worst case.
        Space: O(1) iterative, O(h) recursive.
        """
        raise NotImplementedError

    def delete(self, value: Any) -> bool:
        """Remove value from the tree.

        Three cases to handle:
          1. Leaf node - detach it from its parent.
          2. One child - splice the child into the node's position.
          3. Two children - replace the node's value with its in-order
             successor (smallest value in the right subtree), then delete
             that successor from the right subtree. Using the in-order
             predecessor is equally valid.

        The root is a special case in all three: it has no parent to update.

        Returns:
            True if the value was removed, False if it was not found.

        Remember to decrement self._size on a successful delete.

        Time:  O(h) - O(log n) average, O(n) worst case.
        Space: O(1) iterative, O(h) recursive.
        """
        raise NotImplementedError

    # --- queries ---------------------------------------------------------

    def find_min(self) -> Any:
        """Return the smallest value in the tree.

        The minimum is the leftmost node - follow left pointers to the end.

        Raises:
            ValueError: if the tree is empty.

        Time:  O(h) - O(log n) average, O(n) worst case.
        Space: O(1).
        """
        raise NotImplementedError

    def find_max(self) -> Any:
        """Return the largest value in the tree.

        The maximum is the rightmost node - follow right pointers to the end.

        Raises:
            ValueError: if the tree is empty.

        Time:  O(h) - O(log n) average, O(n) worst case.
        Space: O(1).
        """
        raise NotImplementedError

    def height(self) -> int:
        """Return the height of the tree measured in edges.

        An empty tree is -1, a single node is 0. Measuring in edges rather
        than nodes is what makes the empty case -1; both conventions exist,
        so state which one you mean.

        Time:  O(n) - every node must be visited.
        Space: O(h) for the recursion stack.
        """
        raise NotImplementedError

    def is_valid_bst(self) -> bool:
        """Return True if the tree satisfies the BST ordering property.

        An empty tree is vacuously valid.

        The common bug is comparing each node only against its immediate
        children. That misses violations further down: a node in the far left
        subtree can still exceed the root. Carry a (low, high) bound down the
        recursion, narrowing it at each step, and check every node against it.

        Time:  O(n).
        Space: O(h) for the recursion stack.
        """
        raise NotImplementedError

    # --- traversals ------------------------------------------------------

    def in_order(self) -> List[Any]:
        """Left, node, right - returns sorted order for a valid BST.

        Returns an empty list for an empty tree.

        Time:  O(n).
        Space: O(n) for the output, plus O(h) for the recursion stack.
        """
        raise NotImplementedError

    def pre_order(self) -> List[Any]:
        """Node, left, right.

        Useful for serializing a tree: re-inserting values in pre-order
        rebuilds the identical structure.

        Returns an empty list for an empty tree.

        Time:  O(n).
        Space: O(n) for the output, plus O(h) for the recursion stack.
        """
        raise NotImplementedError

    def post_order(self) -> List[Any]:
        """Left, right, node.

        Children are always visited before their parent, which is what you
        want when freeing or aggregating bottom-up.

        Returns an empty list for an empty tree.

        Time:  O(n).
        Space: O(n) for the output, plus O(h) for the recursion stack.
        """
        raise NotImplementedError

    def level_order(self) -> List[Any]:
        """Breadth-first, top to bottom, left to right.

        The only traversal here that is naturally iterative rather than
        recursive - use a queue rather than the call stack.

        Returns an empty list for an empty tree.

        Time:  O(n).
        Space: O(n) - the widest level can hold up to n/2 nodes.
        """
        raise NotImplementedError

    # --- dunder helpers --------------------------------------------------

    def __len__(self) -> int:
        """O(1) - reads the maintained size counter."""
        return self._size

    def __contains__(self, value: Any) -> bool:
        """Enables `value in tree`. Delegates to contains()."""
        return self.contains(value)

    def __iter__(self) -> Iterator[Any]:
        """Enables `for value in tree`, yielding sorted order."""
        return iter(self.in_order())


if __name__ == "__main__":
    bst = BinarySearchTree()
    for n in [50, 30, 70, 20, 40, 60, 80]:
        bst.insert(n)
    print(bst.in_order())
