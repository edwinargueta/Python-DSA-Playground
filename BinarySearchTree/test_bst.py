"""Behavior tests for BinarySearchTree.

Run from inside this directory:

    python3 -m unittest test_bst -v
    python3 test_bst.py            # equivalent

Each test class depends only on the method it names, so you can implement the
methods in any order and a green class always means that method is done.
Tests for insert and delete check structure with the walk_in_order and is_bst
helpers below instead of calling the traversal methods.

The one deliberate exception is TestDunders: __iter__ delegates to in_order by
design, so two of its tests stay red until in_order is written.

Assertions reach .left and .right only through those helpers, and never
require a particular implementation strategy - recursive and iterative
solutions both satisfy them.
"""

from __future__ import annotations

import unittest

from binary_search_tree import BinarySearchTree
from node import Node


def build(values) -> BinarySearchTree:
    """Return a tree with values inserted in the given order."""
    bst = BinarySearchTree()
    for v in values:
        bst.insert(v)
    return bst


def walk_in_order(node) -> list:
    """Collect values left to right by walking the nodes directly.

    Lets the insert and delete tests verify tree shape without depending on
    in_order() being written yet.
    """
    if node is None:
        return []
    return walk_in_order(node.left) + [node.value] + walk_in_order(node.right)


def is_bst(node, low=None, high=None) -> bool:
    """Check the BST ordering property by walking the nodes directly."""
    if node is None:
        return True
    if low is not None and node.value <= low:
        return False
    if high is not None and node.value >= high:
        return False
    return is_bst(node.left, low, node.value) and is_bst(node.right, node.value, high)


# A balanced tree used throughout:
#
#            50
#          /    \
#        30      70
#       /  \    /  \
#     20    40 60   80
BALANCED = [50, 30, 70, 20, 40, 60, 80]


class TestEmptyTree(unittest.TestCase):
    """An empty tree is the edge case most implementations get wrong."""

    def setUp(self) -> None:
        self.bst = BinarySearchTree()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.bst), 0)

    def test_root_is_none(self) -> None:
        self.assertIsNone(self.bst.root)

    def test_contains_returns_false(self) -> None:
        self.assertFalse(self.bst.contains(1))

    def test_delete_returns_false(self) -> None:
        self.assertFalse(self.bst.delete(1))

    def test_height_is_negative_one(self) -> None:
        self.assertEqual(self.bst.height(), -1)

    def test_is_vacuously_valid(self) -> None:
        self.assertTrue(self.bst.is_valid_bst())

    def test_traversals_are_empty(self) -> None:
        self.assertEqual(self.bst.in_order(), [])
        self.assertEqual(self.bst.pre_order(), [])
        self.assertEqual(self.bst.post_order(), [])
        self.assertEqual(self.bst.level_order(), [])

    def test_find_min_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.bst.find_min()

    def test_find_max_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.bst.find_max()


class TestInsert(unittest.TestCase):

    def test_returns_true_on_new_value(self) -> None:
        self.assertTrue(BinarySearchTree().insert(10))

    def test_sets_root_on_first_insert(self) -> None:
        bst = build([10])
        self.assertIsInstance(bst.root, Node)
        self.assertEqual(bst.root.value, 10)

    def test_rejects_duplicates(self) -> None:
        bst = build([10])
        self.assertFalse(bst.insert(10))
        self.assertEqual(len(bst), 1)

    def test_size_tracks_successful_inserts(self) -> None:
        bst = build(BALANCED)
        self.assertEqual(len(bst), len(BALANCED))

    def test_duplicates_do_not_grow_the_tree(self) -> None:
        bst = build([50, 30, 70, 50, 30, 70])
        self.assertEqual(len(bst), 3)
        self.assertEqual(walk_in_order(bst.root), [30, 50, 70])

    def test_maintains_ordering(self) -> None:
        bst = build([50, 30, 70, 20, 40])
        self.assertEqual(walk_in_order(bst.root), [20, 30, 40, 50, 70])


class TestContains(unittest.TestCase):

    def setUp(self) -> None:
        self.bst = build(BALANCED)

    def test_finds_root(self) -> None:
        self.assertTrue(self.bst.contains(50))

    def test_finds_leaf(self) -> None:
        self.assertTrue(self.bst.contains(20))

    def test_finds_every_inserted_value(self) -> None:
        for v in BALANCED:
            self.assertTrue(self.bst.contains(v), f"missing {v}")

    def test_rejects_absent_value(self) -> None:
        self.assertFalse(self.bst.contains(99))

    def test_rejects_value_below_min(self) -> None:
        self.assertFalse(self.bst.contains(0))


class TestDelete(unittest.TestCase):
    """The three delete cases, each also tested at the root."""

    def test_missing_value_returns_false(self) -> None:
        bst = build(BALANCED)
        self.assertFalse(bst.delete(99))
        self.assertEqual(len(bst), len(BALANCED))

    def test_leaf(self) -> None:
        bst = build(BALANCED)
        self.assertTrue(bst.delete(20))
        self.assertEqual(walk_in_order(bst.root), [30, 40, 50, 60, 70, 80])
        self.assertEqual(len(bst), 6)

    def test_node_with_only_left_child(self) -> None:
        bst = build([50, 30, 20])
        self.assertTrue(bst.delete(30))
        self.assertEqual(walk_in_order(bst.root), [20, 50])
        self.assertTrue(is_bst(bst.root))

    def test_node_with_only_right_child(self) -> None:
        bst = build([50, 30, 40])
        self.assertTrue(bst.delete(30))
        self.assertEqual(walk_in_order(bst.root), [40, 50])
        self.assertTrue(is_bst(bst.root))

    def test_node_with_two_children(self) -> None:
        bst = build(BALANCED)
        self.assertTrue(bst.delete(30))
        self.assertEqual(walk_in_order(bst.root), [20, 40, 50, 60, 70, 80])
        self.assertTrue(is_bst(bst.root))

    def test_root_when_only_node(self) -> None:
        bst = build([50])
        self.assertTrue(bst.delete(50))
        self.assertIsNone(bst.root)
        self.assertEqual(len(bst), 0)
        self.assertEqual(walk_in_order(bst.root), [])

    def test_root_with_one_child(self) -> None:
        bst = build([50, 30])
        self.assertTrue(bst.delete(50))
        self.assertEqual(walk_in_order(bst.root), [30])
        self.assertEqual(bst.root.value, 30)

    def test_root_with_two_children(self) -> None:
        bst = build(BALANCED)
        self.assertTrue(bst.delete(50))
        self.assertEqual(walk_in_order(bst.root), [20, 30, 40, 60, 70, 80])
        self.assertTrue(is_bst(bst.root))
        self.assertNotEqual(bst.root.value, 50)

    def test_deleting_everything_empties_the_tree(self) -> None:
        bst = build(BALANCED)
        for v in BALANCED:
            self.assertTrue(bst.delete(v), f"failed to delete {v}")
        self.assertEqual(len(bst), 0)
        self.assertIsNone(bst.root)

    def test_tree_stays_valid_after_each_delete(self) -> None:
        for target in BALANCED:
            bst = build(BALANCED)
            bst.delete(target)
            self.assertTrue(is_bst(bst.root), f"invalid after deleting {target}")
            expected = sorted(v for v in BALANCED if v != target)
            self.assertEqual(walk_in_order(bst.root), expected)

    def test_reinsert_after_delete(self) -> None:
        bst = build(BALANCED)
        bst.delete(30)
        self.assertTrue(bst.insert(30))
        self.assertEqual(walk_in_order(bst.root), sorted(BALANCED))
        self.assertEqual(len(bst), len(BALANCED))


class TestMinMax(unittest.TestCase):

    def test_min_and_max(self) -> None:
        bst = build(BALANCED)
        self.assertEqual(bst.find_min(), 20)
        self.assertEqual(bst.find_max(), 80)

    def test_single_node_is_both(self) -> None:
        bst = build([42])
        self.assertEqual(bst.find_min(), 42)
        self.assertEqual(bst.find_max(), 42)

    def test_after_deleting_the_extremes(self) -> None:
        bst = build(BALANCED)
        bst.delete(20)
        bst.delete(80)
        self.assertEqual(bst.find_min(), 30)
        self.assertEqual(bst.find_max(), 70)


class TestHeight(unittest.TestCase):

    def test_empty_is_negative_one(self) -> None:
        self.assertEqual(BinarySearchTree().height(), -1)

    def test_single_node_is_zero(self) -> None:
        self.assertEqual(build([50]).height(), 0)

    def test_balanced_tree(self) -> None:
        self.assertEqual(build(BALANCED).height(), 2)

    def test_degenerate_tree_is_linked_list(self) -> None:
        # Sorted input produces a right spine - the BST worst case.
        bst = build([10, 20, 30, 40, 50])
        self.assertEqual(bst.height(), 4)

    def test_uses_the_deeper_subtree(self) -> None:
        bst = build([50, 30, 70, 20, 10])
        self.assertEqual(bst.height(), 3)


class TestIsValidBst(unittest.TestCase):

    def test_valid_tree(self) -> None:
        self.assertTrue(build(BALANCED).is_valid_bst())

    def test_single_node(self) -> None:
        self.assertTrue(build([50]).is_valid_bst())

    def test_detects_shallow_violation(self) -> None:
        bst = build([50, 30])
        bst.root.left.value = 60  # 60 belongs to the right of 50
        self.assertFalse(bst.is_valid_bst())

    def test_detects_deep_violation(self) -> None:
        # 55 is correctly placed relative to its parent 30, but violates the
        # root's bound. Naive parent-only checks pass this and are wrong.
        bst = build([50, 30, 70, 20, 40])
        bst.root.left.right.value = 55
        self.assertFalse(bst.is_valid_bst())


class TestTraversals(unittest.TestCase):

    def setUp(self) -> None:
        self.bst = build(BALANCED)

    def test_in_order_is_sorted(self) -> None:
        self.assertEqual(self.bst.in_order(), [20, 30, 40, 50, 60, 70, 80])

    def test_pre_order(self) -> None:
        self.assertEqual(self.bst.pre_order(), [50, 30, 20, 40, 70, 60, 80])

    def test_post_order(self) -> None:
        self.assertEqual(self.bst.post_order(), [20, 40, 30, 60, 80, 70, 50])

    def test_level_order(self) -> None:
        self.assertEqual(self.bst.level_order(), [50, 30, 70, 20, 40, 60, 80])

    def test_single_node_same_for_all(self) -> None:
        bst = build([42])
        self.assertEqual(bst.in_order(), [42])
        self.assertEqual(bst.pre_order(), [42])
        self.assertEqual(bst.post_order(), [42])
        self.assertEqual(bst.level_order(), [42])

    def test_in_order_stays_sorted_on_random_input(self) -> None:
        import random

        values = random.sample(range(1000), 100)
        bst = build(values)
        self.assertEqual(bst.in_order(), sorted(values))


class TestDunders(unittest.TestCase):
    """These are already written - they should pass once the methods
    they delegate to are implemented."""

    def setUp(self) -> None:
        self.bst = build(BALANCED)

    def test_len(self) -> None:
        self.assertEqual(len(self.bst), 7)

    def test_in_operator(self) -> None:
        self.assertIn(50, self.bst)
        self.assertNotIn(99, self.bst)

    def test_iteration_yields_sorted_order(self) -> None:
        self.assertEqual(list(self.bst), [20, 30, 40, 50, 60, 70, 80])

    def test_iterable_in_comprehension(self) -> None:
        self.assertEqual([v * 2 for v in self.bst if v > 50], [120, 140, 160])


class TestNode(unittest.TestCase):

    def test_starts_with_no_children(self) -> None:
        node = Node(10)
        self.assertEqual(node.value, 10)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)

    def test_repr(self) -> None:
        self.assertEqual(repr(Node(10)), "Node(10)")
        self.assertEqual(repr(Node("a")), "Node('a')")


if __name__ == "__main__":
    unittest.main(verbosity=2)
