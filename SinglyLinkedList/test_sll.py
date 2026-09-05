"""Behavior tests for SinglyLinkedList.

Run from inside this directory:

    python3 -m unittest test_sll -v
    python3 test_sll.py            # equivalent

Each test class depends only on the method it names, so you can implement the
methods in any order and a green class always means that method is done.

Two things make that possible. build() wires nodes together by hand instead of
calling append(), so every fixture exists before a single method is written.
walk() and assert_intact() then inspect the node chain directly, so a test for
pop() never has to call to_list() to see what happened.

The deliberate exceptions are __contains__ and __getitem__, which delegate to
contains() and get() by design and stay red until those are written.

The helpers reach into .head, .tail, .next and ._size, but the assertions
never require a particular implementation strategy - recursive and iterative
solutions both satisfy them.
"""

from __future__ import annotations

import unittest

from node import Node
from singly_linked_list import SinglyLinkedList

# The list used throughout:  10 -> 20 -> 30 -> 40 -> 50
BASE = [10, 20, 30, 40, 50]


def build(values) -> SinglyLinkedList:
    """Return a list holding values, wiring the nodes directly.

    Deliberately avoids append() so that fixtures work before any method is
    implemented, and so a broken append() cannot fail unrelated classes.
    """
    sll = SinglyLinkedList()
    for v in values:
        node = Node(v)
        if sll.head is None:
            sll.head = node
        else:
            sll.tail.next = node
        sll.tail = node
        sll._size += 1
    return sll


def walk(head, max_nodes: int = 1000) -> list:
    """Collect values by following .next from head."""
    values = []
    node = head
    while node is not None:
        if len(values) > max_nodes:
            raise AssertionError("walked past max_nodes - the chain has a cycle")
        values.append(node.value)
        node = node.next
    return values


def last_node(head):
    """Return the final node reachable from head, or None."""
    node = head
    while node is not None and node.next is not None:
        node = node.next
    return node


def assert_intact(case, sll, expected) -> None:
    """Assert the list holds exactly expected and its invariants still hold.

    Checks the node chain, _size, head, tail, and that tail.next is None.
    A stale tail or a drifting _size is the usual linked-list bug, and this
    catches both on every mutation.
    """
    case.assertEqual(walk(sll.head), list(expected), "node chain is wrong")
    case.assertEqual(len(sll), len(expected), "_size is out of step with the chain")
    case.assertIs(sll.tail, last_node(sll.head), "tail is not the final node")
    if expected:
        case.assertIsNotNone(sll.head, "head should not be None")
        case.assertIsNone(sll.tail.next, "tail.next must be None")
    else:
        case.assertIsNone(sll.head, "head must be None when empty")
        case.assertIsNone(sll.tail, "tail must be None when empty")


class TestEmptyList(unittest.TestCase):
    """An empty list is the edge case most implementations get wrong."""

    def setUp(self) -> None:
        self.sll = SinglyLinkedList()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.sll), 0)

    def test_head_and_tail_are_none(self) -> None:
        self.assertIsNone(self.sll.head)
        self.assertIsNone(self.sll.tail)

    def test_contains_returns_false(self) -> None:
        self.assertFalse(self.sll.contains(1))

    def test_index_of_returns_negative_one(self) -> None:
        self.assertEqual(self.sll.index_of(1), -1)

    def test_to_list_is_empty(self) -> None:
        self.assertEqual(self.sll.to_list(), [])

    def test_pop_first_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.pop_first()

    def test_pop_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.pop()

    def test_get_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.get(0)

    def test_remove_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.remove(0)

    def test_remove_value_returns_false(self) -> None:
        self.assertFalse(self.sll.remove_value(1))

    def test_set_value_returns_false(self) -> None:
        self.assertFalse(self.sll.set_value(0, 1))

    def test_find_middle_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.find_middle()

    def test_nth_from_end_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.nth_from_end(1)

    def test_has_no_cycle(self) -> None:
        self.assertFalse(self.sll.has_cycle())

    def test_reverse_is_a_no_op(self) -> None:
        self.sll.reverse()
        assert_intact(self, self.sll, [])

    def test_clear_is_a_no_op(self) -> None:
        self.sll.clear()
        assert_intact(self, self.sll, [])

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.sll), [])


class TestAppend(unittest.TestCase):

    def test_append_to_empty_sets_head_and_tail(self) -> None:
        sll = SinglyLinkedList()
        sll.append(10)
        assert_intact(self, sll, [10])
        self.assertIs(sll.head, sll.tail)

    def test_appends_go_to_the_end(self) -> None:
        sll = SinglyLinkedList()
        for v in BASE:
            sll.append(v)
        assert_intact(self, sll, BASE)

    def test_extends_an_existing_list(self) -> None:
        sll = build(BASE)
        sll.append(60)
        assert_intact(self, sll, BASE + [60])

    def test_allows_duplicates(self) -> None:
        sll = build([10])
        sll.append(10)
        assert_intact(self, sll, [10, 10])

    def test_returns_none(self) -> None:
        self.assertIsNone(SinglyLinkedList().append(1))


class TestPrepend(unittest.TestCase):

    def test_prepend_to_empty_sets_head_and_tail(self) -> None:
        sll = SinglyLinkedList()
        sll.prepend(10)
        assert_intact(self, sll, [10])
        self.assertIs(sll.head, sll.tail)

    def test_prepends_go_to_the_front(self) -> None:
        sll = SinglyLinkedList()
        for v in [30, 20, 10]:
            sll.prepend(v)
        assert_intact(self, sll, [10, 20, 30])

    def test_leaves_tail_alone(self) -> None:
        sll = build(BASE)
        original_tail = sll.tail
        sll.prepend(5)
        assert_intact(self, sll, [5] + BASE)
        self.assertIs(sll.tail, original_tail)


class TestInsert(unittest.TestCase):

    def test_into_empty_at_zero(self) -> None:
        sll = SinglyLinkedList()
        self.assertTrue(sll.insert(0, 10))
        assert_intact(self, sll, [10])

    def test_at_front(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.insert(0, 5))
        assert_intact(self, sll, [5] + BASE)

    def test_in_the_middle(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.insert(2, 25))
        assert_intact(self, sll, [10, 20, 25, 30, 40, 50])

    def test_at_length_appends(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.insert(len(BASE), 60))
        assert_intact(self, sll, BASE + [60])

    def test_past_the_end_is_rejected(self) -> None:
        sll = build(BASE)
        self.assertFalse(sll.insert(len(BASE) + 1, 99))
        assert_intact(self, sll, BASE)

    def test_negative_index_is_rejected(self) -> None:
        sll = build(BASE)
        self.assertFalse(sll.insert(-1, 99))
        assert_intact(self, sll, BASE)


class TestPopFirst(unittest.TestCase):

    def test_returns_the_head_value(self) -> None:
        sll = build(BASE)
        self.assertEqual(sll.pop_first(), 10)
        assert_intact(self, sll, [20, 30, 40, 50])

    def test_only_node_empties_the_list(self) -> None:
        sll = build([10])
        self.assertEqual(sll.pop_first(), 10)
        assert_intact(self, sll, [])

    def test_drains_from_the_front(self) -> None:
        sll = build(BASE)
        self.assertEqual([sll.pop_first() for _ in range(len(BASE))], BASE)
        assert_intact(self, sll, [])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            SinglyLinkedList().pop_first()


class TestPop(unittest.TestCase):

    def test_returns_the_tail_value(self) -> None:
        sll = build(BASE)
        self.assertEqual(sll.pop(), 50)
        assert_intact(self, sll, [10, 20, 30, 40])

    def test_only_node_empties_the_list(self) -> None:
        sll = build([10])
        self.assertEqual(sll.pop(), 10)
        assert_intact(self, sll, [])

    def test_two_nodes_leaves_head_as_tail(self) -> None:
        sll = build([10, 20])
        self.assertEqual(sll.pop(), 20)
        assert_intact(self, sll, [10])
        self.assertIs(sll.head, sll.tail)

    def test_drains_from_the_end(self) -> None:
        sll = build(BASE)
        self.assertEqual([sll.pop() for _ in range(len(BASE))], BASE[::-1])
        assert_intact(self, sll, [])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            SinglyLinkedList().pop()


class TestRemove(unittest.TestCase):

    def test_head(self) -> None:
        sll = build(BASE)
        self.assertEqual(sll.remove(0), 10)
        assert_intact(self, sll, [20, 30, 40, 50])

    def test_middle(self) -> None:
        sll = build(BASE)
        self.assertEqual(sll.remove(2), 30)
        assert_intact(self, sll, [10, 20, 40, 50])

    def test_tail(self) -> None:
        sll = build(BASE)
        self.assertEqual(sll.remove(len(BASE) - 1), 50)
        assert_intact(self, sll, [10, 20, 30, 40])

    def test_only_node(self) -> None:
        sll = build([10])
        self.assertEqual(sll.remove(0), 10)
        assert_intact(self, sll, [])

    def test_out_of_range_raises_and_changes_nothing(self) -> None:
        sll = build(BASE)
        with self.assertRaises(IndexError):
            sll.remove(len(BASE))
        assert_intact(self, sll, BASE)

    def test_negative_index_raises(self) -> None:
        sll = build(BASE)
        with self.assertRaises(IndexError):
            sll.remove(-1)
        assert_intact(self, sll, BASE)


class TestRemoveValue(unittest.TestCase):

    def test_removes_the_head(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.remove_value(10))
        assert_intact(self, sll, [20, 30, 40, 50])

    def test_removes_from_the_middle(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.remove_value(30))
        assert_intact(self, sll, [10, 20, 40, 50])

    def test_removes_the_tail(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.remove_value(50))
        assert_intact(self, sll, [10, 20, 30, 40])

    def test_removes_only_the_first_match(self) -> None:
        sll = build([10, 20, 10, 30, 10])
        self.assertTrue(sll.remove_value(10))
        assert_intact(self, sll, [20, 10, 30, 10])

    def test_missing_value_changes_nothing(self) -> None:
        sll = build(BASE)
        self.assertFalse(sll.remove_value(99))
        assert_intact(self, sll, BASE)

    def test_only_node(self) -> None:
        sll = build([10])
        self.assertTrue(sll.remove_value(10))
        assert_intact(self, sll, [])


class TestGet(unittest.TestCase):

    def setUp(self) -> None:
        self.sll = build(BASE)

    def test_every_index(self) -> None:
        for i, expected in enumerate(BASE):
            self.assertEqual(self.sll.get(i), expected, f"index {i}")

    def test_first_and_last(self) -> None:
        self.assertEqual(self.sll.get(0), 10)
        self.assertEqual(self.sll.get(len(BASE) - 1), 50)

    def test_past_the_end_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.get(len(BASE))

    def test_negative_index_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.get(-1)


class TestSetValue(unittest.TestCase):

    def test_overwrites_in_place(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.set_value(2, 99))
        assert_intact(self, sll, [10, 20, 99, 40, 50])

    def test_head_and_tail(self) -> None:
        sll = build(BASE)
        self.assertTrue(sll.set_value(0, 1))
        self.assertTrue(sll.set_value(len(BASE) - 1, 5))
        assert_intact(self, sll, [1, 20, 30, 40, 5])

    def test_out_of_range_changes_nothing(self) -> None:
        sll = build(BASE)
        self.assertFalse(sll.set_value(len(BASE), 99))
        self.assertFalse(sll.set_value(-1, 99))
        assert_intact(self, sll, BASE)


class TestSearch(unittest.TestCase):

    def setUp(self) -> None:
        self.sll = build(BASE)

    def test_contains_every_value(self) -> None:
        for v in BASE:
            self.assertTrue(self.sll.contains(v), f"missing {v}")

    def test_contains_rejects_absent_value(self) -> None:
        self.assertFalse(self.sll.contains(99))

    def test_index_of_every_value(self) -> None:
        for i, v in enumerate(BASE):
            self.assertEqual(self.sll.index_of(v), i)

    def test_index_of_absent_value(self) -> None:
        self.assertEqual(self.sll.index_of(99), -1)

    def test_index_of_returns_the_first_match(self) -> None:
        sll = build([10, 20, 10])
        self.assertEqual(sll.index_of(10), 0)


class TestToList(unittest.TestCase):

    def test_matches_the_chain(self) -> None:
        self.assertEqual(build(BASE).to_list(), BASE)

    def test_single_node(self) -> None:
        self.assertEqual(build([42]).to_list(), [42])

    def test_does_not_mutate(self) -> None:
        sll = build(BASE)
        sll.to_list()
        assert_intact(self, sll, BASE)

    def test_returns_a_detached_copy(self) -> None:
        sll = build(BASE)
        sll.to_list().append(99)
        assert_intact(self, sll, BASE)


class TestReverse(unittest.TestCase):

    def test_reverses_the_values(self) -> None:
        sll = build(BASE)
        sll.reverse()
        assert_intact(self, sll, BASE[::-1])

    def test_swaps_head_and_tail(self) -> None:
        sll = build(BASE)
        original_head, original_tail = sll.head, sll.tail
        sll.reverse()
        self.assertIs(sll.head, original_tail)
        self.assertIs(sll.tail, original_head)

    def test_two_nodes(self) -> None:
        sll = build([10, 20])
        sll.reverse()
        assert_intact(self, sll, [20, 10])

    def test_single_node_is_unchanged(self) -> None:
        sll = build([42])
        sll.reverse()
        assert_intact(self, sll, [42])

    def test_reversing_twice_restores_the_original(self) -> None:
        sll = build(BASE)
        sll.reverse()
        sll.reverse()
        assert_intact(self, sll, BASE)


class TestFindMiddle(unittest.TestCase):

    def test_odd_length(self) -> None:
        self.assertEqual(build(BASE).find_middle(), 30)

    def test_even_length_returns_the_second_middle(self) -> None:
        self.assertEqual(build([10, 20, 30, 40]).find_middle(), 30)

    def test_two_nodes(self) -> None:
        self.assertEqual(build([10, 20]).find_middle(), 20)

    def test_single_node(self) -> None:
        self.assertEqual(build([42]).find_middle(), 42)

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            SinglyLinkedList().find_middle()


class TestNthFromEnd(unittest.TestCase):

    def setUp(self) -> None:
        self.sll = build(BASE)

    def test_one_is_the_tail(self) -> None:
        self.assertEqual(self.sll.nth_from_end(1), 50)

    def test_length_is_the_head(self) -> None:
        self.assertEqual(self.sll.nth_from_end(len(BASE)), 10)

    def test_every_position(self) -> None:
        for n in range(1, len(BASE) + 1):
            self.assertEqual(self.sll.nth_from_end(n), BASE[-n], f"n={n}")

    def test_zero_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.nth_from_end(0)

    def test_past_the_head_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.sll.nth_from_end(len(BASE) + 1)


class TestHasCycle(unittest.TestCase):
    """Cycles are wired by hand - no public method can create one."""

    def test_empty_list(self) -> None:
        self.assertFalse(SinglyLinkedList().has_cycle())

    def test_single_node(self) -> None:
        self.assertFalse(build([10]).has_cycle())

    def test_ordinary_list(self) -> None:
        self.assertFalse(build(BASE).has_cycle())

    def test_self_loop(self) -> None:
        sll = build([10])
        sll.head.next = sll.head
        self.assertTrue(sll.has_cycle())

    def test_tail_points_at_head(self) -> None:
        sll = build(BASE)
        sll.tail.next = sll.head
        self.assertTrue(sll.has_cycle())

    def test_tail_points_into_the_middle(self) -> None:
        sll = build(BASE)
        sll.tail.next = sll.head.next.next
        self.assertTrue(sll.has_cycle())


class TestClear(unittest.TestCase):

    def test_empties_the_list(self) -> None:
        sll = build(BASE)
        sll.clear()
        assert_intact(self, sll, [])

    def test_single_node(self) -> None:
        sll = build([42])
        sll.clear()
        assert_intact(self, sll, [])

    def test_list_is_reusable_afterwards(self) -> None:
        sll = build(BASE)
        sll.clear()
        sll.head = Node(1)
        sll.tail = sll.head
        sll._size = 1
        assert_intact(self, sll, [1])


class TestDunders(unittest.TestCase):
    """__contains__ and __getitem__ delegate by design, so they stay red
    until contains() and get() are written."""

    def setUp(self) -> None:
        self.sll = build(BASE)

    def test_len(self) -> None:
        self.assertEqual(len(self.sll), 5)

    def test_in_operator(self) -> None:
        self.assertIn(30, self.sll)
        self.assertNotIn(99, self.sll)

    def test_indexing(self) -> None:
        self.assertEqual(self.sll[0], 10)
        self.assertEqual(self.sll[4], 50)

    def test_iteration_yields_head_first(self) -> None:
        self.assertEqual(list(self.sll), BASE)

    def test_iterable_in_comprehension(self) -> None:
        self.assertEqual([v * 2 for v in self.sll if v > 30], [80, 100])

    def test_repr(self) -> None:
        self.assertEqual(repr(build([1, 2, 3])), "SinglyLinkedList([1, 2, 3])")

    def test_repr_when_empty(self) -> None:
        self.assertEqual(repr(SinglyLinkedList()), "SinglyLinkedList([])")


class TestNode(unittest.TestCase):

    def test_starts_with_no_next(self) -> None:
        node = Node(10)
        self.assertEqual(node.value, 10)
        self.assertIsNone(node.next)

    def test_repr(self) -> None:
        self.assertEqual(repr(Node(10)), "Node(10)")
        self.assertEqual(repr(Node("a")), "Node('a')")


if __name__ == "__main__":
    unittest.main(verbosity=2)
