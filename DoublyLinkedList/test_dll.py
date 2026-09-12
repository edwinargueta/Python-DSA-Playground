"""Behavior tests for DoublyLinkedList.

Run from inside this directory:

    python3 -m unittest test_dll -v
    python3 test_dll.py            # equivalent

Each test class depends only on the method it names, so you can implement the
methods in any order and a green class always means that method is done.

Two things make that possible. build() wires nodes together by hand, setting
both .next and .prev, so every fixture exists before a single method is
written. walk(), walk_back() and assert_intact() then inspect the node chain
directly, so a test for pop() never has to call to_list() to see what happened.

Every mutation is checked in both directions. A doubly linked list has twice
the pointers to get wrong, and the classic bug is a correct .next chain with a
stale .prev behind it - assert_intact() walks back from the tail specifically
to catch that.

The deliberate exceptions are __len__, __contains__ and __getitem__, which
delegate to _size, contains() and get() by design and stay red until those are
written. TestPrintList and TestPrintReverse assert only the order of the
values printed, not the separator you choose between them.

The helpers reach into .head, .tail, .next, .prev and ._size, but the
assertions never require a particular implementation strategy - iterative and
recursive solutions both satisfy them.
"""

from __future__ import annotations

import io
import re
import unittest
from contextlib import redirect_stdout

from doubly_linked_list import DoublyLinkedList
from node import Node

# The list used throughout:  10 <-> 20 <-> 30 <-> 40 <-> 50
BASE = [10, 20, 30, 40, 50]


def build(values) -> DoublyLinkedList:
    """Return a list holding values, wiring both directions by hand.

    Deliberately avoids append() so that fixtures work before any method is
    implemented, and so a broken append() cannot fail unrelated classes.
    """
    dll = DoublyLinkedList()
    for v in values:
        node = Node(v)
        if dll.head is None:
            dll.head = node
        else:
            node.prev = dll.tail
            dll.tail.next = node
        dll.tail = node
        dll._size += 1
    return dll


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


def walk_back(tail, max_nodes: int = 1000) -> list:
    """Collect values by following .prev from tail."""
    values = []
    node = tail
    while node is not None:
        if len(values) > max_nodes:
            raise AssertionError("walked past max_nodes - the chain has a cycle")
        values.append(node.value)
        node = node.prev
    return values


def last_node(head, max_nodes: int = 1000):
    """Return the final node reachable from head, or None."""
    node, steps = head, 0
    while node is not None and node.next is not None:
        steps += 1
        if steps > max_nodes:
            raise AssertionError("walked past max_nodes - the chain has a cycle")
        node = node.next
    return node


def first_node(tail, max_nodes: int = 1000):
    """Return the first node reachable backwards from tail, or None."""
    node, steps = tail, 0
    while node is not None and node.prev is not None:
        steps += 1
        if steps > max_nodes:
            raise AssertionError("walked past max_nodes - the chain has a cycle")
        node = node.prev
    return node


def broken_link(head, max_nodes: int = 1000):
    """Return the first node whose .next does not point back at it, or None."""
    node, steps = head, 0
    while node is not None and node.next is not None:
        steps += 1
        if steps > max_nodes:
            raise AssertionError("walked past max_nodes - the chain has a cycle")
        if node.next.prev is not node:
            return node
        node = node.next
    return None


# Numbers, quoted strings and bare words - but not the arrows between them,
# so a "<->" separator is never mistaken for a value.
VALUE_TOKEN = re.compile(r"-?\d+(?:\.\d+)?|'[^']*'|[A-Za-z_][A-Za-z0-9_]*")


def printed_values(call) -> list:
    """Run call and return the value tokens it printed, as strings.

    Only the order of the values is captured - the separator is yours to
    choose, so '10 <-> 20 <-> None' and '10, 20' both read as ['10', '20'].
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    tokens = VALUE_TOKEN.findall(buffer.getvalue())
    return [t for t in tokens if t != "None"]


def assert_intact(case, dll, expected) -> None:
    """Assert the list holds exactly expected and its invariants still hold.

    Checks both chains, _size, head, tail, the None terminators and that
    every forward link is mirrored by a backward one. A stale .prev or a
    drifting _size is the usual doubly-linked-list bug; this catches both on
    every mutation.
    """
    expected = list(expected)
    case.assertEqual(walk(dll.head), expected, "forward chain (.next) is wrong")
    case.assertEqual(
        walk_back(dll.tail), expected[::-1], "backward chain (.prev) is wrong"
    )
    case.assertEqual(len(dll), len(expected), "_size is out of step with the chain")
    case.assertIs(dll.tail, last_node(dll.head), "tail is not the final node")
    case.assertIs(dll.head, first_node(dll.tail), "head is not the first node")
    offender = broken_link(dll.head)
    case.assertIsNone(offender, f"{offender!r}.next.prev does not point back at it")
    if expected:
        case.assertIsNotNone(dll.head, "head should not be None")
        case.assertIsNone(dll.head.prev, "head.prev must be None")
        case.assertIsNone(dll.tail.next, "tail.next must be None")
    else:
        case.assertIsNone(dll.head, "head must be None when empty")
        case.assertIsNone(dll.tail, "tail must be None when empty")


class TestEmptyList(unittest.TestCase):
    """An empty list is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.dll = DoublyLinkedList()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.dll), 0)

    def test_head_and_tail_are_none(self) -> None:
        self.assertIsNone(self.dll.head)
        self.assertIsNone(self.dll.tail)

    def test_append_sets_head_and_tail(self) -> None:
        self.dll.append(10)
        assert_intact(self, self.dll, [10])

    def test_prepend_sets_head_and_tail(self) -> None:
        self.dll.prepend(10)
        assert_intact(self, self.dll, [10])

    def test_insert_past_zero_is_rejected(self) -> None:
        self.assertFalse(self.dll.insert(1, 99))
        assert_intact(self, self.dll, [])

    def test_print_list_prints_no_values(self) -> None:
        self.assertEqual(printed_values(self.dll.print_list), [])

    def test_print_reverse_prints_no_values(self) -> None:
        self.assertEqual(printed_values(self.dll.print_reverse), [])

    def test_pop_first_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.pop_first()

    def test_pop_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.pop()

    def test_remove_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.remove(0)

    def test_remove_value_returns_false(self) -> None:
        self.assertFalse(self.dll.remove_value(1))

    def test_clear_is_a_no_op(self) -> None:
        self.dll.clear()
        assert_intact(self, self.dll, [])

    def test_get_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.get(0)

    def test_get_node_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.get_node(0)

    def test_set_value_returns_false(self) -> None:
        self.assertFalse(self.dll.set_value(0, 1))

    def test_index_of_returns_negative_one(self) -> None:
        self.assertEqual(self.dll.index_of(1), -1)

    def test_contains_returns_false(self) -> None:
        self.assertFalse(self.dll.contains(1))

    def test_to_list_is_empty(self) -> None:
        self.assertEqual(self.dll.to_list(), [])

    def test_to_list_reversed_is_empty(self) -> None:
        self.assertEqual(self.dll.to_list_reversed(), [])

    def test_reverse_is_a_no_op(self) -> None:
        self.dll.reverse()
        assert_intact(self, self.dll, [])

    def test_find_middle_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.find_middle()

    def test_nth_from_end_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.nth_from_end(1)

    def test_is_palindrome_is_true(self) -> None:
        self.assertTrue(self.dll.is_palindrome())

    def test_swap_first_last_is_a_no_op(self) -> None:
        self.dll.swap_first_last()
        assert_intact(self, self.dll, [])

    def test_has_no_cycle(self) -> None:
        self.assertFalse(self.dll.has_cycle())

    def test_is_valid(self) -> None:
        self.assertTrue(self.dll.is_valid())

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.dll), [])

    def test_reversed_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(reversed(self.dll)), [])


class TestAppend(unittest.TestCase):

    def test_append_to_empty_sets_head_and_tail(self) -> None:
        dll = DoublyLinkedList()
        dll.append(10)
        assert_intact(self, dll, [10])
        self.assertIs(dll.head, dll.tail)

    def test_appends_go_to_the_end(self) -> None:
        dll = DoublyLinkedList()
        for v in BASE:
            dll.append(v)
        assert_intact(self, dll, BASE)

    def test_second_append_links_both_directions(self) -> None:
        dll = DoublyLinkedList()
        dll.append(10)
        dll.append(20)
        assert_intact(self, dll, [10, 20])
        self.assertIs(dll.tail.prev, dll.head)

    def test_extends_an_existing_list(self) -> None:
        dll = build(BASE)
        dll.append(60)
        assert_intact(self, dll, BASE + [60])

    def test_allows_duplicates(self) -> None:
        dll = build([10])
        dll.append(10)
        assert_intact(self, dll, [10, 10])

    def test_returns_none(self) -> None:
        self.assertIsNone(DoublyLinkedList().append(1))


class TestPrepend(unittest.TestCase):

    def test_prepend_to_empty_sets_head_and_tail(self) -> None:
        dll = DoublyLinkedList()
        dll.prepend(10)
        assert_intact(self, dll, [10])
        self.assertIs(dll.head, dll.tail)

    def test_prepends_go_to_the_front(self) -> None:
        dll = DoublyLinkedList()
        for v in [30, 20, 10]:
            dll.prepend(v)
        assert_intact(self, dll, [10, 20, 30])

    def test_second_prepend_links_both_directions(self) -> None:
        dll = DoublyLinkedList()
        dll.prepend(20)
        dll.prepend(10)
        assert_intact(self, dll, [10, 20])
        self.assertIs(dll.head.next, dll.tail)

    def test_leaves_tail_alone(self) -> None:
        dll = build(BASE)
        original_tail = dll.tail
        dll.prepend(5)
        assert_intact(self, dll, [5] + BASE)
        self.assertIs(dll.tail, original_tail)


class TestInsert(unittest.TestCase):

    def test_into_empty_at_zero(self) -> None:
        dll = DoublyLinkedList()
        self.assertTrue(dll.insert(0, 10))
        assert_intact(self, dll, [10])

    def test_at_front(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.insert(0, 5))
        assert_intact(self, dll, [5] + BASE)

    def test_in_the_middle(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.insert(2, 25))
        assert_intact(self, dll, [10, 20, 25, 30, 40, 50])

    def test_before_the_tail(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.insert(len(BASE) - 1, 45))
        assert_intact(self, dll, [10, 20, 30, 40, 45, 50])

    def test_at_length_appends(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.insert(len(BASE), 60))
        assert_intact(self, dll, BASE + [60])

    def test_into_single_element_list(self) -> None:
        dll = build([10])
        self.assertTrue(dll.insert(1, 20))
        assert_intact(self, dll, [10, 20])

    def test_past_the_end_is_rejected(self) -> None:
        dll = build(BASE)
        self.assertFalse(dll.insert(len(BASE) + 1, 99))
        assert_intact(self, dll, BASE)

    def test_negative_index_is_rejected(self) -> None:
        dll = build(BASE)
        self.assertFalse(dll.insert(-1, 99))
        assert_intact(self, dll, BASE)


class TestPrintList(unittest.TestCase):
    """Asserts the order of the values printed, not the separator."""

    def test_prints_head_first(self) -> None:
        dll = build(BASE)
        printed = printed_values(dll.print_list)
        self.assertEqual(printed, ["10", "20", "30", "40", "50"])

    def test_single_node(self) -> None:
        self.assertEqual(printed_values(build([42]).print_list), ["42"])

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        printed_values(dll.print_list)
        assert_intact(self, dll, BASE)


class TestPrintReverse(unittest.TestCase):
    """Asserts the order of the values printed, not the separator."""

    def test_prints_tail_first(self) -> None:
        dll = build(BASE)
        printed = printed_values(dll.print_reverse)
        self.assertEqual(printed, ["50", "40", "30", "20", "10"])

    def test_single_node(self) -> None:
        self.assertEqual(printed_values(build([42]).print_reverse), ["42"])

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        printed_values(dll.print_reverse)
        assert_intact(self, dll, BASE)


class TestPopFirst(unittest.TestCase):

    def test_returns_the_head_value(self) -> None:
        dll = build(BASE)
        self.assertEqual(dll.pop_first(), 10)
        assert_intact(self, dll, [20, 30, 40, 50])

    def test_clears_the_new_heads_back_pointer(self) -> None:
        dll = build(BASE)
        dll.pop_first()
        self.assertIsNone(dll.head.prev, "the new head still points back")

    def test_only_node_empties_the_list(self) -> None:
        dll = build([10])
        self.assertEqual(dll.pop_first(), 10)
        assert_intact(self, dll, [])

    def test_two_nodes_leaves_tail_as_head(self) -> None:
        dll = build([10, 20])
        self.assertEqual(dll.pop_first(), 10)
        assert_intact(self, dll, [20])
        self.assertIs(dll.head, dll.tail)

    def test_drains_from_the_front(self) -> None:
        dll = build(BASE)
        self.assertEqual([dll.pop_first() for _ in range(len(BASE))], BASE)
        assert_intact(self, dll, [])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            DoublyLinkedList().pop_first()


class TestPop(unittest.TestCase):

    def test_returns_the_tail_value(self) -> None:
        dll = build(BASE)
        self.assertEqual(dll.pop(), 50)
        assert_intact(self, dll, [10, 20, 30, 40])

    def test_clears_the_new_tails_forward_pointer(self) -> None:
        dll = build(BASE)
        dll.pop()
        self.assertIsNone(dll.tail.next, "the new tail still points forward")

    def test_only_node_empties_the_list(self) -> None:
        dll = build([10])
        self.assertEqual(dll.pop(), 10)
        assert_intact(self, dll, [])

    def test_two_nodes_leaves_head_as_tail(self) -> None:
        dll = build([10, 20])
        self.assertEqual(dll.pop(), 20)
        assert_intact(self, dll, [10])
        self.assertIs(dll.head, dll.tail)

    def test_drains_from_the_end(self) -> None:
        dll = build(BASE)
        self.assertEqual([dll.pop() for _ in range(len(BASE))], BASE[::-1])
        assert_intact(self, dll, [])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            DoublyLinkedList().pop()


class TestRemove(unittest.TestCase):

    def test_head(self) -> None:
        dll = build(BASE)
        self.assertEqual(dll.remove(0), 10)
        assert_intact(self, dll, [20, 30, 40, 50])

    def test_middle(self) -> None:
        dll = build(BASE)
        self.assertEqual(dll.remove(2), 30)
        assert_intact(self, dll, [10, 20, 40, 50])

    def test_tail(self) -> None:
        dll = build(BASE)
        self.assertEqual(dll.remove(len(BASE) - 1), 50)
        assert_intact(self, dll, [10, 20, 30, 40])

    def test_only_node(self) -> None:
        dll = build([10])
        self.assertEqual(dll.remove(0), 10)
        assert_intact(self, dll, [])

    def test_removing_every_index_in_turn(self) -> None:
        for i in range(len(BASE)):
            dll = build(BASE)
            self.assertEqual(dll.remove(i), BASE[i], f"index {i}")
            assert_intact(self, dll, BASE[:i] + BASE[i + 1:])

    def test_out_of_range_raises_and_changes_nothing(self) -> None:
        dll = build(BASE)
        with self.assertRaises(IndexError):
            dll.remove(len(BASE))
        assert_intact(self, dll, BASE)

    def test_negative_index_raises(self) -> None:
        dll = build(BASE)
        with self.assertRaises(IndexError):
            dll.remove(-1)
        assert_intact(self, dll, BASE)


class TestRemoveValue(unittest.TestCase):

    def test_removes_the_head(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.remove_value(10))
        assert_intact(self, dll, [20, 30, 40, 50])

    def test_removes_from_the_middle(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.remove_value(30))
        assert_intact(self, dll, [10, 20, 40, 50])

    def test_removes_the_tail(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.remove_value(50))
        assert_intact(self, dll, [10, 20, 30, 40])

    def test_removes_only_the_first_match(self) -> None:
        dll = build([10, 20, 10, 30, 10])
        self.assertTrue(dll.remove_value(10))
        assert_intact(self, dll, [20, 10, 30, 10])

    def test_missing_value_changes_nothing(self) -> None:
        dll = build(BASE)
        self.assertFalse(dll.remove_value(99))
        assert_intact(self, dll, BASE)

    def test_only_node(self) -> None:
        dll = build([10])
        self.assertTrue(dll.remove_value(10))
        assert_intact(self, dll, [])


class TestRemoveNode(unittest.TestCase):
    """The O(1) unlink - the operation a singly linked list cannot offer."""

    def test_middle_node(self) -> None:
        dll = build(BASE)
        node = dll.head.next.next
        self.assertEqual(dll.remove_node(node), 30)
        assert_intact(self, dll, [10, 20, 40, 50])

    def test_head_node(self) -> None:
        dll = build(BASE)
        self.assertEqual(dll.remove_node(dll.head), 10)
        assert_intact(self, dll, [20, 30, 40, 50])

    def test_tail_node(self) -> None:
        dll = build(BASE)
        self.assertEqual(dll.remove_node(dll.tail), 50)
        assert_intact(self, dll, [10, 20, 30, 40])

    def test_only_node(self) -> None:
        dll = build([42])
        self.assertEqual(dll.remove_node(dll.head), 42)
        assert_intact(self, dll, [])

    def test_removing_every_node_in_turn(self) -> None:
        for i in range(len(BASE)):
            dll = build(BASE)
            node = dll.head
            for _ in range(i):
                node = node.next
            self.assertEqual(dll.remove_node(node), BASE[i], f"index {i}")
            assert_intact(self, dll, BASE[:i] + BASE[i + 1:])

    def test_drains_the_whole_list(self) -> None:
        dll = build(BASE)
        while dll.head is not None:
            dll.remove_node(dll.head)
        assert_intact(self, dll, [])


class TestClear(unittest.TestCase):

    def test_empties_the_list(self) -> None:
        dll = build(BASE)
        dll.clear()
        assert_intact(self, dll, [])

    def test_single_node(self) -> None:
        dll = build([42])
        dll.clear()
        assert_intact(self, dll, [])

    def test_already_empty(self) -> None:
        dll = DoublyLinkedList()
        dll.clear()
        assert_intact(self, dll, [])

    def test_list_is_reusable_afterwards(self) -> None:
        dll = build(BASE)
        dll.clear()
        dll.head = Node(1)
        dll.tail = dll.head
        dll._size = 1
        assert_intact(self, dll, [1])


class TestGet(unittest.TestCase):

    def setUp(self) -> None:
        self.dll = build(BASE)

    def test_every_index(self) -> None:
        for i, expected in enumerate(BASE):
            self.assertEqual(self.dll.get(i), expected, f"index {i}")

    def test_first_and_last(self) -> None:
        self.assertEqual(self.dll.get(0), 10)
        self.assertEqual(self.dll.get(len(BASE) - 1), 50)

    def test_single_node(self) -> None:
        self.assertEqual(build([42]).get(0), 42)

    def test_past_the_end_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.get(len(BASE))

    def test_negative_index_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.get(-1)

    def test_does_not_mutate(self) -> None:
        self.dll.get(3)
        assert_intact(self, self.dll, BASE)


class TestGetNode(unittest.TestCase):

    def setUp(self) -> None:
        self.dll = build(BASE)

    def test_returns_the_node_itself(self) -> None:
        node = self.dll.get_node(2)
        self.assertIsInstance(node, Node)
        self.assertEqual(node.value, 30)

    def test_returns_the_node_in_the_chain(self) -> None:
        self.assertIs(self.dll.get_node(0), self.dll.head)
        self.assertIs(self.dll.get_node(len(BASE) - 1), self.dll.tail)

    def test_every_index(self) -> None:
        for i, expected in enumerate(BASE):
            self.assertEqual(self.dll.get_node(i).value, expected, f"index {i}")

    def test_past_the_end_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.get_node(len(BASE))

    def test_negative_index_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.get_node(-1)


class TestSetValue(unittest.TestCase):

    def test_overwrites_in_place(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.set_value(2, 99))
        assert_intact(self, dll, [10, 20, 99, 40, 50])

    def test_head_and_tail(self) -> None:
        dll = build(BASE)
        self.assertTrue(dll.set_value(0, 1))
        self.assertTrue(dll.set_value(len(BASE) - 1, 5))
        assert_intact(self, dll, [1, 20, 30, 40, 5])

    def test_reuses_the_existing_node(self) -> None:
        dll = build(BASE)
        node = dll.head.next
        dll.set_value(1, 99)
        self.assertIs(dll.head.next, node, "the node was replaced, not updated")

    def test_out_of_range_changes_nothing(self) -> None:
        dll = build(BASE)
        self.assertFalse(dll.set_value(len(BASE), 99))
        self.assertFalse(dll.set_value(-1, 99))
        assert_intact(self, dll, BASE)


class TestIndexOf(unittest.TestCase):

    def test_every_value(self) -> None:
        dll = build(BASE)
        for i, v in enumerate(BASE):
            self.assertEqual(dll.index_of(v), i)

    def test_absent_value(self) -> None:
        self.assertEqual(build(BASE).index_of(99), -1)

    def test_returns_the_first_match(self) -> None:
        self.assertEqual(build([10, 20, 10]).index_of(10), 0)

    def test_single_node(self) -> None:
        self.assertEqual(build([42]).index_of(42), 0)

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        dll.index_of(30)
        assert_intact(self, dll, BASE)


class TestContains(unittest.TestCase):

    def setUp(self) -> None:
        self.dll = build(BASE)

    def test_every_value(self) -> None:
        for v in BASE:
            self.assertTrue(self.dll.contains(v), f"missing {v}")

    def test_head_and_tail_values(self) -> None:
        self.assertTrue(self.dll.contains(10))
        self.assertTrue(self.dll.contains(50))

    def test_absent_value(self) -> None:
        self.assertFalse(self.dll.contains(99))

    def test_does_not_mutate(self) -> None:
        self.dll.contains(99)
        assert_intact(self, self.dll, BASE)


class TestToList(unittest.TestCase):

    def test_matches_the_chain(self) -> None:
        self.assertEqual(build(BASE).to_list(), BASE)

    def test_single_node(self) -> None:
        self.assertEqual(build([42]).to_list(), [42])

    def test_keeps_duplicates(self) -> None:
        self.assertEqual(build([10, 10, 20]).to_list(), [10, 10, 20])

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        dll.to_list()
        assert_intact(self, dll, BASE)

    def test_returns_a_detached_copy(self) -> None:
        dll = build(BASE)
        dll.to_list().append(99)
        assert_intact(self, dll, BASE)


class TestToListReversed(unittest.TestCase):

    def test_returns_tail_first(self) -> None:
        self.assertEqual(build(BASE).to_list_reversed(), BASE[::-1])

    def test_single_node(self) -> None:
        self.assertEqual(build([42]).to_list_reversed(), [42])

    def test_two_nodes(self) -> None:
        self.assertEqual(build([10, 20]).to_list_reversed(), [20, 10])

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        dll.to_list_reversed()
        assert_intact(self, dll, BASE)

    def test_returns_a_detached_copy(self) -> None:
        dll = build(BASE)
        dll.to_list_reversed().append(99)
        assert_intact(self, dll, BASE)


class TestReverse(unittest.TestCase):

    def test_reverses_the_values(self) -> None:
        dll = build(BASE)
        dll.reverse()
        assert_intact(self, dll, BASE[::-1])

    def test_swaps_head_and_tail(self) -> None:
        dll = build(BASE)
        original_head, original_tail = dll.head, dll.tail
        dll.reverse()
        self.assertIs(dll.head, original_tail)
        self.assertIs(dll.tail, original_head)

    def test_two_nodes(self) -> None:
        dll = build([10, 20])
        dll.reverse()
        assert_intact(self, dll, [20, 10])

    def test_single_node_is_unchanged(self) -> None:
        dll = build([42])
        dll.reverse()
        assert_intact(self, dll, [42])

    def test_reversing_twice_restores_the_original(self) -> None:
        dll = build(BASE)
        dll.reverse()
        dll.reverse()
        assert_intact(self, dll, BASE)

    def test_reuses_the_original_nodes(self) -> None:
        dll = build(BASE)
        nodes = {id(dll.head), id(dll.head.next), id(dll.tail)}
        dll.reverse()
        after = {id(dll.tail), id(dll.tail.prev), id(dll.head)}
        self.assertEqual(nodes, after, "reverse rebuilt instead of relinking")


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
            DoublyLinkedList().find_middle()

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        dll.find_middle()
        assert_intact(self, dll, BASE)


class TestNthFromEnd(unittest.TestCase):

    def setUp(self) -> None:
        self.dll = build(BASE)

    def test_one_is_the_tail(self) -> None:
        self.assertEqual(self.dll.nth_from_end(1), 50)

    def test_length_is_the_head(self) -> None:
        self.assertEqual(self.dll.nth_from_end(len(BASE)), 10)

    def test_every_position(self) -> None:
        for n in range(1, len(BASE) + 1):
            self.assertEqual(self.dll.nth_from_end(n), BASE[-n], f"n={n}")

    def test_zero_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.nth_from_end(0)

    def test_past_the_head_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.nth_from_end(len(BASE) + 1)

    def test_negative_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll.nth_from_end(-1)


class TestIsPalindrome(unittest.TestCase):

    def test_odd_length_palindrome(self) -> None:
        self.assertTrue(build([10, 20, 10]).is_palindrome())

    def test_even_length_palindrome(self) -> None:
        self.assertTrue(build([10, 20, 20, 10]).is_palindrome())

    def test_not_a_palindrome(self) -> None:
        self.assertFalse(build(BASE).is_palindrome())

    def test_differs_only_at_the_ends(self) -> None:
        self.assertFalse(build([10, 20, 30, 20, 99]).is_palindrome())

    def test_single_node(self) -> None:
        self.assertTrue(build([42]).is_palindrome())

    def test_two_equal_nodes(self) -> None:
        self.assertTrue(build([10, 10]).is_palindrome())

    def test_two_different_nodes(self) -> None:
        self.assertFalse(build([10, 20]).is_palindrome())

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        dll.is_palindrome()
        assert_intact(self, dll, BASE)


class TestSwapFirstLast(unittest.TestCase):
    """The nodes move, not the values - identity is what is asserted."""

    def test_swaps_the_end_values(self) -> None:
        dll = build(BASE)
        dll.swap_first_last()
        assert_intact(self, dll, [50, 20, 30, 40, 10])

    def test_moves_the_nodes_themselves(self) -> None:
        dll = build(BASE)
        original_head, original_tail = dll.head, dll.tail
        dll.swap_first_last()
        self.assertIs(dll.head, original_tail, "the tail node did not become head")
        self.assertIs(dll.tail, original_head, "the head node did not become tail")

    def test_two_nodes(self) -> None:
        dll = build([10, 20])
        dll.swap_first_last()
        assert_intact(self, dll, [20, 10])

    def test_three_nodes(self) -> None:
        dll = build([10, 20, 30])
        dll.swap_first_last()
        assert_intact(self, dll, [30, 20, 10])

    def test_single_node_is_unchanged(self) -> None:
        dll = build([42])
        dll.swap_first_last()
        assert_intact(self, dll, [42])

    def test_swapping_twice_restores_the_original(self) -> None:
        dll = build(BASE)
        dll.swap_first_last()
        dll.swap_first_last()
        assert_intact(self, dll, BASE)


class TestHasCycle(unittest.TestCase):
    """Cycles are wired by hand - no public method can create one."""

    def test_empty_list(self) -> None:
        self.assertFalse(DoublyLinkedList().has_cycle())

    def test_single_node(self) -> None:
        self.assertFalse(build([10]).has_cycle())

    def test_ordinary_list(self) -> None:
        self.assertFalse(build(BASE).has_cycle())

    def test_self_loop(self) -> None:
        dll = build([10])
        dll.head.next = dll.head
        self.assertTrue(dll.has_cycle())

    def test_tail_points_at_head(self) -> None:
        dll = build(BASE)
        dll.tail.next = dll.head
        self.assertTrue(dll.has_cycle())

    def test_tail_points_into_the_middle(self) -> None:
        dll = build(BASE)
        dll.tail.next = dll.head.next.next
        self.assertTrue(dll.has_cycle())


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand. Cycle detection is has_cycle's job, so
    every list here terminates."""

    def test_empty_list(self) -> None:
        self.assertTrue(DoublyLinkedList().is_valid())

    def test_single_node(self) -> None:
        self.assertTrue(build([42]).is_valid())

    def test_ordinary_list(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_forward_link_not_mirrored(self) -> None:
        dll = build(BASE)
        dll.head.next.prev = dll.tail
        self.assertFalse(dll.is_valid())

    def test_backward_link_not_mirrored(self) -> None:
        dll = build(BASE)
        dll.tail.prev = dll.head
        self.assertFalse(dll.is_valid())

    def test_head_still_points_backwards(self) -> None:
        dll = build(BASE)
        dll.head.prev = Node(99)
        self.assertFalse(dll.is_valid())

    def test_tail_still_points_forwards(self) -> None:
        dll = build(BASE)
        dll.tail.next = Node(99)
        self.assertFalse(dll.is_valid())

    def test_size_out_of_step(self) -> None:
        dll = build(BASE)
        dll._size += 1
        self.assertFalse(dll.is_valid())

    def test_head_set_without_tail(self) -> None:
        dll = DoublyLinkedList()
        dll.head = Node(10)
        dll._size = 1
        self.assertFalse(dll.is_valid())


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _size directly, but __contains__ and __getitem__
    delegate by design, so they stay red until contains() and get() are
    written."""

    def setUp(self) -> None:
        self.dll = build(BASE)

    def test_len(self) -> None:
        self.assertEqual(len(self.dll), 5)
        self.assertEqual(len(DoublyLinkedList()), 0)

    def test_in_operator(self) -> None:
        self.assertIn(30, self.dll)
        self.assertNotIn(99, self.dll)

    def test_indexing(self) -> None:
        self.assertEqual(self.dll[0], 10)
        self.assertEqual(self.dll[4], 50)

    def test_indexing_out_of_range_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.dll[len(BASE)]


class TestSetItem(unittest.TestCase):
    """__setitem__ has its own body in the stub; if you implement it by
    calling set_value(), this class inherits that dependency."""

    def test_assigns_by_index(self) -> None:
        dll = build(BASE)
        dll[2] = 99
        assert_intact(self, dll, [10, 20, 99, 40, 50])

    def test_head_and_tail(self) -> None:
        dll = build(BASE)
        dll[0] = 1
        dll[len(BASE) - 1] = 5
        assert_intact(self, dll, [1, 20, 30, 40, 5])

    def test_out_of_range_raises(self) -> None:
        dll = build(BASE)
        with self.assertRaises(IndexError):
            dll[len(BASE)] = 99
        assert_intact(self, dll, BASE)

    def test_negative_index_raises(self) -> None:
        dll = build(BASE)
        with self.assertRaises(IndexError):
            dll[-1] = 99
        assert_intact(self, dll, BASE)


class TestIteration(unittest.TestCase):

    def test_yields_head_first(self) -> None:
        self.assertEqual(list(build(BASE)), BASE)

    def test_empty_list_yields_nothing(self) -> None:
        self.assertEqual(list(DoublyLinkedList()), [])

    def test_usable_in_a_comprehension(self) -> None:
        self.assertEqual([v * 2 for v in build(BASE) if v > 30], [80, 100])

    def test_two_iterations_are_independent(self) -> None:
        dll = build(BASE)
        self.assertEqual(list(dll), BASE)
        self.assertEqual(list(dll), BASE, "the second pass came up short")


class TestReversedIteration(unittest.TestCase):

    def test_yields_tail_first(self) -> None:
        self.assertEqual(list(reversed(build(BASE))), BASE[::-1])

    def test_empty_list_yields_nothing(self) -> None:
        self.assertEqual(list(reversed(DoublyLinkedList())), [])

    def test_single_node(self) -> None:
        self.assertEqual(list(reversed(build([42]))), [42])

    def test_two_iterations_are_independent(self) -> None:
        dll = build(BASE)
        self.assertEqual(list(reversed(dll)), BASE[::-1])
        self.assertEqual(list(reversed(dll)), BASE[::-1], "second pass differed")

    def test_does_not_mutate(self) -> None:
        dll = build(BASE)
        list(reversed(dll))
        assert_intact(self, dll, BASE)


class TestRepr(unittest.TestCase):

    def test_repr(self) -> None:
        self.assertEqual(repr(build([1, 2, 3])), "DoublyLinkedList([1, 2, 3])")

    def test_repr_when_empty(self) -> None:
        self.assertEqual(repr(DoublyLinkedList()), "DoublyLinkedList([])")

    def test_repr_of_strings_quotes_them(self) -> None:
        self.assertEqual(repr(build(["a"])), "DoublyLinkedList(['a'])")


class TestNode(unittest.TestCase):

    def test_starts_unlinked(self) -> None:
        node = Node(10)
        self.assertEqual(node.value, 10)
        self.assertIsNone(node.next)
        self.assertIsNone(node.prev)

    def test_repr(self) -> None:
        self.assertEqual(repr(Node(10)), "Node(10)")
        self.assertEqual(repr(Node("a")), "Node('a')")


if __name__ == "__main__":
    unittest.main(verbosity=2)
