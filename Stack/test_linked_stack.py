"""Behavior tests for LinkedStack, the node-backed stack.

Run from inside this directory:

    python3 -m unittest test_linked_stack -v
    python3 test_linked_stack.py            # equivalent

Each test class depends only on the method it names, so you can implement
the methods in any order and a green class always means that method is done.

build() wires the nodes together by hand, so every fixture exists before a
single method is written and a broken push() cannot fail an unrelated
class. walk() then follows .next from _top directly, which is why a test
for pop() never calls to_list() to see what happened.

Fixtures and assertions take their values bottom first, the order you would
push them, so build([10, 20, 30]) has 30 on top. to_list() and iteration
run the other way, top first, and those tests say so explicitly.

assert_intact() checks _size against the chain on every mutation. A stack
whose _size has drifted answers len() wrongly while every value is still
reachable, and nothing but a walk of the chain notices.

The deliberate couplings: __len__ reads _size and __contains__ delegates to
contains(), so it stays red until contains() is written. The print tests
assert that the values appear in top-to-bottom order, not the layout or
separator you print them with.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from linked_stack import LinkedStack
from node import Node

# The stack used throughout: 10 at the bottom, 40 on top.
BASE = [10, 20, 30, 40]


def build(values) -> LinkedStack:
    """Return a stack holding values, bottom first, wiring nodes by hand.

    Deliberately avoids push() so that fixtures work before any method is
    implemented, and so a broken push() cannot fail unrelated classes.
    """
    stack = LinkedStack()
    for value in values:
        node = Node(value)
        node.next = stack._top
        stack._top = node
        stack._size += 1
    return stack


def walk(top, max_nodes: int = 1000) -> list:
    """Collect values by following .next from the top."""
    values, node = [], top
    while node is not None:
        if len(values) > max_nodes:
            raise AssertionError("walked past max_nodes - the chain has a cycle")
        values.append(node.value)
        node = node.next
    return values


def printed(call) -> str:
    """Run call and return everything it wrote to stdout."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


def assert_printed_in_order(case, output, values) -> None:
    """Assert each value appears in output, in this order.

    Only the order is checked - the separator and any labels around the
    values are yours to choose.
    """
    start = 0
    for value in values:
        found = output.find(str(value), start)
        case.assertNotEqual(
            found, -1, f"{value!r} is missing or out of order in the output"
        )
        start = found + len(str(value))


def assert_intact(case, stack, expected) -> None:
    """Assert the stack holds exactly expected, bottom first, and is intact.

    Checks the chain, _size, and that an empty stack has really let go of
    its top.
    """
    expected = list(expected)
    case.assertEqual(walk(stack._top), expected[::-1], "the node chain is wrong")
    case.assertEqual(len(stack), len(expected), "_size is out of step with the chain")
    if not expected:
        case.assertIsNone(stack._top, "_top should be None when the stack is empty")
    else:
        case.assertIsNotNone(stack._top, "_top should not be None")
        case.assertEqual(stack._top.value, expected[-1], "the wrong node is on top")


class TestEmptyStack(unittest.TestCase):
    """An empty stack is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.stack = LinkedStack()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.stack), 0)

    def test_top_is_none(self) -> None:
        self.assertIsNone(self.stack._top)

    def test_push_stores_the_first_value(self) -> None:
        self.stack.push(10)
        assert_intact(self, self.stack, [10])

    def test_pop_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.stack.pop()

    def test_peek_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.stack.peek()

    def test_is_empty_is_true(self) -> None:
        self.assertTrue(self.stack.is_empty())

    def test_contains_is_false(self) -> None:
        self.assertFalse(self.stack.contains(10))

    def test_to_list_is_empty(self) -> None:
        self.assertEqual(self.stack.to_list(), [])

    def test_print_stack_prints_no_values(self) -> None:
        self.assertNotIn("10", printed(self.stack.print_stack))

    def test_clear_is_a_no_op(self) -> None:
        self.stack.clear()
        assert_intact(self, self.stack, [])

    def test_is_valid(self) -> None:
        self.assertTrue(self.stack.is_valid())

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.stack), [])

    def test_repr(self) -> None:
        self.assertEqual(repr(self.stack), "LinkedStack([])")


class TestPush(unittest.TestCase):

    def test_stores_the_first_value(self) -> None:
        stack = LinkedStack()
        stack.push(10)
        assert_intact(self, stack, [10])

    def test_stacks_values_in_order(self) -> None:
        stack = LinkedStack()
        for value in BASE:
            stack.push(value)
        assert_intact(self, stack, BASE)

    def test_the_new_node_points_at_the_old_top(self) -> None:
        stack = build([10])
        was_top = stack._top
        stack.push(20)
        self.assertIs(stack._top.next, was_top, "the old top was cut loose")

    def test_keeps_duplicates(self) -> None:
        stack = LinkedStack()
        for value in [7, 7, 7]:
            stack.push(value)
        assert_intact(self, stack, [7, 7, 7])

    def test_stores_none_as_a_value(self) -> None:
        stack = LinkedStack()
        stack.push(None)
        self.assertEqual(len(stack), 1, "None was not counted as a value")

    def test_size_tracks_every_push(self) -> None:
        stack = LinkedStack()
        for i, value in enumerate(BASE, start=1):
            stack.push(value)
            self.assertEqual(len(stack), i, f"size wrong after {i} pushes")

    def test_onto_a_built_stack(self) -> None:
        stack = build(BASE)
        stack.push(50)
        assert_intact(self, stack, BASE + [50])

    def test_many_values(self) -> None:
        stack = LinkedStack()
        for value in range(200):
            stack.push(value)
        assert_intact(self, stack, list(range(200)))


class TestPop(unittest.TestCase):

    def test_returns_the_top(self) -> None:
        stack = build(BASE)
        self.assertEqual(stack.pop(), 40)
        assert_intact(self, stack, [10, 20, 30])

    def test_unlinks_the_node_it_returns(self) -> None:
        stack = build(BASE)
        was_top = stack._top
        stack.pop()
        self.assertIsNone(was_top.next, "the popped node still holds the stack")

    def test_drains_in_reverse(self) -> None:
        stack = build(BASE)
        popped = [stack.pop() for _ in range(len(BASE))]
        self.assertEqual(popped, BASE[::-1], "a stack is not last-in-first-out")
        assert_intact(self, stack, [])

    def test_the_only_value(self) -> None:
        stack = build([10])
        self.assertEqual(stack.pop(), 10)
        assert_intact(self, stack, [])

    def test_two_values_leaves_the_bottom_on_top(self) -> None:
        stack = build([10, 20])
        self.assertEqual(stack.pop(), 20)
        assert_intact(self, stack, [10])

    def test_returns_a_stored_none(self) -> None:
        stack = build([10, None])
        self.assertIsNone(stack.pop())
        assert_intact(self, stack, [10])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            LinkedStack().pop()

    def test_raising_changes_nothing(self) -> None:
        stack = LinkedStack()
        with self.assertRaises(IndexError):
            stack.pop()
        assert_intact(self, stack, [])

    def test_draining_then_refilling(self) -> None:
        stack = build(BASE)
        for _ in range(len(BASE)):
            stack.pop()
        stack._top = Node(99)
        stack._size = 1
        assert_intact(self, stack, [99])


class TestPeek(unittest.TestCase):

    def test_returns_the_top(self) -> None:
        self.assertEqual(build(BASE).peek(), 40)

    def test_leaves_the_stack_alone(self) -> None:
        stack = build(BASE)
        stack.peek()
        assert_intact(self, stack, BASE)

    def test_repeats(self) -> None:
        stack = build(BASE)
        self.assertEqual([stack.peek(), stack.peek()], [40, 40])

    def test_the_only_value(self) -> None:
        self.assertEqual(build([10]).peek(), 10)

    def test_returns_a_stored_none(self) -> None:
        self.assertIsNone(build([10, None]).peek())

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            LinkedStack().peek()


class TestIsEmpty(unittest.TestCase):

    def test_true_for_a_new_stack(self) -> None:
        self.assertTrue(LinkedStack().is_empty())

    def test_false_when_it_holds_a_value(self) -> None:
        self.assertFalse(build([10]).is_empty())

    def test_a_stored_none_is_still_a_value(self) -> None:
        self.assertFalse(build([None]).is_empty())

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        stack.is_empty()
        assert_intact(self, stack, BASE)


class TestContains(unittest.TestCase):

    def test_finds_the_top(self) -> None:
        self.assertTrue(build(BASE).contains(40))

    def test_finds_the_bottom(self) -> None:
        self.assertTrue(build(BASE).contains(10))

    def test_finds_the_middle(self) -> None:
        self.assertTrue(build(BASE).contains(20))

    def test_rejects_an_absent_value(self) -> None:
        self.assertFalse(build(BASE).contains(99))

    def test_empty_stack(self) -> None:
        self.assertFalse(LinkedStack().contains(10))

    def test_finds_a_duplicate(self) -> None:
        self.assertTrue(build([7, 7]).contains(7))

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        stack.contains(20)
        assert_intact(self, stack, BASE)


class TestToList(unittest.TestCase):

    def test_returns_the_values_top_first(self) -> None:
        self.assertEqual(build(BASE).to_list(), BASE[::-1])

    def test_one_value(self) -> None:
        self.assertEqual(build([10]).to_list(), [10])

    def test_empty_stack(self) -> None:
        self.assertEqual(LinkedStack().to_list(), [])

    def test_keeps_duplicates(self) -> None:
        self.assertEqual(build([7, 7]).to_list(), [7, 7])

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        stack.to_list()
        assert_intact(self, stack, BASE)


class TestPrintStack(unittest.TestCase):
    """Asserts the values print from top to bottom, not the layout."""

    def test_prints_top_to_bottom(self) -> None:
        output = printed(build(BASE).print_stack)
        assert_printed_in_order(self, output, BASE[::-1])

    def test_one_value(self) -> None:
        self.assertIn("10", printed(build([10]).print_stack))

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        printed(stack.print_stack)
        assert_intact(self, stack, BASE)


class TestClear(unittest.TestCase):

    def test_drops_every_value(self) -> None:
        stack = build(BASE)
        stack.clear()
        assert_intact(self, stack, [])

    def test_the_stack_is_reusable(self) -> None:
        stack = build(BASE)
        stack.clear()
        stack._top = Node(99)
        stack._size = 1
        assert_intact(self, stack, [99])

    def test_on_an_empty_stack(self) -> None:
        stack = LinkedStack()
        stack.clear()
        assert_intact(self, stack, [])

    def test_on_one_value(self) -> None:
        stack = build([10])
        stack.clear()
        assert_intact(self, stack, [])


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it.
    Chains here always terminate - nothing in this class builds a cycle."""

    def test_empty_stack(self) -> None:
        self.assertTrue(LinkedStack().is_valid())

    def test_one_value(self) -> None:
        self.assertTrue(build([10]).is_valid())

    def test_populated_stack(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_size_too_high(self) -> None:
        stack = build(BASE)
        stack._size += 1
        self.assertFalse(stack.is_valid())

    def test_size_too_low(self) -> None:
        stack = build(BASE)
        stack._size -= 1
        self.assertFalse(stack.is_valid())

    def test_top_set_without_a_size(self) -> None:
        stack = LinkedStack()
        stack._top = Node(10)
        self.assertFalse(stack.is_valid())

    def test_size_set_without_a_top(self) -> None:
        stack = LinkedStack()
        stack._size = 1
        self.assertFalse(stack.is_valid())


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _size directly, but __contains__ delegates to
    contains(), so it stays red until contains() is written."""

    def test_len(self) -> None:
        self.assertEqual(len(build(BASE)), 4)
        self.assertEqual(len(LinkedStack()), 0)

    def test_in_operator(self) -> None:
        stack = build(BASE)
        self.assertIn(30, stack)
        self.assertNotIn(99, stack)


class TestIteration(unittest.TestCase):

    def test_yields_top_first(self) -> None:
        self.assertEqual(list(build(BASE)), BASE[::-1])

    def test_yields_nothing_when_empty(self) -> None:
        self.assertEqual(list(LinkedStack()), [])

    def test_restarts_on_each_pass(self) -> None:
        stack = build(BASE)
        self.assertEqual(list(stack), list(stack), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        list(stack)
        assert_intact(self, stack, BASE)


class TestRepr(unittest.TestCase):

    def test_top_first(self) -> None:
        self.assertEqual(repr(build([1, 2, 3])), "LinkedStack([3, 2, 1])")

    def test_empty_stack(self) -> None:
        self.assertEqual(repr(LinkedStack()), "LinkedStack([])")

    def test_quotes_strings(self) -> None:
        self.assertEqual(repr(build(["a"])), "LinkedStack(['a'])")


class TestNode(unittest.TestCase):

    def test_starts_unlinked(self) -> None:
        node = Node(10)
        self.assertEqual(node.value, 10)
        self.assertIsNone(node.next)

    def test_repr(self) -> None:
        self.assertEqual(repr(Node(10)), "Node(10)")
        self.assertEqual(repr(Node("a")), "Node('a')")


if __name__ == "__main__":
    unittest.main(verbosity=2)
