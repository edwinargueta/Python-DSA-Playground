"""Behavior tests for ArrayStack, the array-backed stack.

Run from inside this directory:

    python3 -m unittest test_array_stack -v
    python3 test_array_stack.py            # equivalent

Each test class depends only on the method it names, so you can implement
the methods in any order and a green class always means that method is done.

build() writes into the backing array by hand and sets _count itself, so
every fixture exists before a single method is written, and a broken push()
cannot fail an unrelated class. contents() and spare() then read the array
directly, which is why a test for pop() never calls to_list() to see what
happened.

assert_intact() checks the spare slots on every mutation, not just the
values. An array-backed stack that pops by decrementing _count alone still
answers every question correctly while quietly holding a reference to each
value it claims to have dropped - invariant 4 exists to catch that, and
nothing but a look at the array can see it.

The deliberate couplings: __len__ reads _count and __contains__ delegates
to contains(), so those stay red until contains() is written. The print
tests assert that the values appear in top-to-bottom order, not the layout
or separator you print them with.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from array_stack import ArrayStack

# The stack used throughout: 10 at the bottom, 40 on top.
BASE = [10, 20, 30, 40]


def build(values, capacity: int = 8) -> ArrayStack:
    """Return a stack holding values, bottom first, wiring the array by hand.

    Deliberately avoids push() so that fixtures work before any method is
    implemented. The array is sized to hold the values whatever capacity
    is asked for, and it never grows itself.
    """
    values = list(values)
    stack = ArrayStack(max(capacity, len(values), 1))
    for index, value in enumerate(values):
        stack._items[index] = value
    stack._count = len(values)
    return stack


def contents(stack) -> list:
    """Collect the live values from the array, bottom first."""
    return [stack._items[i] for i in range(stack._count)]


def spare(stack) -> list:
    """Collect the slots past the top, which should all be None."""
    return [stack._items[i] for i in range(stack._count, len(stack._items))]


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

    Checks the values, _count, the capacity, and that every slot above the
    top has been cleared - a pop that only moves _count leaves the value
    sitting in the array, and this is what notices.
    """
    expected = list(expected)
    case.assertEqual(contents(stack), expected, "the stored values are wrong")
    case.assertEqual(len(stack), len(expected), "_count is out of step")
    case.assertGreaterEqual(len(stack._items), 1, "the backing array is gone")
    case.assertLessEqual(
        stack._count, len(stack._items), "_count is past the end of the array"
    )
    leftovers = [value for value in spare(stack) if value is not None]
    case.assertEqual(leftovers, [], "a slot above the top still holds a value")


class TestEmptyStack(unittest.TestCase):
    """An empty stack is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.stack = ArrayStack()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.stack), 0)

    def test_the_array_is_all_none(self) -> None:
        self.assertEqual(self.stack._items, [None] * 8)

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

    def test_capacity_is_unchanged(self) -> None:
        self.assertEqual(self.stack.capacity(), 8)

    def test_grow_keeps_it_empty(self) -> None:
        self.stack._grow(16)
        assert_intact(self, self.stack, [])
        self.assertEqual(len(self.stack._items), 16)

    def test_is_valid(self) -> None:
        self.assertTrue(self.stack.is_valid())

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.stack), [])

    def test_repr(self) -> None:
        self.assertEqual(repr(self.stack), "ArrayStack([])")


class TestPush(unittest.TestCase):

    def test_stores_the_first_value(self) -> None:
        stack = ArrayStack()
        stack.push(10)
        assert_intact(self, stack, [10])

    def test_stacks_values_in_order(self) -> None:
        stack = ArrayStack()
        for value in BASE:
            stack.push(value)
        assert_intact(self, stack, BASE)

    def test_the_last_value_is_on_top(self) -> None:
        stack = ArrayStack()
        for value in BASE:
            stack.push(value)
        self.assertEqual(stack._items[stack._count - 1], 40, "40 is not on top")

    def test_keeps_duplicates(self) -> None:
        stack = ArrayStack()
        for value in [7, 7, 7]:
            stack.push(value)
        assert_intact(self, stack, [7, 7, 7])

    def test_stores_none_as_a_value(self) -> None:
        stack = ArrayStack()
        stack.push(None)
        self.assertEqual(len(stack), 1, "None was not counted as a value")

    def test_size_tracks_every_push(self) -> None:
        stack = ArrayStack()
        for i, value in enumerate(BASE, start=1):
            stack.push(value)
            self.assertEqual(len(stack), i, f"size wrong after {i} pushes")

    def test_onto_a_built_stack(self) -> None:
        stack = build(BASE)
        stack.push(50)
        assert_intact(self, stack, BASE + [50])


class TestGrowth(unittest.TestCase):
    """push() resizes the array itself, so this class needs push() and
    whatever it grows with - the capacity it grows to is yours."""

    def test_grows_once_the_array_is_full(self) -> None:
        stack = ArrayStack(capacity=4)
        for value in [1, 2, 3, 4, 5]:
            stack.push(value)
        self.assertGreater(len(stack._items), 4, "the array never grew")

    def test_keeps_every_value_across_the_growth(self) -> None:
        stack = ArrayStack(capacity=2)
        expected = list(range(20))
        for value in expected:
            stack.push(value)
        assert_intact(self, stack, expected)

    def test_does_not_grow_while_there_is_room(self) -> None:
        stack = ArrayStack(capacity=8)
        for value in BASE:
            stack.push(value)
        self.assertEqual(len(stack._items), 8, "the array grew before it had to")

    def test_the_new_slots_are_empty(self) -> None:
        stack = ArrayStack(capacity=2)
        for value in [1, 2, 3]:
            stack.push(value)
        self.assertEqual(
            spare(stack), [None] * (len(stack._items) - 3), "the new slots hold junk"
        )

    def test_capacity_never_falls_behind_the_count(self) -> None:
        stack = ArrayStack(capacity=1)
        for i in range(50):
            stack.push(i)
            self.assertLessEqual(
                len(stack), len(stack._items), f"overflowed after {i + 1} pushes"
            )


class TestPop(unittest.TestCase):

    def test_returns_the_top(self) -> None:
        stack = build(BASE)
        self.assertEqual(stack.pop(), 40)
        assert_intact(self, stack, [10, 20, 30])

    def test_clears_the_slot_it_vacates(self) -> None:
        stack = build(BASE)
        stack.pop()
        self.assertIsNone(
            stack._items[stack._count], "the popped value is still in the array"
        )

    def test_drains_in_reverse(self) -> None:
        stack = build(BASE)
        popped = [stack.pop() for _ in range(len(BASE))]
        self.assertEqual(popped, BASE[::-1], "a stack is not last-in-first-out")
        assert_intact(self, stack, [])

    def test_the_only_value(self) -> None:
        stack = build([10])
        self.assertEqual(stack.pop(), 10)
        assert_intact(self, stack, [])

    def test_returns_a_stored_none(self) -> None:
        stack = build([10, None])
        self.assertIsNone(stack.pop())
        assert_intact(self, stack, [10])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            ArrayStack().pop()

    def test_raising_changes_nothing(self) -> None:
        stack = ArrayStack()
        with self.assertRaises(IndexError):
            stack.pop()
        assert_intact(self, stack, [])

    def test_does_not_shrink_the_capacity(self) -> None:
        stack = build(BASE, capacity=16)
        for _ in range(len(BASE)):
            stack.pop()
        self.assertEqual(len(stack._items), 16, "popping shrank the array")

    def test_pop_then_push_reuses_the_slot(self) -> None:
        stack = build(BASE)
        stack.pop()
        stack._items[stack._count] = 99
        stack._count += 1
        assert_intact(self, stack, [10, 20, 30, 99])


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
            ArrayStack().peek()


class TestIsEmpty(unittest.TestCase):

    def test_true_for_a_new_stack(self) -> None:
        self.assertTrue(ArrayStack().is_empty())

    def test_false_when_it_holds_a_value(self) -> None:
        self.assertFalse(build([10]).is_empty())

    def test_true_for_a_stack_holding_only_none(self) -> None:
        self.assertFalse(
            build([None]).is_empty(), "a stored None is still a value"
        )

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

    def test_ignores_the_slots_above_the_top(self) -> None:
        stack = build(BASE)
        stack._items[stack._count] = 99
        self.assertFalse(stack.contains(99), "it searched past the top of the stack")

    def test_empty_stack(self) -> None:
        self.assertFalse(ArrayStack().contains(10))

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
        self.assertEqual(ArrayStack().to_list(), [])

    def test_stops_at_the_top(self) -> None:
        stack = build(BASE)
        stack._items[stack._count] = 99
        self.assertEqual(stack.to_list(), BASE[::-1], "it read past the top")

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

    def test_wipes_the_array(self) -> None:
        stack = build(BASE)
        stack.clear()
        self.assertEqual(
            stack._items, [None] * len(stack._items), "a cleared value is still held"
        )

    def test_keeps_the_capacity(self) -> None:
        stack = build(BASE, capacity=16)
        stack.clear()
        self.assertEqual(len(stack._items), 16, "clear() changed the capacity")

    def test_the_stack_is_reusable(self) -> None:
        stack = build(BASE)
        stack.clear()
        stack._items[0] = 99
        stack._count = 1
        assert_intact(self, stack, [99])

    def test_on_an_empty_stack(self) -> None:
        stack = ArrayStack()
        stack.clear()
        assert_intact(self, stack, [])


class TestCapacity(unittest.TestCase):

    def test_default(self) -> None:
        self.assertEqual(ArrayStack().capacity(), 8)

    def test_custom(self) -> None:
        self.assertEqual(ArrayStack(32).capacity(), 32)

    def test_tracks_the_array(self) -> None:
        stack = ArrayStack(8)
        stack._items = [None] * 64
        self.assertEqual(stack.capacity(), 64, "capacity() is not reading the array")

    def test_is_not_the_size(self) -> None:
        self.assertEqual(build(BASE, capacity=16).capacity(), 16)


class TestGrow(unittest.TestCase):

    def test_changes_the_capacity(self) -> None:
        stack = build(BASE)
        stack._grow(32)
        self.assertEqual(len(stack._items), 32)

    def test_keeps_every_value(self) -> None:
        stack = build(BASE)
        stack._grow(32)
        assert_intact(self, stack, BASE)

    def test_the_new_slots_are_empty(self) -> None:
        stack = build(BASE)
        stack._grow(32)
        self.assertEqual(spare(stack), [None] * 28, "the new slots hold junk")

    def test_size_is_unchanged(self) -> None:
        stack = build(BASE)
        stack._grow(32)
        self.assertEqual(len(stack), 4, "_count drifted during the rebuild")

    def test_on_an_empty_stack(self) -> None:
        stack = ArrayStack(4)
        stack._grow(8)
        assert_intact(self, stack, [])
        self.assertEqual(len(stack._items), 8)

    def test_to_exactly_the_count(self) -> None:
        stack = build(BASE, capacity=8)
        stack._grow(4)
        assert_intact(self, stack, BASE)


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it."""

    def test_empty_stack(self) -> None:
        self.assertTrue(ArrayStack().is_valid())

    def test_populated_stack(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_full_array(self) -> None:
        self.assertTrue(build(BASE, capacity=4).is_valid())

    def test_count_past_the_array(self) -> None:
        stack = build(BASE, capacity=4)
        stack._count = 5
        self.assertFalse(stack.is_valid())

    def test_negative_count(self) -> None:
        stack = build(BASE)
        stack._count = -1
        self.assertFalse(stack.is_valid())

    def test_value_left_above_the_top(self) -> None:
        stack = build(BASE)
        stack._items[stack._count] = 99
        self.assertFalse(stack.is_valid(), "a dropped value is still in the array")

    def test_no_array_at_all(self) -> None:
        stack = ArrayStack()
        stack._items = []
        self.assertFalse(stack.is_valid())


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _count directly, but __contains__ delegates to
    contains(), so it stays red until contains() is written."""

    def test_len(self) -> None:
        self.assertEqual(len(build(BASE)), 4)
        self.assertEqual(len(ArrayStack()), 0)

    def test_len_is_not_the_capacity(self) -> None:
        self.assertEqual(len(build(BASE, capacity=64)), 4)

    def test_in_operator(self) -> None:
        stack = build(BASE)
        self.assertIn(30, stack)
        self.assertNotIn(99, stack)


class TestIteration(unittest.TestCase):

    def test_yields_top_first(self) -> None:
        self.assertEqual(list(build(BASE)), BASE[::-1])

    def test_yields_nothing_when_empty(self) -> None:
        self.assertEqual(list(ArrayStack()), [])

    def test_stops_at_the_top(self) -> None:
        stack = build(BASE)
        stack._items[stack._count] = 99
        self.assertEqual(list(stack), BASE[::-1], "it iterated past the top")

    def test_restarts_on_each_pass(self) -> None:
        stack = build(BASE)
        self.assertEqual(list(stack), list(stack), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        list(stack)
        assert_intact(self, stack, BASE)


class TestRepr(unittest.TestCase):

    def test_top_first(self) -> None:
        self.assertEqual(repr(build([1, 2, 3])), "ArrayStack([3, 2, 1])")

    def test_empty_stack(self) -> None:
        self.assertEqual(repr(ArrayStack()), "ArrayStack([])")

    def test_quotes_strings(self) -> None:
        self.assertEqual(repr(build(["a"])), "ArrayStack(['a'])")


if __name__ == "__main__":
    unittest.main(verbosity=2)
