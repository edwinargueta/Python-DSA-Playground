"""Behavior tests for MinStack, the minimum-tracking stack.

Run from inside this directory:

    python3 -m unittest test_min_stack -v
    python3 test_min_stack.py            # equivalent

Each test class depends only on the method it names, with one exception,
and it is the reason to write push() first. A fixture cannot wire _mins by
hand, because how you fill it is yours to choose, so build() pushes - the
same trade the BST suite makes when its build() calls insert(). Nothing
else is borrowed: contents() reads _values directly, so a test for pop()
never calls to_list() or peek() to see what happened.

assert_intact() re-checks the auxiliary stack after every single mutation:
the top of _mins has to be the smallest value still held. That is what
catches the classic bug in this structure, where equal minima are pushed
with < instead of <=, one of the two is popped, and the stack forgets that
the other is still there. Its length is never asserted - push every value
onto _mins or only the new minima, both pass.

Fixtures and assertions take their values bottom first, the order you would
push them, so build([10, 20, 30]) has 30 on top. to_list() and iteration
run the other way, top first, and those tests say so explicitly.

The deliberate couplings: __len__ reads _values and __contains__ delegates
to contains(), so it stays red until contains() is written. The print tests
assert that the values appear in top-to-bottom order, not the layout or
separator you print them with.
"""

from __future__ import annotations

import io
import random
import unittest
from contextlib import redirect_stdout

from min_stack import MinStack

# The stack used throughout: 10 at the bottom, 40 on top, 10 the minimum.
BASE = [10, 20, 30, 40]


def build(values) -> MinStack:
    """Return a stack holding values, bottom first, by pushing each one.

    The one fixture in this repo's suites that calls the class it tests.
    _mins has no fixed shape, so there is nothing for a fixture to wire by
    hand - see the module docstring.
    """
    stack = MinStack()
    for value in values:
        stack.push(value)
    return stack


def contents(stack) -> list:
    """Read the stored values straight off _values, bottom first."""
    return list(stack._values)


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

    The last two checks are the ones that matter: _mins has to empty when
    _values does, and its top has to be the smallest value still held. Its
    length is deliberately not checked, so either way of filling it passes.
    """
    expected = list(expected)
    case.assertEqual(contents(stack), expected, "the stored values are wrong")
    case.assertEqual(len(stack), len(expected), "the size is out of step")
    case.assertEqual(
        bool(stack._mins),
        bool(expected),
        "_mins and _values disagree about whether the stack is empty",
    )
    if expected:
        case.assertEqual(
            stack._mins[-1],
            min(expected),
            f"the top of _mins is not the smallest value in {expected}",
        )


class TestEmptyStack(unittest.TestCase):
    """An empty stack is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.stack = MinStack()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.stack), 0)

    def test_both_stacks_start_empty(self) -> None:
        self.assertEqual(self.stack._values, [])
        self.assertEqual(self.stack._mins, [])

    def test_push_stores_the_first_value(self) -> None:
        self.stack.push(10)
        assert_intact(self, self.stack, [10])

    def test_pop_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.stack.pop()

    def test_peek_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.stack.peek()

    def test_min_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.stack.min()

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
        self.assertEqual(repr(self.stack), "MinStack([])")


class TestPush(unittest.TestCase):

    def test_stores_the_first_value(self) -> None:
        stack = MinStack()
        stack.push(10)
        assert_intact(self, stack, [10])

    def test_stacks_values_in_order(self) -> None:
        stack = MinStack()
        for value in BASE:
            stack.push(value)
        assert_intact(self, stack, BASE)

    def test_a_descending_run(self) -> None:
        stack = MinStack()
        for value in [40, 30, 20, 10]:
            stack.push(value)
        assert_intact(self, stack, [40, 30, 20, 10])

    def test_keeps_duplicates(self) -> None:
        stack = MinStack()
        for value in [7, 7, 7]:
            stack.push(value)
        assert_intact(self, stack, [7, 7, 7])

    def test_duplicate_minima(self) -> None:
        stack = MinStack()
        for value in [2, 1, 3, 1]:
            stack.push(value)
        assert_intact(self, stack, [2, 1, 3, 1])

    def test_size_tracks_every_push(self) -> None:
        stack = MinStack()
        for i, value in enumerate(BASE, start=1):
            stack.push(value)
            self.assertEqual(len(stack), i, f"size wrong after {i} pushes")

    def test_negative_values(self) -> None:
        stack = MinStack()
        for value in [0, -5, 3, -2]:
            stack.push(value)
        assert_intact(self, stack, [0, -5, 3, -2])

    def test_many_values(self) -> None:
        stack = MinStack()
        values = [random.randint(-50, 50) for _ in range(200)]
        for value in values:
            stack.push(value)
            assert_intact(self, stack, values[: len(stack)])


class TestPop(unittest.TestCase):

    def test_returns_the_top(self) -> None:
        stack = build(BASE)
        self.assertEqual(stack.pop(), 40)
        assert_intact(self, stack, [10, 20, 30])

    def test_drains_in_reverse(self) -> None:
        stack = build(BASE)
        popped = [stack.pop() for _ in range(len(BASE))]
        self.assertEqual(popped, BASE[::-1], "a stack is not last-in-first-out")
        assert_intact(self, stack, [])

    def test_the_only_value(self) -> None:
        stack = build([10])
        self.assertEqual(stack.pop(), 10)
        assert_intact(self, stack, [])

    def test_popping_the_minimum_restores_the_one_before_it(self) -> None:
        stack = build([5, 2, 8])
        stack.pop()
        self.assertEqual(stack.pop(), 2, "the minimum was not on top")
        assert_intact(self, stack, [5])

    def test_popping_one_of_two_equal_minima(self) -> None:
        stack = build([3, 1, 1])
        self.assertEqual(stack.pop(), 1)
        assert_intact(self, stack, [3, 1])

    def test_popping_both_equal_minima(self) -> None:
        stack = build([3, 1, 1])
        stack.pop()
        stack.pop()
        assert_intact(self, stack, [3])

    def test_a_descending_run_pops_clean(self) -> None:
        values = [40, 30, 20, 10]
        stack = build(values)
        for i in range(len(values)):
            stack.pop()
            assert_intact(self, stack, values[: len(values) - i - 1])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            MinStack().pop()

    def test_raising_changes_nothing(self) -> None:
        stack = MinStack()
        with self.assertRaises(IndexError):
            stack.pop()
        assert_intact(self, stack, [])

    def test_draining_then_refilling(self) -> None:
        stack = build(BASE)
        for _ in range(len(BASE)):
            stack.pop()
        stack.push(99)
        assert_intact(self, stack, [99])

    def test_random_churn(self) -> None:
        stack, model = MinStack(), []
        for _ in range(300):
            if model and random.random() < 0.4:
                self.assertEqual(stack.pop(), model.pop(), "wrong value popped")
            else:
                value = random.randint(-20, 20)
                stack.push(value)
                model.append(value)
            assert_intact(self, stack, model)


class TestPeek(unittest.TestCase):

    def test_returns_the_top(self) -> None:
        self.assertEqual(build(BASE).peek(), 40)

    def test_is_not_the_minimum(self) -> None:
        self.assertEqual(build([5, 2, 8]).peek(), 8, "peek() returned the minimum")

    def test_leaves_the_stack_alone(self) -> None:
        stack = build(BASE)
        stack.peek()
        assert_intact(self, stack, BASE)

    def test_repeats(self) -> None:
        stack = build(BASE)
        self.assertEqual([stack.peek(), stack.peek()], [40, 40])

    def test_the_only_value(self) -> None:
        self.assertEqual(build([10]).peek(), 10)

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            MinStack().peek()


class TestMin(unittest.TestCase):
    """Half of min()'s contract is what it says after a pop, and there is
    no way to set that up without popping - _mins has no fixed shape for a
    fixture to wire. So this class drives pop() too, and stays red until
    both are written."""

    def test_the_only_value(self) -> None:
        self.assertEqual(build([10]).min(), 10)

    def test_the_smallest_of_several(self) -> None:
        self.assertEqual(build(BASE).min(), 10)

    def test_is_not_the_top(self) -> None:
        self.assertEqual(build([5, 2, 8]).min(), 2, "min() returned the top")

    def test_is_not_the_bottom(self) -> None:
        self.assertEqual(build([5, 2, 8]).min(), 2, "min() returned the bottom")

    def test_updates_as_smaller_values_arrive(self) -> None:
        stack, expected = MinStack(), []
        for value in [40, 30, 20, 10]:
            stack.push(value)
            expected.append(value)
            self.assertEqual(stack.min(), min(expected), f"after pushing {value}")

    def test_holds_when_larger_values_arrive(self) -> None:
        stack = build([10])
        for value in [20, 30, 40]:
            stack.push(value)
            self.assertEqual(stack.min(), 10, f"after pushing {value}")

    def test_reverts_when_the_minimum_is_popped(self) -> None:
        stack = build([5, 2])
        stack.pop()
        self.assertEqual(stack.min(), 5, "the old minimum did not come back")

    def test_survives_popping_one_of_two_equal_minima(self) -> None:
        stack = build([3, 1, 1])
        stack.pop()
        self.assertEqual(stack.min(), 1, "the second 1 is still in the stack")

    def test_negative_values(self) -> None:
        self.assertEqual(build([0, -5, 3, -2]).min(), -5)

    def test_strings(self) -> None:
        self.assertEqual(build(["pear", "apple", "fig"]).min(), "apple")

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        stack.min()
        assert_intact(self, stack, BASE)

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            MinStack().min()

    def test_raises_again_once_drained(self) -> None:
        stack = build([10])
        stack.pop()
        with self.assertRaises(IndexError):
            stack.min()

    def test_tracks_a_random_sequence(self) -> None:
        stack, model = MinStack(), []
        for _ in range(200):
            if model and random.random() < 0.4:
                stack.pop()
                model.pop()
            else:
                value = random.randint(-20, 20)
                stack.push(value)
                model.append(value)
            if model:
                self.assertEqual(stack.min(), min(model), f"holding {model}")


class TestIsEmpty(unittest.TestCase):

    def test_true_for_a_new_stack(self) -> None:
        self.assertTrue(MinStack().is_empty())

    def test_false_when_it_holds_a_value(self) -> None:
        self.assertFalse(build([10]).is_empty())

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
        self.assertFalse(MinStack().contains(10))

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
        self.assertEqual(MinStack().to_list(), [])

    def test_holds_values_not_minima(self) -> None:
        self.assertEqual(build([5, 2, 8]).to_list(), [8, 2, 5], "it returned _mins")

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

    def test_empties_the_auxiliary_stack_too(self) -> None:
        stack = build(BASE)
        stack.clear()
        self.assertEqual(stack._mins, [], "_mins survived the clear")

    def test_the_stack_is_reusable(self) -> None:
        stack = build(BASE)
        stack.clear()
        stack.push(99)
        assert_intact(self, stack, [99])

    def test_on_an_empty_stack(self) -> None:
        stack = MinStack()
        stack.clear()
        assert_intact(self, stack, [])


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it."""

    def test_empty_stack(self) -> None:
        self.assertTrue(MinStack().is_valid())

    def test_one_value(self) -> None:
        self.assertTrue(build([10]).is_valid())

    def test_populated_stack(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_duplicate_minima(self) -> None:
        self.assertTrue(build([3, 1, 1]).is_valid())

    def test_values_without_mins(self) -> None:
        stack = build(BASE)
        stack._mins = []
        self.assertFalse(stack.is_valid())

    def test_mins_without_values(self) -> None:
        stack = MinStack()
        stack._mins = [10]
        self.assertFalse(stack.is_valid())

    def test_the_wrong_minimum_on_top(self) -> None:
        stack = build([5, 2, 8])
        stack._mins[-1] = 99
        self.assertFalse(stack.is_valid(), "99 is not the smallest value held")

    def test_a_minimum_that_is_not_in_the_stack(self) -> None:
        stack = build([5, 2, 8])
        stack._values.remove(2)
        self.assertFalse(stack.is_valid(), "_mins remembers a value that is gone")


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _values directly, but __contains__ delegates to
    contains(), so it stays red until contains() is written."""

    def test_len(self) -> None:
        self.assertEqual(len(build(BASE)), 4)
        self.assertEqual(len(MinStack()), 0)

    def test_len_is_not_the_auxiliary_stack(self) -> None:
        self.assertEqual(len(build([5, 6, 7])), 3, "len() is reading _mins")

    def test_in_operator(self) -> None:
        stack = build(BASE)
        self.assertIn(30, stack)
        self.assertNotIn(99, stack)


class TestIteration(unittest.TestCase):

    def test_yields_top_first(self) -> None:
        self.assertEqual(list(build(BASE)), BASE[::-1])

    def test_yields_nothing_when_empty(self) -> None:
        self.assertEqual(list(MinStack()), [])

    def test_yields_values_not_minima(self) -> None:
        self.assertEqual(list(build([5, 2, 8])), [8, 2, 5], "it iterated _mins")

    def test_restarts_on_each_pass(self) -> None:
        stack = build(BASE)
        self.assertEqual(list(stack), list(stack), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        stack = build(BASE)
        list(stack)
        assert_intact(self, stack, BASE)


class TestRepr(unittest.TestCase):

    def test_top_first(self) -> None:
        self.assertEqual(repr(build([1, 2, 3])), "MinStack([3, 2, 1])")

    def test_empty_stack(self) -> None:
        self.assertEqual(repr(MinStack()), "MinStack([])")

    def test_quotes_strings(self) -> None:
        self.assertEqual(repr(build(["a"])), "MinStack(['a'])")


if __name__ == "__main__":
    unittest.main(verbosity=2)
