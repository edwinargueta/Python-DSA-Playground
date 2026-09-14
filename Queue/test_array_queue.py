"""Behavior tests for ArrayQueue, the ring-buffer queue.

Run from inside this directory:

    python3 -m unittest test_array_queue -v
    python3 test_array_queue.py            # equivalent

Each test class depends only on the method it names, so you can implement
the methods in any order and a green class always means that method is done.

build() writes into the backing array by hand and sets _head and _count
itself, so every fixture exists before a single method is written and a
broken enqueue() cannot fail an unrelated class. It also takes a head
argument, which is what makes this suite worth reading: build(BASE,
capacity=4, head=3) hands you an already-wrapped queue, so the cases that
only show up after the values have run off the end of the array can be
tested first, without driving the queue there through enqueue and dequeue.

assert_intact() checks the slots outside the live window on every mutation,
not just the values. A queue that dequeues by moving _head alone answers
every question correctly while quietly holding a reference to each value it
claims to have handed out - invariant 5 exists to catch that, and nothing
but a look at the array can see it.

The deliberate couplings: __len__ reads _count and __contains__ delegates
to contains(), so it stays red until contains() is written. TestRoundTrip
drives enqueue() and dequeue() together and needs both. The print tests
assert that the values appear in front-to-back order, not the layout or
separator you print them with.
"""

from __future__ import annotations

import io
import random
import unittest
from contextlib import redirect_stdout

from array_queue import ArrayQueue

# The queue used throughout: 10 at the front, 40 at the back.
BASE = [10, 20, 30, 40]


def build(values, capacity: int = 8, head: int = 0) -> ArrayQueue:
    """Return a queue holding values, front first, wiring the array by hand.

    Deliberately avoids enqueue() so that fixtures work before any method
    is implemented. head is where the front value goes, so passing one near
    the end of the array gives you a queue that has already wrapped.
    """
    values = list(values)
    capacity = max(capacity, len(values), 1)
    queue = ArrayQueue(capacity)
    queue._head = head % capacity
    for i, value in enumerate(values):
        queue._items[(queue._head + i) % capacity] = value
    queue._count = len(values)
    return queue


def contents(queue) -> list:
    """Collect the live values from the array, front first."""
    size = len(queue._items)
    return [queue._items[(queue._head + i) % size] for i in range(queue._count)]


def spare(queue) -> list:
    """Collect the slots outside the live window, which should all be None."""
    size = len(queue._items)
    live = {(queue._head + i) % size for i in range(queue._count)}
    return [queue._items[i] for i in range(size) if i not in live]


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


def assert_intact(case, queue, expected) -> None:
    """Assert the queue holds exactly expected, front first, and is intact.

    Checks the values, _count, that _head is still a usable index, and that
    every slot outside the live window has been cleared - a dequeue that
    only moves _head leaves the value sitting in the array, and this is
    what notices.
    """
    expected = list(expected)
    size = len(queue._items)
    case.assertGreaterEqual(size, 1, "the backing array is gone")
    case.assertGreaterEqual(queue._head, 0, "_head fell off the front of the array")
    case.assertLess(queue._head, size, "_head fell off the end of the array")
    case.assertLessEqual(queue._count, size, "_count is larger than the array")
    case.assertEqual(contents(queue), expected, "the stored values are wrong")
    case.assertEqual(len(queue), len(expected), "_count is out of step")
    leftovers = [value for value in spare(queue) if value is not None]
    case.assertEqual(leftovers, [], "a slot outside the queue still holds a value")


class TestEmptyQueue(unittest.TestCase):
    """An empty queue is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.queue = ArrayQueue()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.queue), 0)

    def test_the_array_is_all_none(self) -> None:
        self.assertEqual(self.queue._items, [None] * 8)

    def test_enqueue_stores_the_first_value(self) -> None:
        self.queue.enqueue(10)
        assert_intact(self, self.queue, [10])

    def test_dequeue_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.queue.dequeue()

    def test_peek_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.queue.peek()

    def test_peek_back_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.queue.peek_back()

    def test_is_empty_is_true(self) -> None:
        self.assertTrue(self.queue.is_empty())

    def test_contains_is_false(self) -> None:
        self.assertFalse(self.queue.contains(10))

    def test_to_list_is_empty(self) -> None:
        self.assertEqual(self.queue.to_list(), [])

    def test_print_queue_prints_no_values(self) -> None:
        self.assertNotIn("10", printed(self.queue.print_queue))

    def test_clear_is_a_no_op(self) -> None:
        self.queue.clear()
        assert_intact(self, self.queue, [])

    def test_capacity_is_unchanged(self) -> None:
        self.assertEqual(self.queue.capacity(), 8)

    def test_grow_keeps_it_empty(self) -> None:
        self.queue._grow(16)
        assert_intact(self, self.queue, [])
        self.assertEqual(len(self.queue._items), 16)

    def test_is_valid(self) -> None:
        self.assertTrue(self.queue.is_valid())

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.queue), [])

    def test_repr(self) -> None:
        self.assertEqual(repr(self.queue), "ArrayQueue([])")


class TestEnqueue(unittest.TestCase):

    def test_stores_the_first_value(self) -> None:
        queue = ArrayQueue()
        queue.enqueue(10)
        assert_intact(self, queue, [10])

    def test_queues_values_in_order(self) -> None:
        queue = ArrayQueue()
        for value in BASE:
            queue.enqueue(value)
        assert_intact(self, queue, BASE)

    def test_adds_at_the_back_not_the_front(self) -> None:
        queue = build([10])
        queue.enqueue(20)
        assert_intact(self, queue, [10, 20])

    def test_keeps_duplicates(self) -> None:
        queue = ArrayQueue()
        for value in [7, 7, 7]:
            queue.enqueue(value)
        assert_intact(self, queue, [7, 7, 7])

    def test_stores_none_as_a_value(self) -> None:
        queue = ArrayQueue()
        queue.enqueue(None)
        self.assertEqual(len(queue), 1, "None was not counted as a value")

    def test_size_tracks_every_enqueue(self) -> None:
        queue = ArrayQueue()
        for i, value in enumerate(BASE, start=1):
            queue.enqueue(value)
            self.assertEqual(len(queue), i, f"size wrong after {i} enqueues")

    def test_wraps_around_the_end_of_the_array(self) -> None:
        queue = build([10, 20], capacity=4, head=3)
        queue.enqueue(30)
        assert_intact(self, queue, [10, 20, 30])

    def test_fills_the_last_free_slot(self) -> None:
        queue = build([10, 20, 30], capacity=4, head=2)
        queue.enqueue(40)
        assert_intact(self, queue, [10, 20, 30, 40])
        self.assertEqual(len(queue._items), 4, "the array grew before it was full")


class TestGrowth(unittest.TestCase):
    """enqueue() resizes the array itself, so this class needs enqueue()
    and whatever it grows with - the capacity it grows to is yours."""

    def test_grows_once_the_array_is_full(self) -> None:
        queue = ArrayQueue(capacity=4)
        for value in [1, 2, 3, 4, 5]:
            queue.enqueue(value)
        self.assertGreater(len(queue._items), 4, "the array never grew")

    def test_keeps_every_value_across_the_growth(self) -> None:
        queue = ArrayQueue(capacity=2)
        expected = list(range(20))
        for value in expected:
            queue.enqueue(value)
        assert_intact(self, queue, expected)

    def test_grows_while_wrapped(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        queue.enqueue(50)
        assert_intact(self, queue, BASE + [50])

    def test_a_wrapped_queue_keeps_its_order(self) -> None:
        queue = build([10, 20, 30], capacity=3, head=2)
        queue.enqueue(40)
        self.assertEqual(
            contents(queue), [10, 20, 30, 40], "the wrap was unrolled out of order"
        )

    def test_does_not_grow_while_there_is_room(self) -> None:
        queue = ArrayQueue(capacity=8)
        for value in BASE:
            queue.enqueue(value)
        self.assertEqual(len(queue._items), 8, "the array grew before it had to")

    def test_the_new_slots_are_empty(self) -> None:
        queue = ArrayQueue(capacity=2)
        for value in [1, 2, 3]:
            queue.enqueue(value)
        self.assertEqual(
            spare(queue), [None] * (len(queue._items) - 3), "the new slots hold junk"
        )

    def test_capacity_never_falls_behind_the_count(self) -> None:
        queue = ArrayQueue(capacity=1)
        for i in range(50):
            queue.enqueue(i)
            self.assertLessEqual(
                len(queue), len(queue._items), f"overflowed after {i + 1} enqueues"
            )


class TestDequeue(unittest.TestCase):

    def test_returns_the_front(self) -> None:
        queue = build(BASE)
        self.assertEqual(queue.dequeue(), 10)
        assert_intact(self, queue, [20, 30, 40])

    def test_clears_the_slot_it_vacates(self) -> None:
        queue = build(BASE)
        was_head = queue._head
        queue.dequeue()
        self.assertIsNone(
            queue._items[was_head], "the dequeued value is still in the array"
        )

    def test_drains_in_order(self) -> None:
        queue = build(BASE)
        served = [queue.dequeue() for _ in range(len(BASE))]
        self.assertEqual(served, BASE, "a queue is not first-in-first-out")
        assert_intact(self, queue, [])

    def test_the_only_value(self) -> None:
        queue = build([10])
        self.assertEqual(queue.dequeue(), 10)
        assert_intact(self, queue, [])

    def test_wraps_around_the_end_of_the_array(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        self.assertEqual(queue.dequeue(), 10)
        assert_intact(self, queue, [20, 30, 40])
        self.assertEqual(queue._head, 0, "_head did not wrap to the start")

    def test_drains_a_wrapped_queue(self) -> None:
        queue = build(BASE, capacity=4, head=2)
        served = [queue.dequeue() for _ in range(len(BASE))]
        self.assertEqual(served, BASE)
        assert_intact(self, queue, [])

    def test_returns_a_stored_none(self) -> None:
        queue = build([None, 20])
        self.assertIsNone(queue.dequeue())
        assert_intact(self, queue, [20])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            ArrayQueue().dequeue()

    def test_raising_changes_nothing(self) -> None:
        queue = ArrayQueue()
        with self.assertRaises(IndexError):
            queue.dequeue()
        assert_intact(self, queue, [])

    def test_does_not_shrink_the_capacity(self) -> None:
        queue = build(BASE, capacity=16)
        for _ in range(len(BASE)):
            queue.dequeue()
        self.assertEqual(len(queue._items), 16, "dequeueing shrank the array")


class TestRoundTrip(unittest.TestCase):
    """Drives enqueue() and dequeue() together, so it needs both. A ring
    buffer only shows what it is made of once the values have lapped the
    array a few times, and no single-method class can get it there."""

    def test_many_laps_around_the_array(self) -> None:
        queue, model = ArrayQueue(capacity=4), []
        for i in range(200):
            queue.enqueue(i)
            model.append(i)
            if random.random() < 0.5:
                self.assertEqual(queue.dequeue(), model.pop(0), "wrong value served")
            assert_intact(self, queue, model)

    def test_drain_and_refill(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        for _ in range(len(BASE)):
            queue.dequeue()
        for value in [50, 60]:
            queue.enqueue(value)
        assert_intact(self, queue, [50, 60])

    def test_the_queue_stays_valid_throughout(self) -> None:
        queue = ArrayQueue(capacity=2)
        for i in range(60):
            queue.enqueue(i)
            if i % 3 == 0:
                queue.dequeue()
            self.assertTrue(queue.is_valid(), f"invalid after {i + 1} enqueues")


class TestPeek(unittest.TestCase):

    def test_returns_the_front(self) -> None:
        self.assertEqual(build(BASE).peek(), 10)

    def test_is_not_the_back(self) -> None:
        self.assertEqual(build(BASE).peek(), 10, "peek() returned the back")

    def test_leaves_the_queue_alone(self) -> None:
        queue = build(BASE)
        queue.peek()
        assert_intact(self, queue, BASE)

    def test_repeats(self) -> None:
        queue = build(BASE)
        self.assertEqual([queue.peek(), queue.peek()], [10, 10])

    def test_on_a_wrapped_queue(self) -> None:
        self.assertEqual(build(BASE, capacity=4, head=3).peek(), 10)

    def test_the_only_value(self) -> None:
        self.assertEqual(build([10]).peek(), 10)

    def test_returns_a_stored_none(self) -> None:
        self.assertIsNone(build([None, 20]).peek())

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            ArrayQueue().peek()


class TestPeekBack(unittest.TestCase):

    def test_returns_the_back(self) -> None:
        self.assertEqual(build(BASE).peek_back(), 40)

    def test_is_not_the_front(self) -> None:
        self.assertEqual(build(BASE).peek_back(), 40, "peek_back() gave the front")

    def test_leaves_the_queue_alone(self) -> None:
        queue = build(BASE)
        queue.peek_back()
        assert_intact(self, queue, BASE)

    def test_on_a_wrapped_queue(self) -> None:
        self.assertEqual(build(BASE, capacity=4, head=3).peek_back(), 40)

    def test_the_only_value_is_both_ends(self) -> None:
        queue = build([10])
        self.assertEqual(queue.peek_back(), 10)

    def test_returns_a_stored_none(self) -> None:
        self.assertIsNone(build([10, None]).peek_back())

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            ArrayQueue().peek_back()


class TestIsEmpty(unittest.TestCase):

    def test_true_for_a_new_queue(self) -> None:
        self.assertTrue(ArrayQueue().is_empty())

    def test_false_when_it_holds_a_value(self) -> None:
        self.assertFalse(build([10]).is_empty())

    def test_a_stored_none_is_still_a_value(self) -> None:
        self.assertFalse(build([None]).is_empty())

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        queue.is_empty()
        assert_intact(self, queue, BASE)


class TestContains(unittest.TestCase):

    def test_finds_the_front(self) -> None:
        self.assertTrue(build(BASE).contains(10))

    def test_finds_the_back(self) -> None:
        self.assertTrue(build(BASE).contains(40))

    def test_finds_the_middle(self) -> None:
        self.assertTrue(build(BASE).contains(20))

    def test_rejects_an_absent_value(self) -> None:
        self.assertFalse(build(BASE).contains(99))

    def test_on_a_wrapped_queue(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        self.assertTrue(queue.contains(30), "it stopped at the end of the array")

    def test_ignores_the_slots_outside_the_queue(self) -> None:
        queue = build(BASE, capacity=8, head=0)
        queue._items[6] = 99
        self.assertFalse(queue.contains(99), "it searched outside the queue")

    def test_empty_queue(self) -> None:
        self.assertFalse(ArrayQueue().contains(10))

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        queue.contains(20)
        assert_intact(self, queue, BASE)


class TestToList(unittest.TestCase):

    def test_returns_the_values_front_first(self) -> None:
        self.assertEqual(build(BASE).to_list(), BASE)

    def test_on_a_wrapped_queue(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        self.assertEqual(queue.to_list(), BASE, "the wrap was read out of order")

    def test_one_value(self) -> None:
        self.assertEqual(build([10]).to_list(), [10])

    def test_empty_queue(self) -> None:
        self.assertEqual(ArrayQueue().to_list(), [])

    def test_stops_at_the_back(self) -> None:
        queue = build(BASE, capacity=8)
        queue._items[6] = 99
        self.assertEqual(queue.to_list(), BASE, "it read past the back")

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        queue.to_list()
        assert_intact(self, queue, BASE)


class TestPrintQueue(unittest.TestCase):
    """Asserts the values print from front to back, not the layout."""

    def test_prints_front_to_back(self) -> None:
        assert_printed_in_order(self, printed(build(BASE).print_queue), BASE)

    def test_on_a_wrapped_queue(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        assert_printed_in_order(self, printed(queue.print_queue), BASE)

    def test_one_value(self) -> None:
        self.assertIn("10", printed(build([10]).print_queue))

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        printed(queue.print_queue)
        assert_intact(self, queue, BASE)


class TestClear(unittest.TestCase):

    def test_drops_every_value(self) -> None:
        queue = build(BASE)
        queue.clear()
        assert_intact(self, queue, [])

    def test_wipes_the_array(self) -> None:
        queue = build(BASE)
        queue.clear()
        self.assertEqual(
            queue._items, [None] * len(queue._items), "a cleared value is still held"
        )

    def test_on_a_wrapped_queue(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        queue.clear()
        assert_intact(self, queue, [])

    def test_keeps_the_capacity(self) -> None:
        queue = build(BASE, capacity=16)
        queue.clear()
        self.assertEqual(len(queue._items), 16, "clear() changed the capacity")

    def test_the_queue_is_reusable(self) -> None:
        queue = build(BASE)
        queue.clear()
        queue._items[queue._head] = 99
        queue._count = 1
        assert_intact(self, queue, [99])

    def test_on_an_empty_queue(self) -> None:
        queue = ArrayQueue()
        queue.clear()
        assert_intact(self, queue, [])


class TestCapacity(unittest.TestCase):

    def test_default(self) -> None:
        self.assertEqual(ArrayQueue().capacity(), 8)

    def test_custom(self) -> None:
        self.assertEqual(ArrayQueue(32).capacity(), 32)

    def test_tracks_the_array(self) -> None:
        queue = ArrayQueue(8)
        queue._items = [None] * 64
        self.assertEqual(queue.capacity(), 64, "capacity() is not reading the array")

    def test_is_not_the_size(self) -> None:
        self.assertEqual(build(BASE, capacity=16).capacity(), 16)


class TestGrow(unittest.TestCase):

    def test_changes_the_capacity(self) -> None:
        queue = build(BASE)
        queue._grow(32)
        self.assertEqual(len(queue._items), 32)

    def test_keeps_every_value(self) -> None:
        queue = build(BASE)
        queue._grow(32)
        assert_intact(self, queue, BASE)

    def test_unrolls_a_wrapped_queue(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        queue._grow(8)
        assert_intact(self, queue, BASE)

    def test_the_new_slots_are_empty(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        queue._grow(8)
        self.assertEqual(len(spare(queue)), 4, "the array is the wrong size")
        self.assertEqual(spare(queue), [None] * 4, "the new slots hold junk")

    def test_size_is_unchanged(self) -> None:
        queue = build(BASE)
        queue._grow(32)
        self.assertEqual(len(queue), 4, "_count drifted during the rebuild")

    def test_on_an_empty_queue(self) -> None:
        queue = ArrayQueue(4)
        queue._grow(8)
        assert_intact(self, queue, [])
        self.assertEqual(len(queue._items), 8)

    def test_to_exactly_the_count(self) -> None:
        queue = build(BASE, capacity=8, head=6)
        queue._grow(4)
        assert_intact(self, queue, BASE)


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it."""

    def test_empty_queue(self) -> None:
        self.assertTrue(ArrayQueue().is_valid())

    def test_populated_queue(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_wrapped_queue(self) -> None:
        self.assertTrue(build(BASE, capacity=4, head=3).is_valid())

    def test_full_array(self) -> None:
        self.assertTrue(build(BASE, capacity=4).is_valid())

    def test_head_past_the_end(self) -> None:
        queue = build(BASE, capacity=8)
        queue._head = 8
        self.assertFalse(queue.is_valid())

    def test_negative_head(self) -> None:
        queue = build(BASE, capacity=8)
        queue._head = -1
        self.assertFalse(queue.is_valid())

    def test_count_past_the_array(self) -> None:
        queue = build(BASE, capacity=4)
        queue._count = 5
        self.assertFalse(queue.is_valid())

    def test_negative_count(self) -> None:
        queue = build(BASE)
        queue._count = -1
        self.assertFalse(queue.is_valid())

    def test_value_left_outside_the_queue(self) -> None:
        queue = build(BASE, capacity=8)
        queue._items[6] = 99
        self.assertFalse(queue.is_valid(), "a served value is still in the array")

    def test_no_array_at_all(self) -> None:
        queue = ArrayQueue()
        queue._items = []
        self.assertFalse(queue.is_valid())


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _count directly, but __contains__ delegates to
    contains(), so it stays red until contains() is written."""

    def test_len(self) -> None:
        self.assertEqual(len(build(BASE)), 4)
        self.assertEqual(len(ArrayQueue()), 0)

    def test_len_is_not_the_capacity(self) -> None:
        self.assertEqual(len(build(BASE, capacity=64)), 4)

    def test_in_operator(self) -> None:
        queue = build(BASE)
        self.assertIn(30, queue)
        self.assertNotIn(99, queue)


class TestIteration(unittest.TestCase):

    def test_yields_front_first(self) -> None:
        self.assertEqual(list(build(BASE)), BASE)

    def test_on_a_wrapped_queue(self) -> None:
        queue = build(BASE, capacity=4, head=3)
        self.assertEqual(list(queue), BASE, "the wrap was iterated out of order")

    def test_yields_nothing_when_empty(self) -> None:
        self.assertEqual(list(ArrayQueue()), [])

    def test_stops_at_the_back(self) -> None:
        queue = build(BASE, capacity=8)
        queue._items[6] = 99
        self.assertEqual(list(queue), BASE, "it iterated past the back")

    def test_restarts_on_each_pass(self) -> None:
        queue = build(BASE)
        self.assertEqual(list(queue), list(queue), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        list(queue)
        assert_intact(self, queue, BASE)


class TestRepr(unittest.TestCase):

    def test_front_first(self) -> None:
        self.assertEqual(repr(build([1, 2, 3])), "ArrayQueue([1, 2, 3])")

    def test_on_a_wrapped_queue(self) -> None:
        queue = build([1, 2, 3], capacity=3, head=2)
        self.assertEqual(repr(queue), "ArrayQueue([1, 2, 3])")

    def test_empty_queue(self) -> None:
        self.assertEqual(repr(ArrayQueue()), "ArrayQueue([])")

    def test_quotes_strings(self) -> None:
        self.assertEqual(repr(build(["a"])), "ArrayQueue(['a'])")


if __name__ == "__main__":
    unittest.main(verbosity=2)
