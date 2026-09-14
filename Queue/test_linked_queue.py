"""Behavior tests for LinkedQueue, the node-backed queue.

Run from inside this directory:

    python3 -m unittest test_linked_queue -v
    python3 test_linked_queue.py            # equivalent

Each test class depends only on the method it names, so you can implement
the methods in any order and a green class always means that method is done.

build() wires the nodes together by hand and sets both _head and _tail, so
every fixture exists before a single method is written and a broken
enqueue() cannot fail an unrelated class. walk() then follows .next from
_head directly, which is why a test for dequeue() never calls to_list() to
see what happened.

assert_intact() checks _tail against the end of the chain after every
mutation. A queue that dequeues its last value without clearing _tail keeps
answering peek_back() with a node that is no longer in the queue, and the
next enqueue links onto that orphan instead of the queue - both look fine
from the front until something reads the back.

The deliberate couplings: __len__ reads _size and __contains__ delegates to
contains(), so it stays red until contains() is written. TestRoundTrip
drives enqueue() and dequeue() together and needs both. The print tests
assert that the values appear in front-to-back order, not the layout or
separator you print them with.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from linked_queue import LinkedQueue
from node import Node

# The queue used throughout: 10 at the front, 40 at the back.
BASE = [10, 20, 30, 40]


def build(values) -> LinkedQueue:
    """Return a queue holding values, front first, wiring nodes by hand.

    Deliberately avoids enqueue() so that fixtures work before any method
    is implemented, and so a broken enqueue() cannot fail unrelated classes.
    """
    queue = LinkedQueue()
    for value in values:
        node = Node(value)
        if queue._head is None:
            queue._head = node
        else:
            queue._tail.next = node
        queue._tail = node
        queue._size += 1
    return queue


def walk(head, max_nodes: int = 1000) -> list:
    """Collect values by following .next from the front."""
    values, node = [], head
    while node is not None:
        if len(values) > max_nodes:
            raise AssertionError("walked past max_nodes - the chain has a cycle")
        values.append(node.value)
        node = node.next
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

    Checks the chain, _size, and both ends: _tail has to be the last node
    and has to point at nothing, and an empty queue has to have let go of
    both pointers.
    """
    expected = list(expected)
    case.assertEqual(walk(queue._head), expected, "the node chain is wrong")
    case.assertEqual(len(queue), len(expected), "_size is out of step with the chain")
    case.assertIs(queue._tail, last_node(queue._head), "_tail is not the last node")
    if expected:
        case.assertIsNotNone(queue._head, "_head should not be None")
        case.assertIsNotNone(queue._tail, "_tail should not be None")
        case.assertIsNone(queue._tail.next, "_tail still points forward")
        case.assertEqual(queue._head.value, expected[0], "the wrong node is in front")
        case.assertEqual(queue._tail.value, expected[-1], "the wrong node is at back")
    else:
        case.assertIsNone(queue._head, "_head should be None when empty")
        case.assertIsNone(queue._tail, "_tail should be None when empty")


class TestEmptyQueue(unittest.TestCase):
    """An empty queue is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.queue = LinkedQueue()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.queue), 0)

    def test_both_ends_are_none(self) -> None:
        self.assertIsNone(self.queue._head)
        self.assertIsNone(self.queue._tail)

    def test_enqueue_sets_both_ends(self) -> None:
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

    def test_is_valid(self) -> None:
        self.assertTrue(self.queue.is_valid())

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.queue), [])

    def test_repr(self) -> None:
        self.assertEqual(repr(self.queue), "LinkedQueue([])")


class TestEnqueue(unittest.TestCase):

    def test_sets_both_ends_on_an_empty_queue(self) -> None:
        queue = LinkedQueue()
        queue.enqueue(10)
        assert_intact(self, queue, [10])
        self.assertIs(queue._head, queue._tail, "one value is both ends")

    def test_queues_values_in_order(self) -> None:
        queue = LinkedQueue()
        for value in BASE:
            queue.enqueue(value)
        assert_intact(self, queue, BASE)

    def test_adds_at_the_back_not_the_front(self) -> None:
        queue = build([10])
        queue.enqueue(20)
        assert_intact(self, queue, [10, 20])

    def test_links_the_old_tail_forward(self) -> None:
        queue = build([10])
        was_tail = queue._tail
        queue.enqueue(20)
        self.assertIs(was_tail.next, queue._tail, "the old tail was left dangling")

    def test_keeps_duplicates(self) -> None:
        queue = LinkedQueue()
        for value in [7, 7, 7]:
            queue.enqueue(value)
        assert_intact(self, queue, [7, 7, 7])

    def test_stores_none_as_a_value(self) -> None:
        queue = LinkedQueue()
        queue.enqueue(None)
        self.assertEqual(len(queue), 1, "None was not counted as a value")

    def test_size_tracks_every_enqueue(self) -> None:
        queue = LinkedQueue()
        for i, value in enumerate(BASE, start=1):
            queue.enqueue(value)
            self.assertEqual(len(queue), i, f"size wrong after {i} enqueues")

    def test_onto_a_built_queue(self) -> None:
        queue = build(BASE)
        queue.enqueue(50)
        assert_intact(self, queue, BASE + [50])

    def test_many_values(self) -> None:
        queue = LinkedQueue()
        for value in range(200):
            queue.enqueue(value)
        assert_intact(self, queue, list(range(200)))


class TestDequeue(unittest.TestCase):

    def test_returns_the_front(self) -> None:
        queue = build(BASE)
        self.assertEqual(queue.dequeue(), 10)
        assert_intact(self, queue, [20, 30, 40])

    def test_unlinks_the_node_it_returns(self) -> None:
        queue = build(BASE)
        was_head = queue._head
        queue.dequeue()
        self.assertIsNone(was_head.next, "the dequeued node still holds the queue")

    def test_drains_in_order(self) -> None:
        queue = build(BASE)
        served = [queue.dequeue() for _ in range(len(BASE))]
        self.assertEqual(served, BASE, "a queue is not first-in-first-out")
        assert_intact(self, queue, [])

    def test_the_only_value_clears_both_ends(self) -> None:
        queue = build([10])
        self.assertEqual(queue.dequeue(), 10)
        assert_intact(self, queue, [])

    def test_two_values_leaves_one_at_both_ends(self) -> None:
        queue = build([10, 20])
        self.assertEqual(queue.dequeue(), 10)
        assert_intact(self, queue, [20])
        self.assertIs(queue._head, queue._tail, "one value is both ends")

    def test_returns_a_stored_none(self) -> None:
        queue = build([None, 20])
        self.assertIsNone(queue.dequeue())
        assert_intact(self, queue, [20])

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            LinkedQueue().dequeue()

    def test_raising_changes_nothing(self) -> None:
        queue = LinkedQueue()
        with self.assertRaises(IndexError):
            queue.dequeue()
        assert_intact(self, queue, [])


class TestRoundTrip(unittest.TestCase):
    """Drives enqueue() and dequeue() together, so it needs both. These are
    the cases where _tail goes wrong: it is left pointing at a node that has
    already been served, and the next enqueue links onto that orphan instead
    of onto the queue."""

    def test_after_the_queue_has_drained(self) -> None:
        queue = build([10])
        queue.dequeue()
        queue.enqueue(99)
        assert_intact(self, queue, [99])

    def test_draining_then_refilling(self) -> None:
        queue = build(BASE)
        for _ in range(len(BASE)):
            queue.dequeue()
        queue.enqueue(99)
        assert_intact(self, queue, [99])

    def test_the_back_follows_an_enqueue(self) -> None:
        queue = build(BASE)
        queue.enqueue(50)
        self.assertEqual(queue.peek_back(), 50, "the back did not move")

    def test_the_back_survives_a_dequeue(self) -> None:
        queue = build([10, 20])
        queue.dequeue()
        self.assertEqual(queue.peek_back(), 20, "the back moved with the front")

    def test_alternating(self) -> None:
        queue, model = LinkedQueue(), []
        for i in range(50):
            queue.enqueue(i)
            model.append(i)
            if i % 2:
                self.assertEqual(queue.dequeue(), model.pop(0), "wrong value served")
            assert_intact(self, queue, model)

    def test_the_queue_stays_valid_throughout(self) -> None:
        queue = LinkedQueue()
        for i in range(60):
            queue.enqueue(i)
            if i % 3 == 0:
                queue.dequeue()
            self.assertTrue(queue.is_valid(), f"invalid after {i + 1} enqueues")


class TestPeek(unittest.TestCase):

    def test_returns_the_front(self) -> None:
        self.assertEqual(build(BASE).peek(), 10)

    def test_leaves_the_queue_alone(self) -> None:
        queue = build(BASE)
        queue.peek()
        assert_intact(self, queue, BASE)

    def test_repeats(self) -> None:
        queue = build(BASE)
        self.assertEqual([queue.peek(), queue.peek()], [10, 10])

    def test_the_only_value(self) -> None:
        self.assertEqual(build([10]).peek(), 10)

    def test_returns_a_stored_none(self) -> None:
        self.assertIsNone(build([None, 20]).peek())

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            LinkedQueue().peek()


class TestPeekBack(unittest.TestCase):

    def test_returns_the_back(self) -> None:
        self.assertEqual(build(BASE).peek_back(), 40)

    def test_is_not_the_front(self) -> None:
        self.assertEqual(build(BASE).peek_back(), 40, "peek_back() gave the front")

    def test_leaves_the_queue_alone(self) -> None:
        queue = build(BASE)
        queue.peek_back()
        assert_intact(self, queue, BASE)

    def test_the_only_value_is_both_ends(self) -> None:
        self.assertEqual(build([10]).peek_back(), 10)

    def test_returns_a_stored_none(self) -> None:
        self.assertIsNone(build([10, None]).peek_back())

    def test_raises_when_empty(self) -> None:
        with self.assertRaises(IndexError):
            LinkedQueue().peek_back()


class TestIsEmpty(unittest.TestCase):

    def test_true_for_a_new_queue(self) -> None:
        self.assertTrue(LinkedQueue().is_empty())

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

    def test_finds_a_duplicate(self) -> None:
        self.assertTrue(build([7, 7]).contains(7))

    def test_empty_queue(self) -> None:
        self.assertFalse(LinkedQueue().contains(10))

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        queue.contains(20)
        assert_intact(self, queue, BASE)


class TestToList(unittest.TestCase):

    def test_returns_the_values_front_first(self) -> None:
        self.assertEqual(build(BASE).to_list(), BASE)

    def test_one_value(self) -> None:
        self.assertEqual(build([10]).to_list(), [10])

    def test_empty_queue(self) -> None:
        self.assertEqual(LinkedQueue().to_list(), [])

    def test_keeps_duplicates(self) -> None:
        self.assertEqual(build([7, 7]).to_list(), [7, 7])

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        queue.to_list()
        assert_intact(self, queue, BASE)


class TestPrintQueue(unittest.TestCase):
    """Asserts the values print from front to back, not the layout."""

    def test_prints_front_to_back(self) -> None:
        assert_printed_in_order(self, printed(build(BASE).print_queue), BASE)

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

    def test_the_queue_is_reusable(self) -> None:
        queue = build(BASE)
        queue.clear()
        queue._head = queue._tail = Node(99)
        queue._size = 1
        assert_intact(self, queue, [99])

    def test_on_an_empty_queue(self) -> None:
        queue = LinkedQueue()
        queue.clear()
        assert_intact(self, queue, [])

    def test_on_one_value(self) -> None:
        queue = build([10])
        queue.clear()
        assert_intact(self, queue, [])


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it.
    Chains here always terminate - nothing in this class builds a cycle."""

    def test_empty_queue(self) -> None:
        self.assertTrue(LinkedQueue().is_valid())

    def test_one_value(self) -> None:
        self.assertTrue(build([10]).is_valid())

    def test_populated_queue(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_size_too_high(self) -> None:
        queue = build(BASE)
        queue._size += 1
        self.assertFalse(queue.is_valid())

    def test_size_too_low(self) -> None:
        queue = build(BASE)
        queue._size -= 1
        self.assertFalse(queue.is_valid())

    def test_tail_is_not_the_last_node(self) -> None:
        queue = build(BASE)
        queue._tail = queue._head
        self.assertFalse(queue.is_valid(), "_tail is pointing into the middle")

    def test_tail_still_points_forward(self) -> None:
        queue = build(BASE)
        queue._tail.next = Node(99)
        self.assertFalse(queue.is_valid())

    def test_head_set_without_a_tail(self) -> None:
        queue = LinkedQueue()
        queue._head = Node(10)
        queue._size = 1
        self.assertFalse(queue.is_valid())

    def test_tail_set_without_a_head(self) -> None:
        queue = LinkedQueue()
        queue._tail = Node(10)
        queue._size = 1
        self.assertFalse(queue.is_valid())

    def test_orphaned_tail_after_a_drain(self) -> None:
        queue = build([10])
        queue._head = None
        self.assertFalse(queue.is_valid(), "_tail outlived the value it points at")


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _size directly, but __contains__ delegates to
    contains(), so it stays red until contains() is written."""

    def test_len(self) -> None:
        self.assertEqual(len(build(BASE)), 4)
        self.assertEqual(len(LinkedQueue()), 0)

    def test_in_operator(self) -> None:
        queue = build(BASE)
        self.assertIn(30, queue)
        self.assertNotIn(99, queue)


class TestIteration(unittest.TestCase):

    def test_yields_front_first(self) -> None:
        self.assertEqual(list(build(BASE)), BASE)

    def test_yields_nothing_when_empty(self) -> None:
        self.assertEqual(list(LinkedQueue()), [])

    def test_restarts_on_each_pass(self) -> None:
        queue = build(BASE)
        self.assertEqual(list(queue), list(queue), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        queue = build(BASE)
        list(queue)
        assert_intact(self, queue, BASE)


class TestRepr(unittest.TestCase):

    def test_front_first(self) -> None:
        self.assertEqual(repr(build([1, 2, 3])), "LinkedQueue([1, 2, 3])")

    def test_empty_queue(self) -> None:
        self.assertEqual(repr(LinkedQueue()), "LinkedQueue([])")

    def test_quotes_strings(self) -> None:
        self.assertEqual(repr(build(["a"])), "LinkedQueue(['a'])")


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
