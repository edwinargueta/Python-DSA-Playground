"""Behavior tests for OpenAddressingTable, the linear probing table.

Run from inside this directory:

    python3 -m unittest test_open_addressing -v
    python3 test_open_addressing.py            # equivalent

Each test class depends only on the method it names, so you can implement
the methods in any order and a green class always means that method is done.

There is one unavoidable exception, and it is the reason to write _hash,
_slot_index and _next_index first. A fixture cannot place an entry without
walking the same probe run the table will walk, so build() and
assert_intact() call all three. Until they work, every class in this file
errors - the same trade the BST suite makes when its build() calls insert().
Nothing else is borrowed: slots() and live_pairs() read _keys and _values
straight off the table, so a test for remove() never calls get() to see
what happened.

Tombstones are the thing this structure gets wrong, so they are wired by
hand too. tombstone() marks a slot DELETED without going near remove(),
which is what lets TestPut assert that a put reuses one.

Two things are deliberately not asserted, because they are yours to choose:
which capacity a resize grows to, and the order keys come back in. Every
assertion here sorts or compares as a mapping.

The other deliberate couplings: __len__ reads _size, __contains__ delegates
to contains() and __setitem__ to put(), so those stay red until their
methods are written. TestAutomaticGrowth drives put() and whatever put()
grows the table with. TestPrintTable asserts that every key and value
reaches the output, not the layout you print it in.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from open_addressing_table import DELETED, EMPTY, OpenAddressingTable

# The table used throughout: four keys, exactly at a capacity of 8's limit.
BASE = [("apple", 1), ("banana", 2), ("cherry", 3), ("date", 4)]
BASE_DICT = dict(BASE)


def build(pairs, capacity: int = 8, max_load: float = 0.5) -> OpenAddressingTable:
    """Return a table holding pairs, placing each one by hand.

    Deliberately avoids put() so that fixtures work before any method is
    implemented, and so a broken put() cannot fail unrelated classes. It
    does walk the probe run with _slot_index() and _next_index(), because
    an entry has to sit where the table will look for it - see the module
    docstring.

    Entries land in the order given, so the first key of a colliding group
    takes its home slot, the second the slot after it, and so on. It never
    resizes, so a fixture may sit above its own load factor.
    """
    table = OpenAddressingTable(capacity, max_load)
    for key, value in pairs:
        index = table._slot_index(key)
        for _ in range(capacity + 1):
            if table._keys[index] is EMPTY:
                break
            index = table._next_index(index)
        else:
            raise AssertionError("no free slot - does _next_index() wrap?")
        table._keys[index] = key
        table._values[index] = value
        table._size += 1
    return table


def tombstone(table, index: int) -> None:
    """Mark one live slot DELETED by hand, without calling remove()."""
    key = table._keys[index]
    if key is EMPTY or key is DELETED:
        raise AssertionError(f"slot {index} holds no live entry to bury")
    table._keys[index] = DELETED
    table._values[index] = None
    table._size -= 1
    table._tombstones += 1


def slots(table) -> list:
    """Return every slot as (index, key, value), markers included."""
    return [(i, key, table._values[i]) for i, key in enumerate(table._keys)]


def live_pairs(table) -> list:
    """Collect (key, value) from the slots holding a real entry."""
    return [
        (key, value)
        for _, key, value in slots(table)
        if key is not EMPTY and key is not DELETED
    ]


def slot_of(table, key) -> int:
    """Return the slot a key is actually sitting in, or -1."""
    for index, current in enumerate(table._keys):
        if current is not EMPTY and current is not DELETED and current == key:
            return index
    return -1


def run_of(table, key, length: int) -> list:
    """Return the first length slots of key's probe run, home first."""
    index = table._slot_index(key)
    indices = [index]
    for _ in range(length - 1):
        index = table._next_index(index)
        indices.append(index)
    return indices


def colliding_keys(count: int = 3, capacity: int = 8, pool: int = 500) -> list:
    """Return count keys that _slot_index sends to one shared home slot.

    Found by asking the table itself, so it works whatever hash you write -
    and by the pigeonhole principle a slot has to repeat long before the
    pool runs out.
    """
    probe = OpenAddressingTable(capacity)
    groups: dict = {}
    for i in range(pool):
        group = groups.setdefault(probe._slot_index(f"k{i}"), [])
        group.append(f"k{i}")
        if len(group) == count:
            return group
    raise AssertionError(
        f"no {count} of {pool} keys shared a slot - is _slot_index in range?"
    )


def printed(call) -> str:
    """Run call and return everything it wrote to stdout."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


def assert_intact(case, table, expected, tombstones: int = 0) -> None:
    """Assert the table holds exactly expected and its invariants still hold.

    Checks the live entries, that no key is stored twice, _size, the
    tombstone count, that a free slot remains, and that every key is still
    reachable from its home slot without crossing an EMPTY. That last check
    is the one that matters: an entry orphaned behind an EMPTY slot still
    shows up in a scan of the table, and is never found by a lookup again.
    """
    expected = dict(expected)
    found = live_pairs(table)
    keys = [key for key, _ in found]
    capacity = len(table._keys)
    case.assertEqual(
        len(keys), len(set(keys)), f"a key is stored more than once: {keys}"
    )
    case.assertEqual(dict(found), expected, "the stored entries are wrong")
    case.assertEqual(len(table), len(expected), "_size is out of step")
    case.assertEqual(len(table._values), capacity, "_keys and _values differ in size")
    case.assertGreaterEqual(capacity, 1, "the table has no slots left")
    buried = sum(1 for key in table._keys if key is DELETED)
    case.assertEqual(buried, tombstones, "the table has the wrong tombstone count")
    case.assertEqual(table._tombstones, buried, "_tombstones is out of step")
    case.assertLess(
        len(table) + buried, capacity, "no EMPTY slot is left - a probe cannot end"
    )
    for index, key, _ in slots(table):
        if key is EMPTY or key is DELETED:
            continue
        probe = table._slot_index(key)
        steps = 0
        while probe != index:
            case.assertIsNot(
                table._keys[probe],
                EMPTY,
                f"{key!r} sits behind an EMPTY slot and can never be found",
            )
            probe = table._next_index(probe)
            steps += 1
            case.assertLessEqual(steps, capacity, f"{key!r} is off its probe run")


class TestHash(unittest.TestCase):
    """The foundation - build() and assert_intact() both need _slot_index
    and _next_index, so every other class stays red until these work."""

    def setUp(self) -> None:
        self.table = OpenAddressingTable()

    def test_returns_an_int(self) -> None:
        self.assertIsInstance(self.table._hash("apple"), int)

    def test_is_never_negative(self) -> None:
        for key in ["apple", "", 0, 7, -7, -1, 3.5, (1, 2), None, True]:
            self.assertGreaterEqual(
                self.table._hash(key), 0, f"_hash({key!r}) is negative"
            )

    def test_is_deterministic(self) -> None:
        self.assertEqual(self.table._hash("apple"), self.table._hash("apple"))

    def test_equal_keys_hash_equal(self) -> None:
        self.assertEqual(self.table._hash("apple"), self.table._hash("app" + "le"))
        self.assertEqual(self.table._hash((1, 2)), self.table._hash(tuple([1, 2])))


class TestSlotIndex(unittest.TestCase):

    def test_is_in_range(self) -> None:
        table = OpenAddressingTable(8)
        for i in range(200):
            self.assertIn(table._slot_index(f"k{i}"), range(8), f"k{i} out of range")

    def test_is_in_range_for_one_slot(self) -> None:
        table = OpenAddressingTable(1)
        for i in range(50):
            self.assertEqual(table._slot_index(f"k{i}"), 0)

    def test_is_in_range_for_mixed_key_types(self) -> None:
        table = OpenAddressingTable(8)
        for key in ["apple", "", 0, 7, -7, 3.5, (1, 2), None, True]:
            self.assertIn(table._slot_index(key), range(8), f"{key!r} out of range")

    def test_is_deterministic(self) -> None:
        table = OpenAddressingTable(8)
        self.assertEqual(table._slot_index("apple"), table._slot_index("apple"))

    def test_equal_keys_share_a_home(self) -> None:
        table = OpenAddressingTable(8)
        self.assertEqual(table._slot_index("apple"), table._slot_index("app" + "le"))

    def test_follows_the_capacity(self) -> None:
        self.assertIn(OpenAddressingTable(4)._slot_index("apple"), range(4))
        self.assertIn(OpenAddressingTable(64)._slot_index("apple"), range(64))

    def test_spreads_keys_over_more_than_one_slot(self) -> None:
        table = OpenAddressingTable(64)
        used = {table._slot_index(f"k{i}") for i in range(200)}
        self.assertGreater(len(used), 1, "every key came home to one slot")


class TestNextIndex(unittest.TestCase):
    """The probe run has to reach every slot, or a lookup can report a key
    missing while its slot sits free somewhere the probe never goes."""

    def test_stays_in_range(self) -> None:
        table = OpenAddressingTable(8)
        for index in range(8):
            self.assertIn(table._next_index(index), range(8), f"from {index}")

    def test_moves_on(self) -> None:
        table = OpenAddressingTable(8)
        for index in range(8):
            self.assertNotEqual(table._next_index(index), index, "the probe stalls")

    def test_wraps_past_the_last_slot(self) -> None:
        table = OpenAddressingTable(8)
        self.assertIn(table._next_index(7), range(8), "the probe ran off the end")

    def test_visits_every_slot(self) -> None:
        table = OpenAddressingTable(8)
        seen, index = [], 0
        for _ in range(8):
            seen.append(index)
            index = table._next_index(index)
        self.assertEqual(sorted(seen), list(range(8)), "the probe run skips slots")

    def test_returns_to_the_start(self) -> None:
        table = OpenAddressingTable(8)
        index = 3
        for _ in range(8):
            index = table._next_index(index)
        self.assertEqual(index, 3, "a full lap does not come home")

    def test_one_slot_table_stays_put(self) -> None:
        self.assertEqual(OpenAddressingTable(1)._next_index(0), 0)


class TestEmptyTable(unittest.TestCase):
    """An empty table is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.table = OpenAddressingTable()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.table), 0)

    def test_every_slot_is_empty(self) -> None:
        self.assertEqual(live_pairs(self.table), [])

    def test_find_slot_returns_minus_one(self) -> None:
        self.assertEqual(self.table._find_slot("apple"), -1)

    def test_insert_slot_is_the_home_slot(self) -> None:
        self.assertEqual(
            self.table._insert_slot("apple"), self.table._slot_index("apple")
        )

    def test_put_stores_the_first_entry(self) -> None:
        self.table.put("apple", 1)
        assert_intact(self, self.table, {"apple": 1})

    def test_get_returns_none(self) -> None:
        self.assertIsNone(self.table.get("apple"))

    def test_get_returns_the_default(self) -> None:
        self.assertEqual(self.table.get("apple", "missing"), "missing")

    def test_contains_returns_false(self) -> None:
        self.assertFalse(self.table.contains("apple"))

    def test_keys_is_empty(self) -> None:
        self.assertEqual(self.table.keys(), [])

    def test_values_is_empty(self) -> None:
        self.assertEqual(self.table.values(), [])

    def test_items_is_empty(self) -> None:
        self.assertEqual(self.table.items(), [])

    def test_print_table_prints_no_entries(self) -> None:
        self.assertNotIn("apple", printed(self.table.print_table))

    def test_remove_returns_false(self) -> None:
        self.assertFalse(self.table.remove("apple"))
        assert_intact(self, self.table, {})

    def test_pop_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.table.pop("apple")

    def test_clear_is_a_no_op(self) -> None:
        self.table.clear()
        assert_intact(self, self.table, {})

    def test_capacity_is_unchanged(self) -> None:
        self.assertEqual(self.table.capacity(), 8)

    def test_load_factor_is_zero(self) -> None:
        self.assertEqual(self.table.load_factor(), 0.0)

    def test_occupancy_is_zero(self) -> None:
        self.assertEqual(self.table.occupancy(), 0.0)

    def test_tombstone_count_is_zero(self) -> None:
        self.assertEqual(self.table.tombstone_count(), 0)

    def test_probe_length_is_one(self) -> None:
        self.assertEqual(
            self.table.probe_length("apple"), 1, "the home slot alone ends the search"
        )

    def test_resize_keeps_it_empty(self) -> None:
        self.table._resize(16)
        assert_intact(self, self.table, {})
        self.assertEqual(len(self.table._keys), 16)

    def test_is_valid(self) -> None:
        self.assertTrue(self.table.is_valid())

    def test_getitem_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.table["apple"]

    def test_delitem_raises(self) -> None:
        with self.assertRaises(KeyError):
            del self.table["apple"]

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.table), [])

    def test_repr(self) -> None:
        self.assertEqual(repr(self.table), "OpenAddressingTable({})")


class TestFindSlot(unittest.TestCase):

    def test_finds_a_key_in_its_home_slot(self) -> None:
        table = build(BASE)
        for key, _ in BASE:
            self.assertEqual(
                table._find_slot(key), slot_of(table, key), f"{key!r} was not found"
            )

    def test_follows_the_probe_run(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        expected = run_of(table, keys[0], 3)
        for key, index in zip(keys, expected):
            self.assertEqual(table._find_slot(key), index, f"{key!r} is off its run")

    def test_returns_minus_one_when_absent(self) -> None:
        self.assertEqual(build(BASE)._find_slot("fig"), -1)

    def test_stops_at_the_first_empty_slot(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        self.assertEqual(
            table._find_slot(keys[3]), -1, "matched the probe run, not the key"
        )

    def test_looks_past_a_tombstone(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        tombstone(table, slot_of(table, keys[0]))
        self.assertNotEqual(
            table._find_slot(keys[2]), -1, "a tombstone cut the probe run short"
        )

    def test_a_buried_key_is_gone(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        tombstone(table, slot_of(table, keys[0]))
        self.assertEqual(table._find_slot(keys[0]), -1)


class TestInsertSlot(unittest.TestCase):

    def test_empty_home_slot(self) -> None:
        table = OpenAddressingTable(8)
        self.assertEqual(table._insert_slot("apple"), table._slot_index("apple"))

    def test_returns_the_slot_a_key_already_holds(self) -> None:
        table = build(BASE)
        for key, _ in BASE:
            self.assertEqual(
                table._insert_slot(key), slot_of(table, key), f"{key!r} would move"
            )

    def test_walks_past_an_occupied_slot(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        self.assertEqual(table._insert_slot(keys[3]), run_of(table, keys[3], 4)[3])

    def test_never_returns_another_live_key(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        index = table._insert_slot(keys[3])
        self.assertIn(
            table._keys[index], (EMPTY, DELETED), "it would overwrite a live key"
        )

    def test_reuses_the_first_tombstone(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        home = run_of(table, keys[0], 1)[0]
        tombstone(table, home)
        self.assertEqual(
            table._insert_slot(keys[3]), home, "a fresh slot was taken instead"
        )

    def test_an_existing_key_wins_over_a_tombstone(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        tombstone(table, slot_of(table, keys[0]))
        self.assertEqual(
            table._insert_slot(keys[2]),
            slot_of(table, keys[2]),
            "the key would be stored a second time",
        )


class TestPut(unittest.TestCase):

    def test_stores_the_first_entry(self) -> None:
        table = OpenAddressingTable()
        table.put("apple", 1)
        assert_intact(self, table, {"apple": 1})

    def test_stores_several_keys(self) -> None:
        table = OpenAddressingTable(16)
        for key, value in BASE:
            table.put(key, value)
        assert_intact(self, table, BASE_DICT)

    def test_overwrites_an_existing_key(self) -> None:
        table = build(BASE)
        table.put("banana", 99)
        assert_intact(self, table, {**BASE_DICT, "banana": 99})

    def test_overwriting_does_not_grow_the_size(self) -> None:
        table = build(BASE)
        for _ in range(5):
            table.put("banana", 99)
        self.assertEqual(len(table), len(BASE), "the key was stored more than once")

    def test_overwriting_keeps_the_key_where_it_is(self) -> None:
        table = build(BASE)
        before = slot_of(table, "banana")
        table.put("banana", 99)
        self.assertEqual(slot_of(table, "banana"), before, "the key moved slots")

    def test_stores_none_as_a_value(self) -> None:
        table = OpenAddressingTable()
        table.put("apple", None)
        assert_intact(self, table, {"apple": None})

    def test_keys_sharing_a_home_all_survive(self) -> None:
        keys = colliding_keys(3)
        table = OpenAddressingTable(16)
        for i, key in enumerate(keys):
            table.put(key, i)
        assert_intact(self, table, {key: i for i, key in enumerate(keys)})

    def test_reuses_a_tombstone(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        home = run_of(table, keys[0], 1)[0]
        tombstone(table, home)
        table.put(keys[3], 99)
        self.assertEqual(slot_of(table, keys[3]), home, "the tombstone was skipped")
        self.assertEqual(table._tombstones, 0, "_tombstones was not decremented")

    def test_accepts_mixed_key_types(self) -> None:
        pairs = {"apple": 1, 7: 2, (1, 2): 3, None: 4, 3.5: 5}
        table = OpenAddressingTable(32)
        for key, value in pairs.items():
            table.put(key, value)
        assert_intact(self, table, pairs)

    def test_size_tracks_every_new_key(self) -> None:
        table = OpenAddressingTable()
        for i in range(50):
            table.put(f"k{i}", i)
            self.assertEqual(len(table), i + 1, f"size wrong after {i + 1} puts")

    def test_many_keys_all_survive(self) -> None:
        table = OpenAddressingTable()
        expected = {f"k{i}": i for i in range(200)}
        for key, value in expected.items():
            table.put(key, value)
        assert_intact(self, table, expected)


class TestAutomaticGrowth(unittest.TestCase):
    """put() grows the table on its own, so this class needs put() and
    whatever put() resizes with - the capacity it grows to is yours."""

    def test_grows_before_the_occupancy_is_exceeded(self) -> None:
        table = OpenAddressingTable(capacity=8, max_load=0.5)
        for i in range(5):
            table.put(f"k{i}", i)
        self.assertGreater(len(table._keys), 8, "the table never grew")

    def test_stays_under_the_occupancy_throughout(self) -> None:
        table = OpenAddressingTable(capacity=8, max_load=0.5)
        for i in range(200):
            table.put(f"k{i}", i)
            occupancy = (len(table) + table._tombstones) / len(table._keys)
            self.assertLessEqual(
                occupancy, 0.5, f"occupancy is above max_load after {i + 1} puts"
            )

    def test_every_entry_is_rehomed(self) -> None:
        table = OpenAddressingTable(capacity=4, max_load=0.5)
        expected = {f"k{i}": i for i in range(100)}
        for key, value in expected.items():
            table.put(key, value)
        assert_intact(self, table, expected)

    def test_overwrites_never_grow_the_table(self) -> None:
        table = OpenAddressingTable(capacity=8, max_load=0.5)
        for i in range(50):
            table.put("apple", i)
        self.assertEqual(len(table._keys), 8, "an overwrite grew the table")

    def test_churn_does_not_fill_the_table_with_tombstones(self) -> None:
        table = OpenAddressingTable(capacity=8, max_load=0.5)
        for i in range(200):
            table.put(f"k{i}", i)
            table.remove(f"k{i}")
            occupancy = (len(table) + table._tombstones) / len(table._keys)
            self.assertLess(occupancy, 1.0, f"the table filled up by round {i + 1}")
        self.assertEqual(len(table), 0)


class TestGet(unittest.TestCase):

    def test_returns_the_value(self) -> None:
        table = build(BASE)
        for key, value in BASE:
            self.assertEqual(table.get(key), value, f"wrong value for {key!r}")

    def test_returns_none_when_absent(self) -> None:
        self.assertIsNone(build(BASE).get("fig"))

    def test_returns_the_default_when_absent(self) -> None:
        self.assertEqual(build(BASE).get("fig", "missing"), "missing")

    def test_a_stored_none_is_not_the_default(self) -> None:
        table = build([("apple", None)])
        self.assertIsNone(table.get("apple", "missing"), "a stored None was skipped")

    def test_finds_every_key_on_a_shared_run(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        for i, key in enumerate(keys):
            self.assertEqual(table.get(key), i, f"{key!r} was lost on its run")

    def test_absent_key_on_an_occupied_run(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        self.assertIsNone(table.get(keys[3]), "matched the probe run, not the key")

    def test_reads_past_a_tombstone(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        tombstone(table, slot_of(table, keys[0]))
        self.assertEqual(table.get(keys[2]), 2, "a tombstone cut the run short")

    def test_reads_do_not_mutate(self) -> None:
        table = build(BASE)
        table.get("apple")
        table.get("fig")
        assert_intact(self, table, BASE_DICT)


class TestContains(unittest.TestCase):

    def test_finds_every_key(self) -> None:
        table = build(BASE)
        for key, _ in BASE:
            self.assertTrue(table.contains(key), f"{key!r} was not found")

    def test_rejects_an_absent_key(self) -> None:
        self.assertFalse(build(BASE).contains("fig"))

    def test_distinguishes_a_stored_none_from_a_missing_key(self) -> None:
        table = build([("apple", None)])
        self.assertTrue(table.contains("apple"), "a key holding None reads absent")
        self.assertFalse(table.contains("fig"))

    def test_finds_every_key_on_a_shared_run(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        for key in keys:
            self.assertTrue(table.contains(key), f"{key!r} was lost on its run")

    def test_looks_past_a_tombstone(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        tombstone(table, slot_of(table, keys[0]))
        self.assertTrue(table.contains(keys[2]), "a tombstone cut the run short")

    def test_rejects_a_buried_key(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        tombstone(table, slot_of(table, keys[0]))
        self.assertFalse(table.contains(keys[0]))


class TestRemove(unittest.TestCase):

    def test_removes_a_key(self) -> None:
        table = build(BASE)
        self.assertTrue(table.remove("banana"))
        assert_intact(self, table, {k: v for k, v in BASE if k != "banana"}, 1)

    def test_leaves_a_tombstone_not_an_empty_slot(self) -> None:
        table = build(BASE)
        index = slot_of(table, "banana")
        table.remove("banana")
        self.assertIs(
            table._keys[index], DELETED, "an EMPTY slot orphans the keys behind it"
        )

    def test_keys_behind_it_are_still_reachable(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        table.remove(keys[0])
        expected = {key: i for i, key in enumerate(keys) if key != keys[0]}
        assert_intact(self, table, expected, 1)

    def test_removes_each_key_on_a_shared_run(self) -> None:
        keys = colliding_keys(3)
        pairs = [(key, i) for i, key in enumerate(keys)]
        for target in keys:
            table = build(pairs)
            self.assertTrue(table.remove(target), f"{target!r} was not removed")
            assert_intact(self, table, {k: v for k, v in pairs if k != target}, 1)

    def test_returns_false_when_absent(self) -> None:
        table = build(BASE)
        self.assertFalse(table.remove("fig"))
        assert_intact(self, table, BASE_DICT)

    def test_absent_key_on_an_occupied_run_changes_nothing(self) -> None:
        keys = colliding_keys(4)
        pairs = [(key, i) for i, key in enumerate(keys[:3])]
        table = build(pairs)
        self.assertFalse(table.remove(keys[3]), "matched the probe run, not the key")
        assert_intact(self, table, dict(pairs))

    def test_removing_twice_returns_false(self) -> None:
        table = build(BASE)
        self.assertTrue(table.remove("apple"))
        self.assertFalse(table.remove("apple"))
        assert_intact(self, table, {k: v for k, v in BASE if k != "apple"}, 1)

    def test_drains_every_key(self) -> None:
        table = build(BASE)
        for key, _ in BASE:
            self.assertTrue(table.remove(key), f"{key!r} was not removed")
        assert_intact(self, table, {}, len(BASE))

    def test_does_not_shrink_the_capacity(self) -> None:
        table = build(BASE)
        for key, _ in BASE:
            table.remove(key)
        self.assertEqual(len(table._keys), 8, "removing shrank the table")

    def test_clears_the_value_it_leaves_behind(self) -> None:
        table = build(BASE)
        index = slot_of(table, "banana")
        table.remove("banana")
        self.assertIsNone(table._values[index], "the removed value is still held")


class TestPop(unittest.TestCase):

    def test_returns_the_value_and_removes_the_key(self) -> None:
        table = build(BASE)
        self.assertEqual(table.pop("banana"), 2)
        assert_intact(self, table, {k: v for k, v in BASE if k != "banana"}, 1)

    def test_pops_each_key_on_a_shared_run(self) -> None:
        keys = colliding_keys(3)
        pairs = [(key, i) for i, key in enumerate(keys)]
        for i, target in enumerate(keys):
            table = build(pairs)
            self.assertEqual(table.pop(target), i, f"wrong value for {target!r}")
            assert_intact(self, table, {k: v for k, v in pairs if k != target}, 1)

    def test_returns_a_stored_none(self) -> None:
        table = build([("apple", None)])
        self.assertIsNone(table.pop("apple"))
        assert_intact(self, table, {}, 1)

    def test_raises_when_absent(self) -> None:
        with self.assertRaises(KeyError):
            build(BASE).pop("fig")

    def test_raising_changes_nothing(self) -> None:
        table = build(BASE)
        with self.assertRaises(KeyError):
            table.pop("fig")
        assert_intact(self, table, BASE_DICT)

    def test_popping_twice_raises(self) -> None:
        table = build(BASE)
        table.pop("apple")
        with self.assertRaises(KeyError):
            table.pop("apple")


class TestClear(unittest.TestCase):

    def test_drops_every_entry(self) -> None:
        table = build(BASE)
        table.clear()
        assert_intact(self, table, {})

    def test_drops_the_tombstones_too(self) -> None:
        table = build(BASE)
        tombstone(table, slot_of(table, "apple"))
        table.clear()
        self.assertNotIn(DELETED, table._keys, "a tombstone survived the clear")
        self.assertEqual(table._tombstones, 0, "_tombstones was not reset")

    def test_keeps_the_capacity(self) -> None:
        table = build(BASE, capacity=16)
        table.clear()
        self.assertEqual(len(table._keys), 16, "clear() changed the capacity")

    def test_the_table_is_reusable(self) -> None:
        table = build(BASE)
        table.clear()
        index = table._slot_index("fig")
        table._keys[index] = "fig"
        table._values[index] = 9
        table._size += 1
        assert_intact(self, table, {"fig": 9})

    def test_on_an_empty_table(self) -> None:
        table = OpenAddressingTable()
        table.clear()
        assert_intact(self, table, {})


class TestKeys(unittest.TestCase):

    def test_returns_every_key(self) -> None:
        self.assertEqual(sorted(build(BASE).keys()), sorted(BASE_DICT))

    def test_skips_tombstones(self) -> None:
        table = build(BASE)
        tombstone(table, slot_of(table, "apple"))
        self.assertEqual(sorted(table.keys()), ["banana", "cherry", "date"])

    def test_is_empty_for_an_empty_table(self) -> None:
        self.assertEqual(OpenAddressingTable().keys(), [])

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        table.keys()
        assert_intact(self, table, BASE_DICT)


class TestValues(unittest.TestCase):

    def test_returns_every_value(self) -> None:
        self.assertEqual(sorted(build(BASE).values()), sorted(BASE_DICT.values()))

    def test_keeps_duplicate_values(self) -> None:
        table = build([("apple", 1), ("banana", 1)])
        self.assertEqual(sorted(table.values()), [1, 1], "a duplicate value was lost")

    def test_skips_tombstones(self) -> None:
        table = build(BASE)
        tombstone(table, slot_of(table, "apple"))
        self.assertEqual(sorted(table.values()), [2, 3, 4])

    def test_is_empty_for_an_empty_table(self) -> None:
        self.assertEqual(OpenAddressingTable().values(), [])

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        table.values()
        assert_intact(self, table, BASE_DICT)


class TestItems(unittest.TestCase):

    def test_returns_every_pair(self) -> None:
        self.assertEqual(sorted(build(BASE).items()), sorted(BASE))

    def test_skips_tombstones(self) -> None:
        table = build(BASE)
        tombstone(table, slot_of(table, "apple"))
        self.assertEqual(sorted(table.items()), sorted(BASE[1:]))

    def test_is_empty_for_an_empty_table(self) -> None:
        self.assertEqual(OpenAddressingTable().items(), [])

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        table.items()
        assert_intact(self, table, BASE_DICT)


class TestPrintTable(unittest.TestCase):
    """Asserts that every key and value reaches the output, not the layout."""

    def test_prints_every_key_and_value(self) -> None:
        table = build(BASE)
        output = printed(table.print_table)
        for key, value in BASE:
            self.assertIn(str(key), output, f"{key!r} was not printed")
            self.assertIn(str(value), output, f"the value of {key!r} was not printed")

    def test_prints_keys_sharing_a_run(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        output = printed(table.print_table)
        for key in keys:
            self.assertIn(key, output, f"{key!r} was hidden behind its run")

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        printed(table.print_table)
        assert_intact(self, table, BASE_DICT)


class TestCapacity(unittest.TestCase):

    def test_default(self) -> None:
        self.assertEqual(OpenAddressingTable().capacity(), 8)

    def test_custom(self) -> None:
        self.assertEqual(OpenAddressingTable(32).capacity(), 32)

    def test_tracks_the_slots(self) -> None:
        table = OpenAddressingTable(8)
        table._keys = [EMPTY] * 64
        table._values = [None] * 64
        self.assertEqual(table.capacity(), 64, "capacity() is not reading the slots")


class TestLoadFactor(unittest.TestCase):

    def test_is_zero_when_empty(self) -> None:
        self.assertEqual(OpenAddressingTable(8).load_factor(), 0.0)

    def test_is_live_entries_over_slots(self) -> None:
        self.assertEqual(build(BASE, capacity=8).load_factor(), 0.5)

    def test_follows_the_capacity(self) -> None:
        self.assertEqual(build(BASE, capacity=16).load_factor(), 0.25)

    def test_ignores_tombstones(self) -> None:
        table = build(BASE, capacity=8)
        tombstone(table, slot_of(table, "apple"))
        self.assertEqual(
            table.load_factor(), 0.375, "load_factor() is counting tombstones"
        )


class TestOccupancy(unittest.TestCase):
    """The quantity that has to drive a resize: tombstones take up slots
    that a probe still has to walk over."""

    def test_is_zero_when_empty(self) -> None:
        self.assertEqual(OpenAddressingTable(8).occupancy(), 0.0)

    def test_matches_the_load_factor_without_tombstones(self) -> None:
        self.assertEqual(build(BASE, capacity=8).occupancy(), 0.5)

    def test_counts_tombstones(self) -> None:
        table = build(BASE, capacity=8)
        tombstone(table, slot_of(table, "apple"))
        self.assertEqual(table.occupancy(), 0.5, "the tombstone was not counted")

    def test_counts_tombstones_alone(self) -> None:
        table = build(BASE, capacity=8)
        for key, _ in BASE:
            tombstone(table, slot_of(table, key))
        self.assertEqual(table.occupancy(), 0.5, "an emptied table still holds slots")


class TestTombstoneCount(unittest.TestCase):

    def test_is_zero_when_empty(self) -> None:
        self.assertEqual(OpenAddressingTable().tombstone_count(), 0)

    def test_is_zero_when_nothing_was_removed(self) -> None:
        self.assertEqual(build(BASE).tombstone_count(), 0)

    def test_counts_each_one(self) -> None:
        table = build(BASE)
        for i, (key, _) in enumerate(BASE, start=1):
            tombstone(table, slot_of(table, key))
            self.assertEqual(table.tombstone_count(), i, f"after {i} removals")


class TestProbeLength(unittest.TestCase):
    """Counts every slot read, the one that ends the search included, so a
    key in its home slot reads 1 and a miss on an empty slot reads 1."""

    def test_a_key_in_its_home_slot(self) -> None:
        table = build([("apple", 1)])
        self.assertEqual(table.probe_length("apple"), 1)

    def test_a_miss_on_an_empty_slot(self) -> None:
        self.assertEqual(OpenAddressingTable(8).probe_length("apple"), 1)

    def test_grows_along_a_shared_run(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        for i, key in enumerate(keys):
            self.assertEqual(table.probe_length(key), i + 1, f"{key!r} is misplaced")

    def test_a_miss_walks_the_whole_run(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        self.assertEqual(
            table.probe_length(keys[3]), 4, "three occupied slots, then the empty one"
        )

    def test_a_tombstone_does_not_end_the_search(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        tombstone(table, slot_of(table, keys[0]))
        self.assertEqual(
            table.probe_length(keys[2]), 3, "the tombstone was read as an ending"
        )

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        table.probe_length("apple")
        table.probe_length("fig")
        assert_intact(self, table, BASE_DICT)


class TestResize(unittest.TestCase):

    def test_changes_the_capacity(self) -> None:
        table = build(BASE)
        table._resize(32)
        self.assertEqual(len(table._keys), 32)
        self.assertEqual(len(table._values), 32, "_values was left at the old size")

    def test_keeps_every_entry(self) -> None:
        table = build(BASE)
        table._resize(32)
        assert_intact(self, table, BASE_DICT)

    def test_shrinking_keeps_every_entry(self) -> None:
        table = build(BASE, capacity=32)
        table._resize(8)
        assert_intact(self, table, BASE_DICT)

    def test_drops_the_tombstones(self) -> None:
        table = build(BASE)
        tombstone(table, slot_of(table, "apple"))
        table._resize(16)
        expected = {k: v for k, v in BASE if k != "apple"}
        assert_intact(self, table, expected)
        self.assertEqual(table._tombstones, 0, "_tombstones was not reset")

    def test_on_an_empty_table(self) -> None:
        table = OpenAddressingTable(8)
        table._resize(16)
        assert_intact(self, table, {})
        self.assertEqual(len(table._keys), 16)

    def test_unpacks_a_shared_run(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)], capacity=8)
        table._resize(64)
        assert_intact(self, table, {key: i for i, key in enumerate(keys)})

    def test_many_entries_survive(self) -> None:
        expected = {f"k{i}": i for i in range(100)}
        table = build(list(expected.items()), capacity=256)
        table._resize(512)
        assert_intact(self, table, expected)

    def test_size_is_unchanged(self) -> None:
        table = build(BASE)
        table._resize(32)
        self.assertEqual(len(table), 4, "_size drifted during the rebuild")


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it."""

    def test_empty_table(self) -> None:
        self.assertTrue(OpenAddressingTable().is_valid())

    def test_populated_table(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_shared_run(self) -> None:
        keys = colliding_keys(3)
        self.assertTrue(build([(key, i) for i, key in enumerate(keys)]).is_valid())

    def test_table_holding_tombstones(self) -> None:
        table = build(BASE)
        tombstone(table, slot_of(table, "apple"))
        self.assertTrue(table.is_valid(), "a tombstone is a legitimate slot")

    def test_size_too_high(self) -> None:
        table = build(BASE)
        table._size += 1
        self.assertFalse(table.is_valid())

    def test_size_too_low(self) -> None:
        table = build(BASE)
        table._size -= 1
        self.assertFalse(table.is_valid())

    def test_tombstone_count_out_of_step(self) -> None:
        table = build(BASE)
        table._tombstones += 1
        self.assertFalse(table.is_valid())

    def test_key_stranded_behind_an_empty_slot(self) -> None:
        table = OpenAddressingTable(8)
        home = table._slot_index("fig")
        stranded = table._next_index(table._next_index(home))
        table._keys[stranded] = "fig"
        table._values[stranded] = 9
        table._size += 1
        self.assertFalse(table.is_valid(), "no lookup will ever reach that key")

    def test_duplicate_key_in_two_slots(self) -> None:
        table = build([("apple", 1)])
        index = table._next_index(slot_of(table, "apple"))
        table._keys[index] = "apple"
        table._values[index] = 2
        table._size += 1
        self.assertFalse(table.is_valid(), "the same key is stored twice")

    def test_table_with_no_empty_slot(self) -> None:
        table = build([("apple", 1), ("banana", 2), ("cherry", 3)], capacity=4)
        remaining = table._keys.index(EMPTY)
        table._keys[remaining] = "date"
        table._values[remaining] = 4
        table._size += 1
        self.assertFalse(table.is_valid(), "a probe for a missing key cannot end")


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _size directly, but __contains__ delegates to
    contains() and __setitem__ to put(), so those stay red until their
    methods are written."""

    def test_len(self) -> None:
        self.assertEqual(len(build(BASE)), 4)
        self.assertEqual(len(OpenAddressingTable()), 0)

    def test_in_operator(self) -> None:
        table = build(BASE)
        self.assertIn("apple", table)
        self.assertNotIn("fig", table)

    def test_assignment(self) -> None:
        table = OpenAddressingTable()
        table["apple"] = 1
        assert_intact(self, table, {"apple": 1})

    def test_assignment_overwrites(self) -> None:
        table = build(BASE)
        table["banana"] = 99
        assert_intact(self, table, {**BASE_DICT, "banana": 99})


class TestGetItem(unittest.TestCase):

    def test_returns_the_value(self) -> None:
        self.assertEqual(build(BASE)["banana"], 2)

    def test_raises_when_absent(self) -> None:
        with self.assertRaises(KeyError):
            build(BASE)["fig"]

    def test_returns_a_stored_none(self) -> None:
        self.assertIsNone(build([("apple", None)])["apple"])


class TestDelItem(unittest.TestCase):

    def test_removes_the_key(self) -> None:
        table = build(BASE)
        del table["banana"]
        assert_intact(self, table, {k: v for k, v in BASE if k != "banana"}, 1)

    def test_raises_when_absent(self) -> None:
        with self.assertRaises(KeyError):
            del build(BASE)["fig"]

    def test_raising_changes_nothing(self) -> None:
        table = build(BASE)
        with self.assertRaises(KeyError):
            del table["fig"]
        assert_intact(self, table, BASE_DICT)


class TestIteration(unittest.TestCase):

    def test_yields_every_key(self) -> None:
        self.assertEqual(sorted(build(BASE)), sorted(BASE_DICT))

    def test_yields_nothing_when_empty(self) -> None:
        self.assertEqual(list(OpenAddressingTable()), [])

    def test_skips_tombstones(self) -> None:
        table = build(BASE)
        tombstone(table, slot_of(table, "apple"))
        self.assertEqual(sorted(table), ["banana", "cherry", "date"])

    def test_restarts_on_each_pass(self) -> None:
        table = build(BASE)
        self.assertEqual(sorted(table), sorted(table), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        list(table)
        assert_intact(self, table, BASE_DICT)


class TestRepr(unittest.TestCase):

    def test_one_entry(self) -> None:
        self.assertEqual(repr(build([("a", 1)])), "OpenAddressingTable({'a': 1})")

    def test_empty_table(self) -> None:
        self.assertEqual(repr(OpenAddressingTable()), "OpenAddressingTable({})")

    def test_quotes_string_values(self) -> None:
        self.assertEqual(repr(build([("a", "b")])), "OpenAddressingTable({'a': 'b'})")


class TestSlotMarkers(unittest.TestCase):

    def test_a_new_table_is_all_empty(self) -> None:
        table = OpenAddressingTable(8)
        self.assertEqual(table._keys, [EMPTY] * 8)
        self.assertEqual(table._values, [None] * 8)

    def test_the_two_markers_are_distinct(self) -> None:
        self.assertIsNot(EMPTY, DELETED)
        self.assertNotEqual(EMPTY, DELETED)

    def test_a_marker_is_not_a_key(self) -> None:
        self.assertNotEqual(EMPTY, "EMPTY")
        self.assertNotEqual(DELETED, None)

    def test_repr(self) -> None:
        self.assertEqual(repr(EMPTY), "EMPTY")
        self.assertEqual(repr(DELETED), "DELETED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
