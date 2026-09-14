"""Behavior tests for HashTable, the separate chaining table.

Run from inside this directory:

    python3 -m unittest test_hash_table -v
    python3 test_hash_table.py            # equivalent

Each test class depends only on the method it names, so you can implement
the methods in any order and a green class always means that method is done.

There is one unavoidable exception, and it is the reason to write _hash and
_bucket_index first. A fixture cannot wire an entry into a bucket without
knowing which bucket the table will look in, so build() and assert_intact()
both call _bucket_index(). Until it works, every class in this file errors -
the same trade the BST suite makes when its build() calls insert(). Nothing
else is borrowed: entries() and chain_of() read .next, .key and .value
straight off the nodes, so a test for remove() never calls get() to see
what happened.

Two things are deliberately not asserted, because they are yours to choose:
where a new entry lands in its chain, and the order keys come back in. Every
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

from hash_table import HashTable
from node import Node

# The table used throughout: four keys, comfortably inside a capacity of 8.
BASE = [("apple", 1), ("banana", 2), ("cherry", 3), ("date", 4)]
BASE_DICT = dict(BASE)


def build(pairs, capacity: int = 8, max_load: float = 0.75) -> HashTable:
    """Return a table holding pairs, wiring the chains by hand.

    Deliberately avoids put() so that fixtures work before any method is
    implemented, and so a broken put() cannot fail unrelated classes. It
    does call _bucket_index(), because an entry has to go in the bucket the
    table will look in - see the module docstring.

    It never resizes, so a fixture may sit above its own load factor.
    """
    table = HashTable(capacity, max_load)
    for key, value in pairs:
        index = table._bucket_index(key)
        node = Node(key, value)
        node.next = table._buckets[index]
        table._buckets[index] = node
        table._size += 1
    return table


def chain_of(table, index: int, max_nodes: int = 1000) -> list:
    """Collect (key, value) along one bucket's chain."""
    found = []
    node = table._buckets[index]
    while node is not None:
        if len(found) > max_nodes:
            raise AssertionError("walked past max_nodes - a chain has a cycle")
        found.append((node.key, node.value))
        node = node.next
    return found


def entries(table) -> list:
    """Collect (key, value) from every chain in the table."""
    found = []
    for index in range(len(table._buckets)):
        found.extend(chain_of(table, index))
    return found


def colliding_keys(count: int = 3, capacity: int = 8, pool: int = 500) -> list:
    """Return count keys that _bucket_index sends to one shared bucket.

    Found by asking the table itself, so it works whatever hash you write -
    and by the pigeonhole principle a bucket has to repeat long before the
    pool runs out.
    """
    probe = HashTable(capacity)
    groups: dict = {}
    for i in range(pool):
        key = f"k{i}"
        group = groups.setdefault(probe._bucket_index(key), [])
        group.append(key)
        if len(group) == count:
            return group
    raise AssertionError(
        f"no {count} of {pool} keys shared a bucket - is _bucket_index in range?"
    )


def printed(call) -> str:
    """Run call and return everything it wrote to stdout."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call()
    return buffer.getvalue()


def assert_intact(case, table, expected) -> None:
    """Assert the table holds exactly expected and its invariants still hold.

    Checks the stored pairs, that no key is stored twice, that _size agrees
    with the chains, and that every entry sits in the bucket its own hash
    chooses. A key stranded in the wrong bucket after a resize is the
    classic bug here, and nothing but a placement check catches it - the
    entry is still reachable by walking, just never found by a lookup.
    """
    expected = dict(expected)
    found = entries(table)
    keys = [key for key, _ in found]
    case.assertEqual(
        len(keys), len(set(keys)), f"a key is stored more than once: {keys}"
    )
    case.assertEqual(dict(found), expected, "the stored entries are wrong")
    case.assertEqual(len(table), len(expected), "_size is out of step")
    case.assertGreaterEqual(len(table._buckets), 1, "the table has no buckets left")
    for index in range(len(table._buckets)):
        for key, _ in chain_of(table, index):
            case.assertEqual(
                table._bucket_index(key),
                index,
                f"{key!r} sits in bucket {index}, not the one its hash chooses",
            )


class TestHash(unittest.TestCase):
    """The foundation - build() and assert_intact() both need _bucket_index,
    so every other class in this file stays red until these two work."""

    def setUp(self) -> None:
        self.table = HashTable()

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

    def test_different_tables_agree(self) -> None:
        self.assertEqual(HashTable(4)._hash("apple"), HashTable(64)._hash("apple"))


class TestBucketIndex(unittest.TestCase):

    def test_is_in_range(self) -> None:
        table = HashTable(8)
        for i in range(200):
            index = table._bucket_index(f"k{i}")
            self.assertGreaterEqual(index, 0, f"k{i} indexed below zero")
            self.assertLess(index, 8, f"k{i} indexed past the last bucket")

    def test_is_in_range_for_one_bucket(self) -> None:
        table = HashTable(1)
        for i in range(50):
            self.assertEqual(table._bucket_index(f"k{i}"), 0)

    def test_is_in_range_for_mixed_key_types(self) -> None:
        table = HashTable(8)
        for key in ["apple", "", 0, 7, -7, 3.5, (1, 2), None, True]:
            self.assertIn(table._bucket_index(key), range(8), f"{key!r} out of range")

    def test_is_deterministic(self) -> None:
        table = HashTable(8)
        self.assertEqual(table._bucket_index("apple"), table._bucket_index("apple"))

    def test_equal_keys_share_a_bucket(self) -> None:
        table = HashTable(8)
        self.assertEqual(
            table._bucket_index("apple"), table._bucket_index("app" + "le")
        )

    def test_follows_the_capacity(self) -> None:
        small, large = HashTable(4), HashTable(64)
        self.assertIn(small._bucket_index("apple"), range(4))
        self.assertIn(large._bucket_index("apple"), range(64))

    def test_spreads_keys_over_more_than_one_bucket(self) -> None:
        table = HashTable(64)
        used = {table._bucket_index(f"k{i}") for i in range(200)}
        self.assertGreater(
            len(used), 1, "every key landed in one bucket - that is a linked list"
        )


class TestEmptyTable(unittest.TestCase):
    """An empty table is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.table = HashTable()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.table), 0)

    def test_every_bucket_is_empty(self) -> None:
        self.assertEqual(entries(self.table), [])

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

    def test_bucket_sizes_are_all_zero(self) -> None:
        self.assertEqual(self.table.bucket_sizes(), [0] * 8)

    def test_resize_keeps_it_empty(self) -> None:
        self.table._resize(16)
        assert_intact(self, self.table, {})
        self.assertEqual(len(self.table._buckets), 16)

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
        self.assertEqual(repr(self.table), "HashTable({})")


class TestPut(unittest.TestCase):

    def test_stores_the_first_entry(self) -> None:
        table = HashTable()
        table.put("apple", 1)
        assert_intact(self, table, {"apple": 1})

    def test_stores_several_keys(self) -> None:
        table = HashTable()
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

    def test_stores_none_as_a_value(self) -> None:
        table = HashTable()
        table.put("apple", None)
        assert_intact(self, table, {"apple": None})

    def test_keys_sharing_a_bucket_all_survive(self) -> None:
        keys = colliding_keys(3)
        table = HashTable(8)
        for i, key in enumerate(keys):
            table.put(key, i)
        assert_intact(self, table, {key: i for i, key in enumerate(keys)})
        index = table._bucket_index(keys[0])
        self.assertEqual(len(chain_of(table, index)), 3, "the chain lost an entry")

    def test_overwrites_inside_a_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        for key in keys:
            table.put(key, "new")
        assert_intact(self, table, {key: "new" for key in keys})

    def test_accepts_mixed_key_types(self) -> None:
        pairs = {"apple": 1, 7: 2, (1, 2): 3, None: 4, 3.5: 5}
        table = HashTable()
        for key, value in pairs.items():
            table.put(key, value)
        assert_intact(self, table, pairs)

    def test_size_tracks_every_new_key(self) -> None:
        table = HashTable()
        for i in range(50):
            table.put(f"k{i}", i)
            self.assertEqual(len(table), i + 1, f"size wrong after {i + 1} puts")

    def test_many_keys_all_survive(self) -> None:
        table = HashTable()
        expected = {f"k{i}": i for i in range(200)}
        for key, value in expected.items():
            table.put(key, value)
        assert_intact(self, table, expected)


class TestAutomaticGrowth(unittest.TestCase):
    """put() grows the table on its own, so this class needs put() and
    whatever put() resizes with - the capacity it grows to is yours."""

    def test_grows_before_the_load_factor_is_exceeded(self) -> None:
        table = HashTable(capacity=8, max_load=0.75)
        for i in range(7):
            table.put(f"k{i}", i)
        self.assertGreater(len(table._buckets), 8, "the table never grew")
        self.assertLessEqual(
            len(table) / len(table._buckets), 0.75, "load factor is above max_load"
        )

    def test_stays_under_the_load_factor_throughout(self) -> None:
        table = HashTable(capacity=8, max_load=0.75)
        for i in range(200):
            table.put(f"k{i}", i)
            self.assertLessEqual(
                len(table) / len(table._buckets),
                0.75,
                f"load factor is above max_load after {i + 1} puts",
            )

    def test_every_entry_is_rehomed(self) -> None:
        table = HashTable(capacity=4, max_load=0.75)
        expected = {f"k{i}": i for i in range(100)}
        for key, value in expected.items():
            table.put(key, value)
        assert_intact(self, table, expected)

    def test_overwrites_never_grow_the_table(self) -> None:
        table = HashTable(capacity=8, max_load=0.75)
        for i in range(50):
            table.put("apple", i)
        self.assertEqual(len(table._buckets), 8, "an overwrite grew the table")

    def test_a_generous_load_factor_is_respected(self) -> None:
        table = HashTable(capacity=8, max_load=4.0)
        for i in range(8):
            table.put(f"k{i}", i)
        self.assertEqual(len(table._buckets), 8, "the table grew before it had to")


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

    def test_finds_every_key_in_a_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        for i, key in enumerate(keys):
            self.assertEqual(table.get(key), i, f"{key!r} was lost in its chain")

    def test_absent_key_in_an_occupied_bucket(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        self.assertIsNone(table.get(keys[3]), "matched the bucket, not the key")

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

    def test_finds_every_key_in_a_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        for key in keys:
            self.assertTrue(table.contains(key), f"{key!r} was lost in its chain")

    def test_rejects_an_absent_key_in_an_occupied_bucket(self) -> None:
        keys = colliding_keys(4)
        table = build([(key, i) for i, key in enumerate(keys[:3])])
        self.assertFalse(table.contains(keys[3]), "matched the bucket, not the key")


class TestRemove(unittest.TestCase):

    def test_removes_the_only_entry_in_its_bucket(self) -> None:
        table = build(BASE)
        self.assertTrue(table.remove("banana"))
        assert_intact(self, table, {k: v for k, v in BASE if k != "banana"})

    def test_removes_each_key_from_a_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        pairs = [(key, i) for i, key in enumerate(keys)]
        for target in keys:
            table = build(pairs)
            self.assertTrue(table.remove(target), f"{target!r} was not removed")
            assert_intact(self, table, {k: v for k, v in pairs if k != target})

    def test_removes_the_last_entry(self) -> None:
        table = build([("apple", 1)])
        self.assertTrue(table.remove("apple"))
        assert_intact(self, table, {})

    def test_returns_false_when_absent(self) -> None:
        table = build(BASE)
        self.assertFalse(table.remove("fig"))
        assert_intact(self, table, BASE_DICT)

    def test_absent_key_in_an_occupied_bucket_changes_nothing(self) -> None:
        keys = colliding_keys(4)
        pairs = [(key, i) for i, key in enumerate(keys[:3])]
        table = build(pairs)
        self.assertFalse(table.remove(keys[3]), "matched the bucket, not the key")
        assert_intact(self, table, dict(pairs))

    def test_removing_twice_returns_false(self) -> None:
        table = build(BASE)
        self.assertTrue(table.remove("apple"))
        self.assertFalse(table.remove("apple"))
        assert_intact(self, table, {k: v for k, v in BASE if k != "apple"})

    def test_drains_every_key(self) -> None:
        table = build(BASE)
        for key, _ in BASE:
            self.assertTrue(table.remove(key), f"{key!r} was not removed")
        assert_intact(self, table, {})

    def test_reinserting_after_removal(self) -> None:
        table = build(BASE)
        table.remove("apple")
        index = table._bucket_index("apple")
        node = Node("apple", 99)
        node.next = table._buckets[index]
        table._buckets[index] = node
        table._size += 1
        assert_intact(self, table, {**BASE_DICT, "apple": 99})

    def test_does_not_shrink_the_capacity(self) -> None:
        table = build(BASE)
        for key, _ in BASE:
            table.remove(key)
        self.assertEqual(len(table._buckets), 8, "removing shrank the table")


class TestPop(unittest.TestCase):

    def test_returns_the_value_and_removes_the_key(self) -> None:
        table = build(BASE)
        self.assertEqual(table.pop("banana"), 2)
        assert_intact(self, table, {k: v for k, v in BASE if k != "banana"})

    def test_pops_each_key_from_a_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        pairs = [(key, i) for i, key in enumerate(keys)]
        for i, target in enumerate(keys):
            table = build(pairs)
            self.assertEqual(table.pop(target), i, f"wrong value for {target!r}")
            assert_intact(self, table, {k: v for k, v in pairs if k != target})

    def test_returns_a_stored_none(self) -> None:
        table = build([("apple", None)])
        self.assertIsNone(table.pop("apple"))
        assert_intact(self, table, {})

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

    def test_keeps_the_capacity(self) -> None:
        table = build(BASE, capacity=16)
        table.clear()
        self.assertEqual(len(table._buckets), 16, "clear() changed the capacity")

    def test_the_table_is_reusable(self) -> None:
        table = build(BASE)
        table.clear()
        index = table._bucket_index("fig")
        node = Node("fig", 9)
        node.next = table._buckets[index]
        table._buckets[index] = node
        table._size += 1
        assert_intact(self, table, {"fig": 9})

    def test_on_an_empty_table(self) -> None:
        table = HashTable()
        table.clear()
        assert_intact(self, table, {})


class TestKeys(unittest.TestCase):

    def test_returns_every_key(self) -> None:
        self.assertEqual(sorted(build(BASE).keys()), sorted(BASE_DICT))

    def test_returns_keys_sharing_a_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        self.assertEqual(sorted(table.keys()), sorted(keys))

    def test_is_empty_for_an_empty_table(self) -> None:
        self.assertEqual(HashTable().keys(), [])

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

    def test_is_empty_for_an_empty_table(self) -> None:
        self.assertEqual(HashTable().values(), [])

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        table.values()
        assert_intact(self, table, BASE_DICT)


class TestItems(unittest.TestCase):

    def test_returns_every_pair(self) -> None:
        self.assertEqual(sorted(build(BASE).items()), sorted(BASE))

    def test_returns_pairs_sharing_a_bucket(self) -> None:
        keys = colliding_keys(3)
        pairs = [(key, i) for i, key in enumerate(keys)]
        self.assertEqual(sorted(build(pairs).items()), sorted(pairs))

    def test_is_empty_for_an_empty_table(self) -> None:
        self.assertEqual(HashTable().items(), [])

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

    def test_prints_keys_sharing_a_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        output = printed(table.print_table)
        for key in keys:
            self.assertIn(key, output, f"{key!r} was hidden behind its chain")

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        printed(table.print_table)
        assert_intact(self, table, BASE_DICT)


class TestCapacity(unittest.TestCase):

    def test_default(self) -> None:
        self.assertEqual(HashTable().capacity(), 8)

    def test_custom(self) -> None:
        self.assertEqual(HashTable(32).capacity(), 32)

    def test_tracks_the_buckets(self) -> None:
        table = HashTable(8)
        table._buckets = [None] * 64
        self.assertEqual(table.capacity(), 64, "capacity() is not reading _buckets")


class TestLoadFactor(unittest.TestCase):

    def test_is_zero_when_empty(self) -> None:
        self.assertEqual(HashTable(8).load_factor(), 0.0)

    def test_is_entries_over_buckets(self) -> None:
        self.assertEqual(build(BASE, capacity=8).load_factor(), 0.5)

    def test_follows_the_capacity(self) -> None:
        self.assertEqual(build(BASE, capacity=4).load_factor(), 1.0)

    def test_can_exceed_one(self) -> None:
        table = build(BASE, capacity=2)
        self.assertEqual(table.load_factor(), 2.0, "chaining allows more than one")


class TestBucketSizes(unittest.TestCase):

    def test_one_entry_per_bucket_used(self) -> None:
        table = build(BASE, capacity=8)
        self.assertEqual(len(table.bucket_sizes()), 8, "one number per bucket")
        self.assertEqual(sum(table.bucket_sizes()), 4)

    def test_all_zero_when_empty(self) -> None:
        self.assertEqual(HashTable(8).bucket_sizes(), [0] * 8)

    def test_counts_a_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        sizes = table.bucket_sizes()
        self.assertEqual(max(sizes), 3, "the shared chain was not counted")
        self.assertEqual(sum(sizes), 3)

    def test_matches_the_chains(self) -> None:
        table = build(BASE, capacity=8)
        expected = [len(chain_of(table, i)) for i in range(8)]
        self.assertEqual(table.bucket_sizes(), expected)


class TestResize(unittest.TestCase):

    def test_changes_the_capacity(self) -> None:
        table = build(BASE)
        table._resize(32)
        self.assertEqual(len(table._buckets), 32)

    def test_keeps_every_entry(self) -> None:
        table = build(BASE)
        table._resize(32)
        assert_intact(self, table, BASE_DICT)

    def test_shrinking_keeps_every_entry(self) -> None:
        table = build(BASE, capacity=32)
        table._resize(2)
        assert_intact(self, table, BASE_DICT)

    def test_to_one_bucket(self) -> None:
        table = build(BASE)
        table._resize(1)
        assert_intact(self, table, BASE_DICT)
        self.assertEqual(len(chain_of(table, 0)), 4, "one bucket holds them all")

    def test_on_an_empty_table(self) -> None:
        table = HashTable(8)
        table._resize(16)
        assert_intact(self, table, {})
        self.assertEqual(len(table._buckets), 16)

    def test_unpacks_a_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)], capacity=8)
        table._resize(64)
        assert_intact(self, table, {key: i for i, key in enumerate(keys)})

    def test_many_entries_survive(self) -> None:
        expected = {f"k{i}": i for i in range(100)}
        table = build(list(expected.items()), capacity=8)
        table._resize(256)
        assert_intact(self, table, expected)

    def test_size_is_unchanged(self) -> None:
        table = build(BASE)
        table._resize(32)
        self.assertEqual(len(table), 4, "_size drifted during the rehome")


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it.
    Chains here always terminate - nothing in this class builds a cycle."""

    def test_empty_table(self) -> None:
        self.assertTrue(HashTable().is_valid())

    def test_populated_table(self) -> None:
        self.assertTrue(build(BASE).is_valid())

    def test_shared_bucket(self) -> None:
        keys = colliding_keys(3)
        self.assertTrue(build([(key, i) for i, key in enumerate(keys)]).is_valid())

    def test_size_too_high(self) -> None:
        table = build(BASE)
        table._size += 1
        self.assertFalse(table.is_valid())

    def test_size_too_low(self) -> None:
        table = build(BASE)
        table._size -= 1
        self.assertFalse(table.is_valid())

    def test_entry_in_the_wrong_bucket(self) -> None:
        table = build(BASE)
        wrong = (table._bucket_index("fig") + 1) % len(table._buckets)
        node = Node("fig", 9)
        node.next = table._buckets[wrong]
        table._buckets[wrong] = node
        table._size += 1
        self.assertFalse(table.is_valid(), "a key no lookup can reach is valid")

    def test_duplicate_key_in_one_chain(self) -> None:
        table = build(BASE)
        index = table._bucket_index("apple")
        node = Node("apple", 99)
        node.next = table._buckets[index]
        table._buckets[index] = node
        table._size += 1
        self.assertFalse(table.is_valid(), "the same key is stored twice")


class TestDelegatingDunders(unittest.TestCase):
    """__len__ reads _size directly, but __contains__ delegates to
    contains() and __setitem__ to put(), so those stay red until their
    methods are written."""

    def test_len(self) -> None:
        self.assertEqual(len(build(BASE)), 4)
        self.assertEqual(len(HashTable()), 0)

    def test_in_operator(self) -> None:
        table = build(BASE)
        self.assertIn("apple", table)
        self.assertNotIn("fig", table)

    def test_assignment(self) -> None:
        table = HashTable()
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
        assert_intact(self, table, {k: v for k, v in BASE if k != "banana"})

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
        self.assertEqual(list(HashTable()), [])

    def test_yields_keys_sharing_a_bucket(self) -> None:
        keys = colliding_keys(3)
        table = build([(key, i) for i, key in enumerate(keys)])
        self.assertEqual(sorted(table), sorted(keys))

    def test_restarts_on_each_pass(self) -> None:
        table = build(BASE)
        self.assertEqual(sorted(table), sorted(table), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        table = build(BASE)
        list(table)
        assert_intact(self, table, BASE_DICT)


class TestRepr(unittest.TestCase):

    def test_one_entry(self) -> None:
        self.assertEqual(repr(build([("a", 1)])), "HashTable({'a': 1})")

    def test_empty_table(self) -> None:
        self.assertEqual(repr(HashTable()), "HashTable({})")

    def test_quotes_string_values(self) -> None:
        self.assertEqual(repr(build([("a", "b")])), "HashTable({'a': 'b'})")


class TestNode(unittest.TestCase):

    def test_starts_unlinked(self) -> None:
        node = Node("apple", 1)
        self.assertEqual(node.key, "apple")
        self.assertEqual(node.value, 1)
        self.assertIsNone(node.next)

    def test_repr(self) -> None:
        self.assertEqual(repr(Node("apple", 1)), "Node('apple': 1)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
