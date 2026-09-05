# Working in this repo

Notes for Claude. This repo is deliberate practice: the owner implements every
structure by hand. Read [README.md](README.md) for the ground rules — the two
that constrain you most are *implement from first principles* (no `heapq`, no
`collections.deque`, no `bisect` for the structure being built) and *test the
behavior, not the implementation*.

## The one rule that matters most

**Never implement a method the owner has not asked you to implement.** Stubs
raise `NotImplementedError` on purpose. Writing the solution destroys the point
of the exercise. Write stubs, write tests, explain the failure — but leave the
body alone unless explicitly asked.

## Folder layout

Each top-level folder is one self-contained structure. No cross-imports, no
shared framework, no package `__init__.py` — imports are flat, which means
**tests only run from inside their own folder**:

```
StructureName/
├── node.py              # the node primitive, if the structure needs one
├── structure_name.py    # the class, snake_case filename
└── test_xyz.py          # the suite (test_bst.py, test_sll.py)
```

```bash
cd StructureName
python3 -m unittest test_xyz -v      # or: python3 test_xyz.py
```

Every implementation file ends with a `__main__` block demonstrating the
structure in use.

## Stub conventions

**One line of docstring per method. No more.** A stub is a signature and a
contract, not a lesson — long docstrings crowd out the code, and hints about
which case is tricky are exactly the thinking the owner is here to do.

Keep the summary, the return or raise contract, and the cost. Drop everything
else: `Args:`/`Returns:`/`Raises:` blocks, worked examples, algorithm sketches,
"remember to..." reminders, and any explanation of *why* a case is hard.

```python
def pop(self) -> Any:
    """Remove and return the tail; IndexError if empty. O(n) time, O(1) space."""
    raise NotImplementedError
```

Not this:

```python
def pop(self) -> Any:
    """Remove and return the value at the tail.

    Raises:
        IndexError: if the list is empty.

    This is the operation a singly linked list is bad at: holding tail does
    not help, because you need the node before it and there are no backward
    pointers.

    Time:  O(n)
    Space: O(1)
    """
    raise NotImplementedError
```

Fitting the cost onto that line keeps README ground rule 3 satisfied — use the
compact `O(n) time, O(1) space` form rather than `Time:`/`Space:` lines. Trim
the prose until the whole line fits the repo's 86-character norm; if it will not
fit, the summary is still doing too much explaining.

The **class** docstring is the exception, and the place for anything structural:
the ordering invariant, what happens to duplicates, whether indices may be
negative. Facts that hold across every method belong there once, not restated in
each stub.

Type hints on every signature. `from __future__ import annotations` at the top.

---

# How to write a test suite

The goal is that **a green test class means that method is done** — no more, no
less. A suite that fails for the wrong reasons is worse than no suite, because
it destroys the progress signal the owner is working against.

## 1. One class per method, and it fails only on that method

Group tests into a class per method (`TestPop`, `TestReverse`). A test in
`TestPop` must fail *only* when `pop` is wrong. If it also calls `to_list()` to
check the result, then `TestPop` goes red because `to_list` is unwritten, and
the owner is now debugging the wrong method.

This is the mistake to avoid, from the original BST suite:

```python
# WRONG - a test for insert that fails because in_order is unwritten
def test_maintains_ordering(self) -> None:
    bst = build([50, 30, 70, 20, 40])
    self.assertEqual(bst.in_order(), [20, 30, 40, 50, 70])
```

## 2. Give the tests their own way to see inside

The fix is a **test-side oracle**: a module-level helper that inspects the
structure by walking its fields directly, so tests never borrow a method to
verify another method.

```python
# BST                                         # SinglyLinkedList
walk_in_order(node) -> list                   walk(head) -> list
is_bst(node, low, high) -> bool               last_node(head) -> Node
```

```python
def test_maintains_ordering(self) -> None:
    bst = build([50, 30, 70, 20, 40])
    self.assertEqual(walk_in_order(bst.root), [20, 30, 40, 50, 70])
```

Yes, these touch `.left`, `.next`, `._size`. That is fine and deliberate — the
*helpers* read internals, the *assertions* still describe behavior, so a
recursive and an iterative solution both pass. Give any walking helper a
`max_nodes` guard if the structure can be wired into a cycle, or a broken
implementation hangs the suite instead of failing it.

## 3. Build fixtures without the implementation

`build()` must construct the structure by wiring nodes directly, not by calling
`append()` or `insert()`. Otherwise every class in the file depends on that one
method, and nothing can be implemented out of order.

```python
def build(values) -> SinglyLinkedList:
    """Wire the nodes directly so fixtures work before any method exists."""
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
```

The BST suite is the exception worth knowing: its `build()` calls `insert()`,
because wiring a tree by hand is more error-prone than the thing it protects
against. Trade that off per structure, and say which you chose in the docstring.

## 4. Assert the invariants, not just the values

Most bugs in these structures are a stale pointer or a size counter that
drifted — not a wrong value. One helper, called after every mutation, catches
the whole class of them:

```python
def assert_intact(case, sll, expected) -> None:
    case.assertEqual(walk(sll.head), list(expected), "node chain is wrong")
    case.assertEqual(len(sll), len(expected), "_size is out of step")
    case.assertIs(sll.tail, last_node(sll.head), "tail is not the final node")
    ...
```

Every message says what broke. `assertEqual(x, y)` with no message makes the
owner read the test to find out what it meant.

## 5. Document the exceptions instead of hiding them

Some coupling is real: `__iter__` delegating to `in_order` *is* the behavior
under test. Keep those, and say so in the class docstring and the module
docstring, so a red test is never a surprise:

```python
class TestDunders(unittest.TestCase):
    """__contains__ and __getitem__ delegate by design, so they stay red
    until contains() and get() are written."""
```

## 6. Cover the cases the owner is practicing

Ground rule 4 is *handle the edges deliberately*, so the suite has to exercise
them. Always include a `TestEmptyX` class with one test per method — the empty
case is the one most implementations get wrong. Then per method:

- **Boundaries** — first and last index, off-by-one on each side, `len` exactly.
- **Sizes** — empty, one element, two elements, many. One and two elements are
  where head/tail aliasing breaks.
- **The structural cases** — for delete: leaf, one child, two children, and each
  again at the root. For a list: head, middle, tail, only node.
- **Failure paths** — assert the return value or exception *and* that the
  structure was left unchanged.
- **Duplicates** — whether the structure stores them (linked list) or rejects
  them (BST), test the choice.
- **Round trips** — reverse twice, delete then reinsert, drain then refill.

## 7. Verify the suite before handing it over

A test suite is code, and an unverified one is worse than useless here. Before
saying it is ready:

1. Write a throwaway reference implementation in the scratchpad, copy the suite
   next to it, and confirm **every test passes**. A test that cannot pass will
   read as the owner's bug.
2. Run each class against the stubs and confirm it is blocked only by its own
   method — this catches accidental coupling.
3. Delete the reference implementation. Never commit it, never leave it where it
   could be pasted into the real file.

## Checklist

- [ ] One class per method; each fails only on its own method
- [ ] Test-side oracle helpers; no method used to verify another
- [ ] `build()` does not depend on the implementation (or the docstring says why)
- [ ] Invariant helper called after every mutation, with failure messages
- [ ] Documented delegation exceptions
- [ ] Empty / one / two / many, boundaries, failure paths, duplicates
- [ ] Stub docstrings are one line each, cost included, class docstring
      carries the invariants
- [ ] Module docstring: how to run it, and what is deliberately coupled
- [ ] Verified green against a scratchpad reference, which is then deleted
