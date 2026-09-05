# Python DSA Playground

A working repository for implementing classic data structures and algorithms from
scratch in **Python 3** — no library shortcuts, no copied solutions.

---

## Why this repo exists

I'm a Senior Lead Software Engineer. Day to day, that role is mostly architecture,
technical direction, code review, and mentorship — the leverage comes from judgment
rather than from writing tree traversals by hand. But the fundamentals are exactly
what that judgment is built on, and they atrophy quietly if you never exercise them.

This repo is deliberate practice against that decay:

- **Better technical judgment.** Choosing between a hash map, a balanced tree, and a
  heap is a design decision with real cost implications. Reasoning fluently about
  complexity, memory layout, and access patterns is what separates a defensible
  architecture from one that merely works today.
- **Sharper code review.** Recognizing an accidental O(n²) or a subtle off-by-one in
  someone else's pull request requires having written — and debugged — those patterns
  yourself.
- **Credible mentorship.** Engineers I support hit these problems in interviews and in
  production. Explaining *why* a red-black tree rebalances the way it does is far more
  useful than pointing at documentation.
- **Staying interview-ready.** Senior roles still assess fundamentals. Keeping this warm
  year-round beats cramming under pressure.

Continual growth at this level isn't about learning the newest framework. It's about
keeping the foundations sharp enough that the higher-order decisions come easily.

---

## Ground rules

The constraints are the point — they're what make this practice rather than review:

1. **Implement from first principles.** No `heapq`, no `collections.deque`, no
   `bisect` for the structure being built. The standard library is the thing being
   reimplemented, not the tool used to reimplement it.
2. **Type hints throughout.** Every public method carries annotations. Signatures
   should be self-documenting.
3. **Document the complexity.** Each operation states its time and space cost, and
   the reasoning behind it.
4. **Handle the edges deliberately.** Empty structures, single elements, duplicates,
   removing the root. The edge cases are where the real understanding lives.
5. **Test the behavior, not the implementation.** Tests should survive a rewrite of
   the internals.

---

## Repository structure

Each top-level folder is one self-contained data structure or algorithm family, with
its implementation, its tests, and its own notes. Folders are independent — there are
no cross-imports and no shared framework, so any one can be read in isolation.

```
Python-DSA-Playground/
├── BinarySearchTree/
│   ├── node.py                 # Node primitive
│   └── binary_search_tree.py   # BST operations and traversals
└── ...                         # further structures as they're built
```

---

## Progress

| Structure | Status | Key operations |
|---|---|---|
| Binary Search Tree | 🟡 In progress | insert, contains, delete, traversals, height, validation |
| Linked List | ⚪ Planned | singly and doubly linked, reversal, cycle detection |
| Stack & Queue | ⚪ Planned | array- and node-backed, min-stack |
| Hash Table | ⚪ Planned | separate chaining, open addressing, resize |
| Heap / Priority Queue | ⚪ Planned | sift up/down, heapify, k-largest |
| Graph | ⚪ Planned | BFS, DFS, topological sort, Dijkstra |
| Sorting | ⚪ Planned | merge, quick, heap, counting |
| Dynamic Programming | ⚪ Planned | memoization vs. tabulation, classic problems |

🟢 Complete · 🟡 In progress · ⚪ Planned

---

## Running the code

Requires Python 3.10 or newer — developed against 3.13. No third-party dependencies.

```bash
git clone git@github.com:edwinargueta/Python-DSA-Playground.git
cd Python-DSA-Playground/BinarySearchTree
python3 binary_search_tree.py
```

Each module includes a `__main__` block demonstrating its structure in use.

---

## Complexity reference

Target complexities for the operations implemented here:

| Structure | Access | Search | Insert | Delete | Space |
|---|---|---|---|---|---|
| Binary Search Tree (avg) | O(log n) | O(log n) | O(log n) | O(log n) | O(n) |
| Binary Search Tree (worst) | O(n) | O(n) | O(n) | O(n) | O(n) |
| Hash Table | — | O(1) | O(1) | O(1) | O(n) |
| Heap | O(1) peek | O(n) | O(log n) | O(log n) | O(n) |

The gap between a BST's average and worst case is the entire argument for
self-balancing trees — an unbalanced BST fed sorted input degrades into a linked list.
