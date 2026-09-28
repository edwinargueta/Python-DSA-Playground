"""Behavior tests for Graph, the adjacency-map graph.

Run from inside this directory:

    python3 -m unittest test_graph -v
    python3 test_graph.py            # equivalent

Each test class depends only on the method it names, so you can implement
the methods in any order and a green class always means that method is done.

build() writes _adj and _edge_count by hand, so every fixture exists before
a single method is written and a broken add_edge() cannot fail an unrelated
class. assert_intact() then compares the whole of _adj against the
adjacency the test expects, which is why a test for remove_vertex() never
calls has_edge() to see what happened.

The traversal and path tests do not pin one answer, because neighbor order
is unspecified. is_bfs_order() and is_dfs_order() replay a search that
tries neighbors in the order your result visited them, and accept the
result only if the replay reproduces it - so every correct order passes and
no incorrect one does. A shortest path is accepted if each step is an edge
and it is as short as hops() says it should be. dijkstra() is checked
against cheapest(), a Bellman-Ford oracle that shares none of its logic.

The deliberate couplings: __len__ and edge_count() read fields directly,
and __contains__ delegates to has_vertex(), so it stays red until
has_vertex() is written. TestRoundTrip drives the mutating methods
together and needs add_vertex(), add_edge(), remove_edge(), remove_vertex()
and is_valid().
"""

from __future__ import annotations

import unittest
from collections import Counter

from graph import Graph

# Shapes used throughout. An edge is (u, v) with weight 1, or (u, v, weight).
PATH = [(1, 2), (2, 3), (3, 4)]
TREE = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]
DIAMOND = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
TRIANGLE = [("A", "B"), ("B", "C"), ("C", "A")]
# From A: C is 1, B is 3 by way of C, D is 4 by way of C and B.
WEIGHTED = [("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 1), ("C", "D", 5)]
# Each of A's neighbors is also reachable deeper down, through the other.
DEEP = [
    ("A", "B"), ("A", "C"), ("B", "E"), ("B", "F"),
    ("E", "C"), ("C", "G"), ("G", "B"),
]


def split(edge) -> tuple:
    """Return (u, v, weight) for an edge written as (u, v) or (u, v, weight)."""
    return (edge[0], edge[1], edge[2] if len(edge) == 3 else 1)


def adjacency(edges=(), vertices=(), directed=False) -> dict:
    """Return the _adj of a graph holding exactly these edges and vertices."""
    adj = {}
    for vertex in vertices:
        adj.setdefault(vertex, {})
    for edge in edges:
        u, v, weight = split(edge)
        adj.setdefault(u, {})[v] = weight
        adj.setdefault(v, {})
        if not directed:
            adj[v][u] = weight
    return adj


def build(edges=(), vertices=(), directed=False) -> Graph:
    """Return a graph holding edges plus any extra, isolated vertices.

    Writes _adj and _edge_count by hand so that fixtures work before any
    method is implemented, and so a broken add_edge() cannot fail unrelated
    classes.
    """
    graph = Graph(directed=directed)
    graph._adj = adjacency(edges, vertices, directed)
    graph._edge_count = len(edges)
    return graph


def grid(n: int, weighted: bool = False) -> list:
    """Return the edges of an n x n grid whose vertices are (row, col)."""
    edges = []
    for r in range(n):
        for c in range(n):
            weight = (r * 7 + c * 3) % 5 + 1 if weighted else 1
            if r + 1 < n:
                edges.append(((r, c), (r + 1, c), weight))
            if c + 1 < n:
                edges.append(((r, c), (r, c + 1), weight + 1))
    return edges


def reachable(adj, start) -> set:
    """Return every vertex reachable from start."""
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def hops(adj, start) -> dict:
    """Return the fewest edges from start to each reachable vertex."""
    dist, frontier, i = {start: 0}, [start], 0
    while i < len(frontier):
        u = frontier[i]
        i += 1
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                frontier.append(v)
    return dist


def cheapest(adj, start) -> dict:
    """Return the cheapest distance from start to each reachable vertex.

    Bellman-Ford: relax every edge until nothing improves. Slow, and it
    shares no logic with Dijkstra, which is what makes it an oracle.
    """
    dist, changed = {start: 0}, True
    while changed:
        changed = False
        for u in list(dist):
            for v, weight in adj[u].items():
                if v not in dist or dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    changed = True
    return dist


def visits_each_reachable_once(adj, start, order) -> bool:
    """Return True if order holds every vertex reachable from start, once."""
    try:
        return len(order) == len(set(order)) and set(order) == reachable(adj, start)
    except TypeError:
        return False


def is_bfs_order(adj, start, order) -> bool:
    """Return True if some breadth-first search from start visits in order.

    Replays a BFS that tries each vertex's neighbors in the order they
    appear in order itself. A correct BFS result survives the replay intact.
    """
    if not visits_each_reachable_once(adj, start, order):
        return False
    rank = {v: i for i, v in enumerate(order)}
    replay, seen, i = [start], {start}, 0
    while i < len(replay):
        u = replay[i]
        i += 1
        for v in sorted(adj[u], key=rank.__getitem__):
            if v not in seen:
                seen.add(v)
                replay.append(v)
    return replay == list(order)


def is_dfs_order(adj, start, order) -> bool:
    """Return True if some depth-first search from start visits in order.

    Replays a DFS that tries each vertex's neighbors in the order they
    appear in order itself. A correct DFS result survives the replay intact.
    """
    if not visits_each_reachable_once(adj, start, order):
        return False
    rank = {v: i for i, v in enumerate(order)}
    replay, seen = [], set()

    def visit(u) -> None:
        seen.add(u)
        replay.append(u)
        for v in sorted(adj[u], key=rank.__getitem__):
            if v not in seen:
                visit(v)

    visit(start)
    return replay == list(order)


def is_topological(adj, order) -> bool:
    """Return True if order holds every vertex once and every edge points forward."""
    if len(order) != len(adj) or set(order) != set(adj):
        return False
    rank = {v: i for i, v in enumerate(order)}
    return all(rank[u] < rank[v] for u in adj for v in adj[u])


def assert_intact(case, graph, edges=(), vertices=()) -> None:
    """Assert the graph holds exactly these edges and vertices, and is intact.

    Compares all of _adj, so a missing reverse entry, a neighbor that is not
    a vertex, a stale edge or a wrong weight all fail here. Then checks
    _edge_count against the length of the edge list.
    """
    expected = adjacency(edges, vertices, graph._directed)
    case.assertEqual(graph._adj, expected, "_adj does not hold the expected edges")
    case.assertEqual(
        graph._edge_count, len(edges), "_edge_count is out of step with the edges"
    )


def assert_bfs(case, graph, start) -> None:
    order = graph.bfs(start)
    case.assertTrue(
        is_bfs_order(graph._adj, start, order),
        f"{order!r} is not a breadth-first order from {start!r}",
    )


def assert_dfs(case, graph, start) -> None:
    order = graph.dfs(start)
    case.assertTrue(
        is_dfs_order(graph._adj, start, order),
        f"{order!r} is not a depth-first order from {start!r}",
    )


def assert_shortest_path(case, graph, u, v) -> None:
    """Assert shortest_path() runs u to v along edges, using the fewest."""
    path = graph.shortest_path(u, v)
    case.assertIsNotNone(path, f"no path found from {u!r} to {v!r}")
    case.assertEqual(path[0], u, f"{path!r} does not start at {u!r}")
    case.assertEqual(path[-1], v, f"{path!r} does not end at {v!r}")
    for a, b in zip(path, path[1:]):
        case.assertIn(b, graph._adj.get(a, {}), f"{a!r} -> {b!r} is not an edge")
    case.assertEqual(
        len(path) - 1, hops(graph._adj, u)[v], f"{path!r} is not a shortest path"
    )


def assert_topological(case, graph) -> None:
    order = graph.topological_sort()
    case.assertTrue(
        is_topological(graph._adj, order), f"{order!r} is not a topological order"
    )


def assert_components(case, graph, expected) -> None:
    """Assert connected_components() returns expected, in any order.

    Checks the groups as sets, then that no vertex was listed twice.
    """
    result = graph.connected_components()
    case.assertEqual(
        {frozenset(group) for group in result},
        {frozenset(group) for group in expected},
        "the components are wrong",
    )
    case.assertEqual(
        Counter(v for group in result for v in group),
        Counter(v for group in expected for v in group),
        "a vertex appears in more than one component, or twice in one",
    )


class TestEmptyGraph(unittest.TestCase):
    """An empty graph is the edge case most implementations get wrong.

    One test per method, so a failure names the method that is unwritten.
    """

    def setUp(self) -> None:
        self.graph = Graph()

    def test_len_is_zero(self) -> None:
        self.assertEqual(len(self.graph), 0)

    def test_edge_count_is_zero(self) -> None:
        self.assertEqual(self.graph.edge_count(), 0)

    def test_add_vertex_adds_the_first(self) -> None:
        self.assertTrue(self.graph.add_vertex("A"))
        assert_intact(self, self.graph, vertices=["A"])

    def test_add_edge_creates_both_ends(self) -> None:
        self.graph.add_edge("A", "B")
        assert_intact(self, self.graph, [("A", "B")])

    def test_remove_edge_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.remove_edge("A", "B")
        assert_intact(self, self.graph)

    def test_remove_vertex_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.remove_vertex("A")
        assert_intact(self, self.graph)

    def test_has_vertex_is_false(self) -> None:
        self.assertFalse(self.graph.has_vertex("A"))

    def test_has_edge_is_false(self) -> None:
        self.assertFalse(self.graph.has_edge("A", "B"))

    def test_weight_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.weight("A", "B")

    def test_neighbors_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.neighbors("A")

    def test_bfs_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.bfs("A")

    def test_dfs_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.dfs("A")

    def test_shortest_path_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.shortest_path("A", "A")

    def test_dijkstra_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.dijkstra("A")

    def test_has_cycle_is_false(self) -> None:
        self.assertFalse(self.graph.has_cycle())

    def test_topological_sort_is_empty(self) -> None:
        self.assertEqual(Graph(directed=True).topological_sort(), [])

    def test_connected_components_is_empty(self) -> None:
        self.assertEqual(self.graph.connected_components(), [])

    def test_is_valid(self) -> None:
        self.assertTrue(self.graph.is_valid())

    def test_iteration_yields_nothing(self) -> None:
        self.assertEqual(list(self.graph), [])

    def test_repr(self) -> None:
        self.assertEqual(
            repr(self.graph), "Graph(directed=False, vertices=0, edges=0)"
        )


class TestAddVertex(unittest.TestCase):

    def test_adds_an_isolated_vertex(self) -> None:
        graph = Graph()
        self.assertTrue(graph.add_vertex("A"))
        assert_intact(self, graph, vertices=["A"])

    def test_returns_false_for_a_vertex_already_there(self) -> None:
        graph = build(vertices=["A"])
        self.assertFalse(graph.add_vertex("A"))
        assert_intact(self, graph, vertices=["A"])

    def test_keeps_the_edges_of_a_vertex_already_there(self) -> None:
        graph = build(TRIANGLE)
        self.assertFalse(graph.add_vertex("A"))
        assert_intact(self, graph, TRIANGLE)

    def test_adds_beside_existing_edges(self) -> None:
        graph = build(PATH)
        self.assertTrue(graph.add_vertex(9))
        assert_intact(self, graph, PATH, vertices=[9])

    def test_falsy_values_are_vertices(self) -> None:
        graph = Graph()
        for vertex in [0, "", None]:
            self.assertTrue(graph.add_vertex(vertex), f"{vertex!r} was not added")
        assert_intact(self, graph, vertices=[0, "", None])

    def test_tuples_are_vertices(self) -> None:
        graph = Graph()
        self.assertTrue(graph.add_vertex((0, 0)))
        assert_intact(self, graph, vertices=[(0, 0)])

    def test_in_a_directed_graph(self) -> None:
        graph = build(PATH, directed=True)
        self.assertTrue(graph.add_vertex(9))
        assert_intact(self, graph, PATH, vertices=[9])

    def test_many_vertices(self) -> None:
        graph = Graph()
        for vertex in range(200):
            graph.add_vertex(vertex)
        assert_intact(self, graph, vertices=range(200))


class TestAddEdge(unittest.TestCase):

    def test_undirected_stores_both_directions(self) -> None:
        graph = build(vertices=["A", "B"])
        graph.add_edge("A", "B")
        assert_intact(self, graph, [("A", "B")])

    def test_directed_stores_one_direction(self) -> None:
        graph = build(vertices=["A", "B"], directed=True)
        graph.add_edge("A", "B")
        assert_intact(self, graph, [("A", "B")])

    def test_creates_both_missing_vertices(self) -> None:
        graph = Graph()
        graph.add_edge("A", "B")
        assert_intact(self, graph, [("A", "B")])

    def test_creates_one_missing_vertex(self) -> None:
        graph = build(vertices=["A"])
        graph.add_edge("A", "B")
        assert_intact(self, graph, [("A", "B")])

    def test_directed_creates_the_target_vertex(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("A", "B")
        assert_intact(self, graph, [("A", "B")])

    def test_weight_defaults_to_one(self) -> None:
        graph = Graph()
        graph.add_edge("A", "B")
        self.assertEqual(graph._adj["A"]["B"], 1, "the default weight is not 1")

    def test_stores_a_given_weight(self) -> None:
        graph = Graph()
        graph.add_edge("A", "B", 7)
        assert_intact(self, graph, [("A", "B", 7)])

    def test_accepts_zero_and_negative_weights(self) -> None:
        graph = Graph()
        graph.add_edge("A", "B", 0)
        graph.add_edge("B", "C", -3)
        assert_intact(self, graph, [("A", "B", 0), ("B", "C", -3)])

    def test_reweights_an_existing_edge(self) -> None:
        graph = build([("A", "B", 1)])
        graph.add_edge("A", "B", 5)
        assert_intact(self, graph, [("A", "B", 5)])

    def test_undirected_reweights_from_the_other_end(self) -> None:
        graph = build([("A", "B", 1)])
        graph.add_edge("B", "A", 5)
        assert_intact(self, graph, [("A", "B", 5)])

    def test_directed_opposite_edges_are_two_edges(self) -> None:
        graph = build([("A", "B")], directed=True)
        graph.add_edge("B", "A")
        assert_intact(self, graph, [("A", "B"), ("B", "A")])

    def test_undirected_self_loop_counts_once(self) -> None:
        graph = Graph()
        graph.add_edge("A", "A")
        assert_intact(self, graph, [("A", "A")])

    def test_directed_self_loop_counts_once(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("A", "A")
        assert_intact(self, graph, [("A", "A")])

    def test_keeps_existing_edges(self) -> None:
        graph = build(TRIANGLE)
        graph.add_edge("C", "D")
        assert_intact(self, graph, TRIANGLE + [("C", "D")])

    def test_counts_every_edge(self) -> None:
        graph = Graph()
        for i, edge in enumerate(TREE, start=1):
            graph.add_edge(*edge)
            self.assertEqual(graph._edge_count, i, f"count wrong after {i} edges")
        assert_intact(self, graph, TREE)

    def test_many_edges(self) -> None:
        graph = Graph()
        for edge in grid(10, weighted=True):
            graph.add_edge(*edge)
        assert_intact(self, graph, grid(10, weighted=True))


class TestRemoveEdge(unittest.TestCase):

    def test_undirected_removes_both_directions(self) -> None:
        graph = build(TRIANGLE)
        graph.remove_edge("A", "B")
        assert_intact(self, graph, [("B", "C"), ("C", "A")])

    def test_undirected_accepts_either_end_first(self) -> None:
        graph = build(TRIANGLE)
        graph.remove_edge("B", "A")
        assert_intact(self, graph, [("B", "C"), ("C", "A")])

    def test_keeps_both_vertices(self) -> None:
        graph = build([("A", "B")])
        graph.remove_edge("A", "B")
        assert_intact(self, graph, vertices=["A", "B"])

    def test_directed_removes_only_that_direction(self) -> None:
        graph = build([("A", "B"), ("B", "A")], directed=True)
        graph.remove_edge("A", "B")
        assert_intact(self, graph, [("B", "A")])

    def test_directed_reverse_is_absent(self) -> None:
        graph = build([("A", "B")], directed=True)
        with self.assertRaises(KeyError):
            graph.remove_edge("B", "A")
        assert_intact(self, graph, [("A", "B")])

    def test_self_loop(self) -> None:
        graph = build([("A", "A"), ("A", "B")])
        graph.remove_edge("A", "A")
        assert_intact(self, graph, [("A", "B")])

    def test_directed_self_loop(self) -> None:
        graph = build([("A", "A"), ("A", "B")], directed=True)
        graph.remove_edge("A", "A")
        assert_intact(self, graph, [("A", "B")])

    def test_leaves_other_weights_alone(self) -> None:
        graph = build(WEIGHTED)
        graph.remove_edge("A", "C")
        assert_intact(self, graph, [e for e in WEIGHTED if e[:2] != ("A", "C")])

    def test_removing_every_edge(self) -> None:
        graph = build(TREE)
        for edge in TREE:
            graph.remove_edge(*edge)
        assert_intact(self, graph, vertices="ABCDEF")

    def test_absent_edge_raises_and_changes_nothing(self) -> None:
        graph = build(PATH)
        with self.assertRaises(KeyError):
            graph.remove_edge(1, 3)
        assert_intact(self, graph, PATH)

    def test_missing_vertex_raises_and_changes_nothing(self) -> None:
        graph = build(PATH)
        with self.assertRaises(KeyError):
            graph.remove_edge(1, 99)
        with self.assertRaises(KeyError):
            graph.remove_edge(99, 1)
        assert_intact(self, graph, PATH)


class TestRemoveVertex(unittest.TestCase):

    def test_isolated_vertex(self) -> None:
        graph = build(PATH, vertices=[9])
        graph.remove_vertex(9)
        assert_intact(self, graph, PATH)

    def test_undirected_drops_it_from_its_neighbors(self) -> None:
        graph = build(TRIANGLE)
        graph.remove_vertex("A")
        assert_intact(self, graph, [("B", "C")])

    def test_a_leaf(self) -> None:
        graph = build(TREE)
        graph.remove_vertex("F")
        assert_intact(self, graph, [e for e in TREE if e != ("C", "F")])

    def test_the_hub_of_a_star(self) -> None:
        graph = build([("hub", i) for i in range(5)])
        graph.remove_vertex("hub")
        assert_intact(self, graph, vertices=range(5))

    def test_other_vertices_keep_their_edges(self) -> None:
        graph = build(DIAMOND + [("D", "E")])
        graph.remove_vertex("A")
        assert_intact(self, graph, [("B", "D"), ("C", "D"), ("D", "E")])

    def test_directed_drops_outgoing_and_incoming(self) -> None:
        graph = build([("A", "B"), ("C", "A"), ("B", "C")], directed=True)
        graph.remove_vertex("A")
        assert_intact(self, graph, [("B", "C")])

    def test_directed_edges_both_ways_are_two_edges(self) -> None:
        graph = build([("A", "B"), ("B", "A"), ("B", "C")], directed=True)
        graph.remove_vertex("A")
        assert_intact(self, graph, [("B", "C")])

    def test_undirected_self_loop(self) -> None:
        graph = build([("A", "A"), ("A", "B"), ("B", "C")])
        graph.remove_vertex("A")
        assert_intact(self, graph, [("B", "C")])

    def test_directed_self_loop(self) -> None:
        graph = build([("A", "A"), ("A", "B"), ("C", "A"), ("B", "C")], directed=True)
        graph.remove_vertex("A")
        assert_intact(self, graph, [("B", "C")])

    def test_the_last_vertex(self) -> None:
        graph = build(vertices=["A"])
        graph.remove_vertex("A")
        assert_intact(self, graph)

    def test_missing_vertex_raises_and_changes_nothing(self) -> None:
        graph = build(TRIANGLE)
        with self.assertRaises(KeyError):
            graph.remove_vertex("Z")
        assert_intact(self, graph, TRIANGLE)


class TestHasVertex(unittest.TestCase):

    def test_finds_a_vertex_with_edges(self) -> None:
        self.assertTrue(build(PATH).has_vertex(2))

    def test_finds_an_isolated_vertex(self) -> None:
        self.assertTrue(build(PATH, vertices=[9]).has_vertex(9))

    def test_rejects_an_absent_vertex(self) -> None:
        self.assertFalse(build(PATH).has_vertex(99))

    def test_finds_a_falsy_vertex(self) -> None:
        self.assertTrue(build(vertices=[0]).has_vertex(0))

    def test_does_not_mutate(self) -> None:
        graph = build(PATH)
        graph.has_vertex(99)
        assert_intact(self, graph, PATH)


class TestHasEdge(unittest.TestCase):

    def test_undirected_answers_both_ways(self) -> None:
        graph = build([("A", "B")])
        self.assertTrue(graph.has_edge("A", "B"))
        self.assertTrue(graph.has_edge("B", "A"))

    def test_directed_answers_one_way(self) -> None:
        graph = build([("A", "B")], directed=True)
        self.assertTrue(graph.has_edge("A", "B"))
        self.assertFalse(graph.has_edge("B", "A"), "found a directed edge backwards")

    def test_rejects_an_absent_edge(self) -> None:
        self.assertFalse(build(PATH).has_edge(1, 3))

    def test_missing_vertices_answer_false(self) -> None:
        graph = build(PATH)
        self.assertFalse(graph.has_edge(1, 99))
        self.assertFalse(graph.has_edge(99, 1))

    def test_self_loop(self) -> None:
        graph = build([("A", "A"), ("A", "B")])
        self.assertTrue(graph.has_edge("A", "A"))
        self.assertFalse(graph.has_edge("B", "B"))

    def test_a_zero_weight_edge_exists(self) -> None:
        self.assertTrue(build([("A", "B", 0)]).has_edge("A", "B"))

    def test_does_not_mutate(self) -> None:
        graph = build(PATH)
        graph.has_edge(1, 99)
        assert_intact(self, graph, PATH)


class TestWeight(unittest.TestCase):

    def test_default_weight(self) -> None:
        self.assertEqual(build([("A", "B")]).weight("A", "B"), 1)

    def test_stored_weight(self) -> None:
        self.assertEqual(build(WEIGHTED).weight("C", "B"), 2)

    def test_undirected_reverse_has_the_same_weight(self) -> None:
        self.assertEqual(build(WEIGHTED).weight("B", "C"), 2)

    def test_zero_and_negative_weights(self) -> None:
        graph = build([("A", "B", 0), ("B", "C", -3)])
        self.assertEqual(graph.weight("A", "B"), 0)
        self.assertEqual(graph.weight("B", "C"), -3)

    def test_self_loop(self) -> None:
        self.assertEqual(build([("A", "A", 6)]).weight("A", "A"), 6)

    def test_directed_reverse_raises(self) -> None:
        with self.assertRaises(KeyError):
            build([("A", "B")], directed=True).weight("B", "A")

    def test_absent_edge_raises(self) -> None:
        with self.assertRaises(KeyError):
            build(PATH).weight(1, 3)

    def test_missing_vertex_raises(self) -> None:
        with self.assertRaises(KeyError):
            build(PATH).weight(1, 99)

    def test_does_not_mutate(self) -> None:
        graph = build(WEIGHTED)
        graph.weight("A", "B")
        assert_intact(self, graph, WEIGHTED)


class TestNeighbors(unittest.TestCase):

    def test_undirected_lists_every_neighbor(self) -> None:
        self.assertEqual(Counter(build(TREE).neighbors("B")), Counter("ADE"))

    def test_directed_lists_only_outgoing(self) -> None:
        graph = build([("A", "B"), ("C", "A")], directed=True)
        self.assertEqual(graph.neighbors("A"), ["B"])

    def test_isolated_vertex(self) -> None:
        self.assertEqual(build(PATH, vertices=[9]).neighbors(9), [])

    def test_self_loop_includes_itself(self) -> None:
        graph = build([("A", "A"), ("A", "B")])
        self.assertEqual(Counter(graph.neighbors("A")), Counter("AB"))

    def test_missing_vertex_raises(self) -> None:
        with self.assertRaises(KeyError):
            build(PATH).neighbors(99)

    def test_returns_a_list_the_caller_owns(self) -> None:
        graph = build(TREE)
        result = graph.neighbors("B")
        self.assertIsInstance(result, list)
        result.append("Z")
        assert_intact(self, graph, TREE)

    def test_does_not_mutate(self) -> None:
        graph = build(TREE)
        graph.neighbors("B")
        assert_intact(self, graph, TREE)


class TestBfs(unittest.TestCase):

    def test_isolated_start(self) -> None:
        self.assertEqual(build(vertices=["A"]).bfs("A"), ["A"])

    def test_path_from_one_end(self) -> None:
        self.assertEqual(build(PATH).bfs(1), [1, 2, 3, 4])

    def test_path_from_the_middle(self) -> None:
        assert_bfs(self, build(PATH), 2)

    def test_visits_level_by_level(self) -> None:
        assert_bfs(self, build(TREE), "A")

    def test_visits_a_shared_vertex_once(self) -> None:
        order = build(DIAMOND).bfs("A")
        self.assertEqual(order.count("D"), 1, "D was visited more than once")
        assert_bfs(self, build(DIAMOND), "A")

    def test_stops_at_the_reachable_vertices(self) -> None:
        graph = build(TRIANGLE + [("X", "Y")])
        self.assertNotIn("X", graph.bfs("A"), "crossed into another component")
        assert_bfs(self, graph, "A")

    def test_directed_follows_edge_direction(self) -> None:
        graph = build(DIAMOND, directed=True)
        self.assertEqual(graph.bfs("B"), ["B", "D"])
        self.assertEqual(graph.bfs("D"), ["D"])

    def test_terminates_on_a_cycle(self) -> None:
        assert_bfs(self, build(TRIANGLE), "A")

    def test_self_loop(self) -> None:
        self.assertEqual(build([("A", "A"), ("A", "B")]).bfs("A"), ["A", "B"])

    def test_grid(self) -> None:
        assert_bfs(self, build(grid(8)), (0, 0))

    def test_missing_start_raises(self) -> None:
        with self.assertRaises(KeyError):
            build(PATH).bfs(99)

    def test_does_not_mutate(self) -> None:
        graph = build(TREE)
        graph.bfs("A")
        assert_intact(self, graph, TREE)


class TestDfs(unittest.TestCase):

    def test_isolated_start(self) -> None:
        self.assertEqual(build(vertices=["A"]).dfs("A"), ["A"])

    def test_path_from_one_end(self) -> None:
        self.assertEqual(build(PATH).dfs(1), [1, 2, 3, 4])

    def test_path_from_the_middle(self) -> None:
        assert_dfs(self, build(PATH), 2)

    def test_finishes_a_branch_before_the_next(self) -> None:
        assert_dfs(self, build(TREE), "A")

    def test_goes_deep_before_wide(self) -> None:
        assert_dfs(self, build(DEEP), "A")

    def test_visits_a_shared_vertex_once(self) -> None:
        order = build(DIAMOND).dfs("A")
        self.assertEqual(order.count("D"), 1, "D was visited more than once")
        assert_dfs(self, build(DIAMOND), "A")

    def test_stops_at_the_reachable_vertices(self) -> None:
        graph = build(TRIANGLE + [("X", "Y")])
        self.assertNotIn("X", graph.dfs("A"), "crossed into another component")
        assert_dfs(self, graph, "A")

    def test_directed_follows_edge_direction(self) -> None:
        graph = build(DIAMOND, directed=True)
        self.assertEqual(graph.dfs("B"), ["B", "D"])
        self.assertEqual(graph.dfs("D"), ["D"])

    def test_terminates_on_a_cycle(self) -> None:
        assert_dfs(self, build(TRIANGLE), "A")

    def test_self_loop(self) -> None:
        self.assertEqual(build([("A", "A"), ("A", "B")]).dfs("A"), ["A", "B"])

    def test_grid(self) -> None:
        assert_dfs(self, build(grid(8)), (0, 0))

    def test_missing_start_raises(self) -> None:
        with self.assertRaises(KeyError):
            build(PATH).dfs(99)

    def test_does_not_mutate(self) -> None:
        graph = build(TREE)
        graph.dfs("A")
        assert_intact(self, graph, TREE)


class TestShortestPath(unittest.TestCase):

    def test_a_vertex_to_itself(self) -> None:
        self.assertEqual(build(PATH).shortest_path(2, 2), [2])

    def test_an_isolated_vertex_to_itself(self) -> None:
        self.assertEqual(build(vertices=["A"]).shortest_path("A", "A"), ["A"])

    def test_neighbors(self) -> None:
        self.assertEqual(build(PATH).shortest_path(1, 2), [1, 2])

    def test_end_to_end(self) -> None:
        self.assertEqual(build(PATH).shortest_path(1, 4), [1, 2, 3, 4])

    def test_undirected_runs_backwards_too(self) -> None:
        self.assertEqual(build(PATH).shortest_path(4, 1), [4, 3, 2, 1])

    def test_takes_the_shortcut(self) -> None:
        ring = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
        self.assertEqual(build(ring).shortest_path("A", "D"), ["A", "E", "D"])

    def test_either_route_of_a_tie(self) -> None:
        assert_shortest_path(self, build(DIAMOND), "A", "D")

    def test_counts_edges_not_weight(self) -> None:
        graph = build([("A", "D", 100), ("A", "B", 1), ("B", "C", 1), ("C", "D", 1)])
        self.assertEqual(graph.shortest_path("A", "D"), ["A", "D"])

    def test_through_a_cycle(self) -> None:
        self.assertEqual(build(TRIANGLE).shortest_path("A", "C"), ["A", "C"])

    def test_across_a_grid(self) -> None:
        assert_shortest_path(self, build(grid(6)), (0, 0), (5, 5))

    def test_unreachable_is_none(self) -> None:
        self.assertIsNone(build(PATH, vertices=[9]).shortest_path(1, 9))

    def test_directed_follows_edge_direction(self) -> None:
        graph = build(PATH, directed=True)
        self.assertEqual(graph.shortest_path(1, 4), [1, 2, 3, 4])
        self.assertIsNone(graph.shortest_path(4, 1), "walked an edge backwards")

    def test_missing_vertex_raises(self) -> None:
        graph = build(PATH)
        with self.assertRaises(KeyError):
            graph.shortest_path(1, 99)
        with self.assertRaises(KeyError):
            graph.shortest_path(99, 1)

    def test_does_not_mutate(self) -> None:
        graph = build(PATH)
        graph.shortest_path(1, 4)
        assert_intact(self, graph, PATH)


class TestDijkstra(unittest.TestCase):

    def test_start_alone(self) -> None:
        self.assertEqual(build(vertices=["A"]).dijkstra("A"), {"A": 0})

    def test_weighted_graph(self) -> None:
        self.assertEqual(
            build(WEIGHTED).dijkstra("A"), {"A": 0, "C": 1, "B": 3, "D": 4}
        )

    def test_many_cheap_edges_beat_one_dear_one(self) -> None:
        graph = build([("A", "D", 10), ("A", "B", 1), ("B", "C", 1), ("C", "D", 1)])
        self.assertEqual(graph.dijkstra("A")["D"], 3)

    def test_settles_the_nearest_vertex_first(self) -> None:
        graph = build([("A", "B", 5), ("A", "C", 1), ("C", "B", 1), ("B", "D", 1)])
        self.assertEqual(graph.dijkstra("A"), {"A": 0, "C": 1, "B": 2, "D": 3})

    def test_leaves_out_unreachable_vertices(self) -> None:
        graph = build(WEIGHTED + [("X", "Y", 1)], vertices=["Z"])
        self.assertEqual(graph.dijkstra("A"), {"A": 0, "C": 1, "B": 3, "D": 4})

    def test_directed_follows_edge_direction(self) -> None:
        graph = build([("A", "B", 1), ("C", "A", 1)], directed=True)
        self.assertEqual(graph.dijkstra("A"), {"A": 0, "B": 1})

    def test_zero_weights(self) -> None:
        graph = build([("A", "B", 0), ("B", "C", 0)])
        self.assertEqual(graph.dijkstra("A"), {"A": 0, "B": 0, "C": 0})

    def test_unit_weights_count_edges(self) -> None:
        graph = build(TREE)
        self.assertEqual(graph.dijkstra("A"), hops(graph._adj, "A"))

    def test_self_loop_changes_nothing(self) -> None:
        graph = build([("A", "A", 3), ("A", "B", 2)])
        self.assertEqual(graph.dijkstra("A"), {"A": 0, "B": 2})

    def test_weighted_grid(self) -> None:
        graph = build(grid(8, weighted=True))
        self.assertEqual(graph.dijkstra((0, 0)), cheapest(graph._adj, (0, 0)))

    def test_negative_weight_raises(self) -> None:
        with self.assertRaises(ValueError):
            build([("A", "B", -1)]).dijkstra("A")

    def test_negative_weight_anywhere_raises(self) -> None:
        with self.assertRaises(ValueError):
            build([("A", "B", 1), ("X", "Y", -1)]).dijkstra("A")

    def test_raising_changes_nothing(self) -> None:
        edges = [("A", "B", 1), ("B", "C", -1)]
        graph = build(edges, directed=True)
        with self.assertRaises(ValueError):
            graph.dijkstra("A")
        assert_intact(self, graph, edges)

    def test_missing_start_raises(self) -> None:
        with self.assertRaises(KeyError):
            build(WEIGHTED).dijkstra("Z")

    def test_does_not_mutate(self) -> None:
        graph = build(WEIGHTED)
        graph.dijkstra("A")
        assert_intact(self, graph, WEIGHTED)


class TestHasCycle(unittest.TestCase):

    def test_single_vertex(self) -> None:
        self.assertFalse(build(vertices=["A"]).has_cycle())

    def test_undirected_edge_is_not_a_cycle(self) -> None:
        self.assertFalse(build([("A", "B")]).has_cycle(), "one edge is not a cycle")

    def test_undirected_tree(self) -> None:
        self.assertFalse(build(TREE).has_cycle())

    def test_undirected_triangle(self) -> None:
        self.assertTrue(build(TRIANGLE).has_cycle())

    def test_undirected_diamond(self) -> None:
        self.assertTrue(build(DIAMOND).has_cycle())

    def test_undirected_cycle_in_a_later_component(self) -> None:
        graph = build([("A", "B"), ("X", "Y"), ("Y", "Z"), ("Z", "X")])
        self.assertTrue(graph.has_cycle(), "missed a cycle in a later component")

    def test_undirected_self_loop(self) -> None:
        self.assertTrue(build([("A", "A"), ("A", "B")]).has_cycle())

    def test_directed_diamond_is_not_a_cycle(self) -> None:
        self.assertFalse(
            build(DIAMOND, directed=True).has_cycle(),
            "two paths to one vertex are not a cycle",
        )

    def test_directed_edge_into_an_earlier_search(self) -> None:
        graph = build([("A", "B"), ("C", "B")], directed=True)
        self.assertFalse(graph.has_cycle(), "two edges into B are not a cycle")

    def test_directed_two_cycle(self) -> None:
        self.assertTrue(build([("A", "B"), ("B", "A")], directed=True).has_cycle())

    def test_directed_triangle(self) -> None:
        self.assertTrue(build(TRIANGLE, directed=True).has_cycle())

    def test_directed_triangle_with_one_edge_turned(self) -> None:
        graph = build([("A", "B"), ("B", "C"), ("A", "C")], directed=True)
        self.assertFalse(graph.has_cycle())

    def test_directed_self_loop(self) -> None:
        self.assertTrue(build([("A", "A")], directed=True).has_cycle())

    def test_directed_cycle_in_a_later_component(self) -> None:
        graph = build([("A", "B"), ("X", "Y"), ("Y", "X")], directed=True)
        self.assertTrue(graph.has_cycle(), "missed a cycle in a later component")

    def test_does_not_mutate(self) -> None:
        graph = build(TRIANGLE, directed=True)
        graph.has_cycle()
        assert_intact(self, graph, TRIANGLE)


class TestTopologicalSort(unittest.TestCase):

    def test_single_vertex(self) -> None:
        graph = build(vertices=["A"], directed=True)
        self.assertEqual(graph.topological_sort(), ["A"])

    def test_a_chain_has_one_answer(self) -> None:
        self.assertEqual(build(PATH, directed=True).topological_sort(), [1, 2, 3, 4])

    def test_edges_added_back_to_front(self) -> None:
        graph = build([(3, 4), (2, 3), (1, 2)], directed=True)
        self.assertEqual(graph.topological_sort(), [1, 2, 3, 4])

    def test_diamond(self) -> None:
        assert_topological(self, build(DIAMOND, directed=True))

    def test_includes_isolated_vertices(self) -> None:
        assert_topological(self, build(DIAMOND, vertices=["Z"], directed=True))

    def test_several_components(self) -> None:
        edges = [("A", "B"), ("X", "Y"), ("Y", "Z")]
        assert_topological(self, build(edges, directed=True))

    def test_many_vertices(self) -> None:
        edges = [
            (i, j) for i in range(30) for j in range(i + 1, 30) if i * j % 7 == 1
        ]
        assert_topological(self, build(edges[::-1], directed=True))

    def test_cycle_raises(self) -> None:
        with self.assertRaises(ValueError):
            build(TRIANGLE, directed=True).topological_sort()

    def test_two_cycle_raises(self) -> None:
        with self.assertRaises(ValueError):
            build([("A", "B"), ("B", "A")], directed=True).topological_sort()

    def test_self_loop_raises(self) -> None:
        with self.assertRaises(ValueError):
            build([("A", "A")], directed=True).topological_sort()

    def test_cycle_in_a_later_component_raises(self) -> None:
        graph = build([("A", "B"), ("X", "Y"), ("Y", "X")], directed=True)
        with self.assertRaises(ValueError):
            graph.topological_sort()

    def test_undirected_raises(self) -> None:
        with self.assertRaises(ValueError):
            build(PATH).topological_sort()

    def test_raising_changes_nothing(self) -> None:
        graph = build(TRIANGLE, directed=True)
        with self.assertRaises(ValueError):
            graph.topological_sort()
        assert_intact(self, graph, TRIANGLE)

    def test_does_not_mutate(self) -> None:
        graph = build(DIAMOND, directed=True)
        graph.topological_sort()
        assert_intact(self, graph, DIAMOND)


class TestConnectedComponents(unittest.TestCase):

    def test_one_component(self) -> None:
        assert_components(self, build(TREE), ["ABCDEF"])

    def test_isolated_vertex_is_its_own_component(self) -> None:
        assert_components(self, build(vertices=["A"]), ["A"])

    def test_several_components(self) -> None:
        graph = build(TRIANGLE + [("X", "Y")], vertices=["Z"])
        assert_components(self, graph, ["ABC", "XY", "Z"])

    def test_self_loop(self) -> None:
        assert_components(self, build([("A", "A")], vertices=["B"]), ["A", "B"])

    def test_grid_is_one_component(self) -> None:
        graph = build(grid(6))
        assert_components(self, graph, [list(graph._adj)])

    def test_directed_raises(self) -> None:
        graph = build(PATH, directed=True)
        with self.assertRaises(ValueError):
            graph.connected_components()
        assert_intact(self, graph, PATH)

    def test_does_not_mutate(self) -> None:
        graph = build(TRIANGLE + [("X", "Y")])
        graph.connected_components()
        assert_intact(self, graph, TRIANGLE + [("X", "Y")])


class TestIsValid(unittest.TestCase):
    """Corruption is wired by hand, since no public method can produce it."""

    def test_empty_graph(self) -> None:
        self.assertTrue(Graph().is_valid())

    def test_undirected_graph(self) -> None:
        self.assertTrue(build(TREE).is_valid())

    def test_directed_graph(self) -> None:
        self.assertTrue(build(DIAMOND, directed=True).is_valid())

    def test_isolated_vertices(self) -> None:
        self.assertTrue(build(PATH, vertices=[9]).is_valid())

    def test_self_loops(self) -> None:
        edges = [("A", "A"), ("A", "B")]
        self.assertTrue(build(edges).is_valid())
        self.assertTrue(build(edges, directed=True).is_valid())

    def test_edge_count_too_high(self) -> None:
        graph = build(TREE)
        graph._edge_count += 1
        self.assertFalse(graph.is_valid())

    def test_edge_count_too_low(self) -> None:
        graph = build(TREE)
        graph._edge_count -= 1
        self.assertFalse(graph.is_valid())

    def test_directed_edge_count_wrong(self) -> None:
        graph = build(DIAMOND, directed=True)
        graph._edge_count += 1
        self.assertFalse(graph.is_valid())

    def test_self_loop_counted_twice(self) -> None:
        graph = build([("A", "A"), ("A", "B")])
        graph._edge_count = 3
        self.assertFalse(graph.is_valid(), "an undirected self-loop is one edge")

    def test_edge_to_a_missing_vertex(self) -> None:
        graph = build(PATH, directed=True)
        graph._adj[1][99] = 1
        graph._edge_count += 1
        self.assertFalse(graph.is_valid(), "1 -> 99 points outside the graph")

    def test_undirected_missing_reverse(self) -> None:
        graph = build(PATH)
        del graph._adj[2][1]
        self.assertFalse(graph.is_valid())

    def test_undirected_one_sided_edges(self) -> None:
        graph = build(PATH)
        del graph._adj[2][1]
        graph._adj[1][3] = 1
        self.assertFalse(graph.is_valid(), "an undirected edge is stored one way")

    def test_undirected_weights_disagree(self) -> None:
        graph = build([("A", "B", 1)])
        graph._adj["B"]["A"] = 2
        self.assertFalse(graph.is_valid(), "the two directions disagree on weight")


class TestDelegatingDunders(unittest.TestCase):
    """__len__ and edge_count() read fields directly, but __contains__
    delegates to has_vertex(), so it stays red until has_vertex() is written."""

    def test_len_counts_vertices(self) -> None:
        self.assertEqual(len(build(PATH, vertices=[9])), 5)
        self.assertEqual(len(Graph()), 0)

    def test_edge_count(self) -> None:
        self.assertEqual(build(TRIANGLE).edge_count(), 3)
        self.assertEqual(build(TRIANGLE, directed=True).edge_count(), 3)

    def test_in_operator(self) -> None:
        graph = build(PATH)
        self.assertIn(1, graph)
        self.assertNotIn(99, graph)


class TestIteration(unittest.TestCase):

    def test_yields_every_vertex_once(self) -> None:
        self.assertEqual(Counter(build(TREE)), Counter("ABCDEF"))

    def test_includes_isolated_vertices(self) -> None:
        self.assertEqual(Counter(build(PATH, vertices=[9])), Counter([1, 2, 3, 4, 9]))

    def test_yields_nothing_when_empty(self) -> None:
        self.assertEqual(list(Graph()), [])

    def test_restarts_on_each_pass(self) -> None:
        graph = build(TREE)
        self.assertEqual(list(graph), list(graph), "iteration is not repeatable")

    def test_does_not_mutate(self) -> None:
        graph = build(TREE)
        list(graph)
        assert_intact(self, graph, TREE)


class TestRepr(unittest.TestCase):

    def test_undirected(self) -> None:
        self.assertEqual(
            repr(build(TRIANGLE)), "Graph(directed=False, vertices=3, edges=3)"
        )

    def test_directed(self) -> None:
        self.assertEqual(
            repr(build(PATH, directed=True)),
            "Graph(directed=True, vertices=4, edges=3)",
        )

    def test_counts_isolated_vertices(self) -> None:
        self.assertEqual(
            repr(build([("A", "B")], vertices=["C"])),
            "Graph(directed=False, vertices=3, edges=1)",
        )

    def test_empty_graph(self) -> None:
        self.assertEqual(repr(Graph()), "Graph(directed=False, vertices=0, edges=0)")


class TestRoundTrip(unittest.TestCase):
    """Drives the mutating methods together, so it needs add_vertex(),
    add_edge(), remove_edge(), remove_vertex() and is_valid(). These are the
    cases where something stale survives: an edge outlives its vertex, or
    the count drifts once a vertex comes back."""

    def test_remove_then_re_add_an_edge(self) -> None:
        graph = build(TRIANGLE)
        graph.remove_edge("A", "B")
        graph.add_edge("A", "B")
        assert_intact(self, graph, TRIANGLE)

    def test_a_removed_vertex_comes_back_bare(self) -> None:
        graph = build(TRIANGLE)
        graph.remove_vertex("A")
        self.assertTrue(graph.add_vertex("A"))
        assert_intact(self, graph, [("B", "C")], vertices=["A"])

    def test_a_new_edge_does_not_revive_old_ones(self) -> None:
        graph = build(TRIANGLE)
        graph.remove_vertex("A")
        graph.add_edge("A", "B")
        assert_intact(self, graph, [("B", "C"), ("A", "B")])

    def test_directed_vertex_comes_back_without_incoming_edges(self) -> None:
        graph = build([("B", "A"), ("A", "C")], directed=True)
        graph.remove_vertex("A")
        graph.add_vertex("A")
        assert_intact(self, graph, vertices="ABC")

    def test_stays_valid_throughout(self) -> None:
        for directed in (False, True):
            graph = Graph(directed=directed)
            for i in range(60):
                u, v = i % 7, i * 3 % 7
                graph.add_edge(u, v, i)
                if i % 4 == 3:
                    graph.remove_edge(u, v)
                if i % 5 == 4:
                    graph.remove_vertex(u)
                self.assertTrue(
                    graph.is_valid(), f"invalid at step {i}, directed={directed}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
