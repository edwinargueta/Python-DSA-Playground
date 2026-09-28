"""Graph stub, adjacency map - implement the methods below."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional


class Graph:
    """A weighted graph over an adjacency map, directed or undirected.

    _adj maps each vertex to a dict of the vertices it has an edge to, and
    each of those to that edge's weight. Vertices may be any hashable value.
    Weights default to 1 and may be any number, negative included.

    Structural invariants every mutating method must preserve:
      1. Every vertex is a key in _adj, including a vertex with no edges.
      2. Every neighbor named in _adj is itself a key in _adj - no edge
         points at a vertex the graph does not hold.
      3. Undirected: v is in _adj[u] exactly when u is in _adj[v], with the
         same weight. Directed: the edge u -> v lives only in _adj[u].
      4. At most one edge runs from u to v. add_edge() on an existing edge
         overwrites its weight.
      5. _edge_count equals the number of edges, counting an undirected
         edge once even though it is stored in both directions.

    A self-loop, an edge from a vertex to itself, is allowed. It is stored
    once as _adj[v][v], counts as one edge, and is a cycle.

    Errors: a method given a vertex the graph does not hold raises KeyError,
    except has_vertex() and has_edge(), which answer False. dijkstra()
    raises ValueError if any weight in the graph is negative.
    topological_sort() raises ValueError on an undirected or cyclic graph,
    and connected_components() raises ValueError on a directed one.

    bfs() and dfs() return each vertex reachable from start once, in the
    order the search first reaches it, start first. A path is a list of
    vertices from its first end to its last, both included. Neighbor order
    is unspecified, so any order a correct search could produce is right,
    and so is any one of several tied shortest paths.

    Costs use V for vertices, E for edges, and deg for the number of
    vertices one vertex has an edge to.
    """

    def __init__(self, directed: bool = False) -> None:
        self._adj: Dict[Any, Dict[Any, float]] = {}
        self._directed: bool = directed
        self._edge_count: int = 0

    # --- building --------------------------------------------------------

    def add_vertex(self, vertex: Any) -> bool:
        """Add vertex; False if it was already there. O(1) time, O(1) space."""
        if vertex in self._adj:
            return False
        self._adj[vertex] = {}
        return True

    def add_edge(self, u: Any, v: Any, weight: float = 1) -> None:
        """Add or reweight u -> v; creates missing vertices. O(1) time, O(1) space."""
        self.add_vertex(u)
        self.add_vertex(v)
        if v not in self._adj[u]:
            self._edge_count += 1
        self._adj[u][v] = weight
        if not self._directed:
            self._adj[v][u] = weight

    def remove_edge(self, u: Any, v: Any) -> None:
        """Remove the edge u -> v; KeyError if absent. O(1) time, O(1) space."""
        if u not in self._adj or v not in self._adj[u]:
            raise KeyError(f"no edge {u!r} -> {v!r}")
        del self._adj[u][v]
        if not self._directed and u != v:
            del self._adj[v][u]
        self._edge_count -= 1

    def remove_vertex(self, vertex: Any) -> None:
        """Remove vertex and every edge touching it. O(V) time, O(1) space."""
        if vertex not in self._adj:
            raise KeyError(f"no vertex {vertex!r}")
        # Every outgoing edge, a self-loop included, is one entry here.
        removed = len(self._adj[vertex])
        if self._directed:
            for u in self._adj:
                if u != vertex and vertex in self._adj[u]:
                    del self._adj[u][vertex]
                    removed += 1
        else:
            for u in self._adj[vertex]:
                if u != vertex:
                    del self._adj[u][vertex]
        del self._adj[vertex]
        self._edge_count -= removed

    # --- reading ---------------------------------------------------------

    def has_vertex(self, vertex: Any) -> bool:
        """Return True if vertex is in the graph. O(1) time, O(1) space."""
        raise NotImplementedError

    def has_edge(self, u: Any, v: Any) -> bool:
        """Return True if u has an edge to v. O(1) time, O(1) space."""
        raise NotImplementedError

    def weight(self, u: Any, v: Any) -> float:
        """Return the weight of u -> v; KeyError if absent. O(1) time, O(1) space."""
        raise NotImplementedError

    def neighbors(self, vertex: Any) -> List[Any]:
        """Return the vertices that vertex points to. O(deg) time, O(deg) space."""
        raise NotImplementedError

    def edge_count(self) -> int:
        return self._edge_count

    # --- traversal -------------------------------------------------------

    def bfs(self, start: Any) -> List[Any]:
        """Reachable vertices in breadth-first order. O(V + E) time, O(V) space."""
        raise NotImplementedError

    def dfs(self, start: Any) -> List[Any]:
        """Reachable vertices in depth-first order. O(V + E) time, O(V) space."""
        raise NotImplementedError

    # --- paths -----------------------------------------------------------

    def shortest_path(self, u: Any, v: Any) -> Optional[List[Any]]:
        """A fewest-edges path from u to v, or None. O(V + E) time, O(V) space."""
        raise NotImplementedError

    def dijkstra(self, start: Any) -> Dict[Any, float]:
        """Cheapest distance to each reachable vertex. O(V²) time, O(V) space."""
        raise NotImplementedError

    # --- structure -------------------------------------------------------

    def has_cycle(self) -> bool:
        """Return True if the graph contains a cycle. O(V + E) time, O(V) space."""
        raise NotImplementedError

    def topological_sort(self) -> List[Any]:
        """Order vertices so each edge points forward. O(V + E) time, O(V) space."""
        raise NotImplementedError

    def connected_components(self) -> List[List[Any]]:
        """Group vertices into connected components. O(V + E) time, O(V) space."""
        raise NotImplementedError

    # --- validation ------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if every invariant holds. O(V + E) time, O(1) space."""
        raise NotImplementedError

    # Dunder Helpers
    def __len__(self) -> int:
        return len(self._adj)

    def __contains__(self, vertex: Any) -> bool:
        return self.has_vertex(vertex)

    def __iter__(self) -> Iterator[Any]:
        """Yield each vertex once, in no particular order."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render as Graph(directed=False, vertices=3, edges=2)."""
        raise NotImplementedError


if __name__ == "__main__":
    graph = Graph()
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]:
        graph.add_edge(u, v)
    print(graph)
    print("bfs from A:", graph.bfs("A"))
    print("dfs from A:", graph.dfs("A"))
    print("A to E:", graph.shortest_path("A", "E"))
    print("has cycle:", graph.has_cycle())

    roads = Graph()
    for u, v, miles in [("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 5)]:
        roads.add_edge(u, v, miles)
    print("cheapest from A:", roads.dijkstra("A"))

    chores = Graph(directed=True)
    for u, v in [("wake", "shower"), ("wake", "coffee"), ("shower", "dress")]:
        chores.add_edge(u, v)
    print("chore order:", chores.topological_sort())
