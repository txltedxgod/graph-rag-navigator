"""Knowledge Graph and multi-hop GraphRAG retrieval engine."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True)
class GraphEdge:
    """Directed typed edge in the knowledge graph."""

    source: str
    relation: str
    target: str
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeRecord:
    """Entity metadata stored on a knowledge graph node."""

    name: str
    kind: str = "entity"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """In-memory property knowledge graph with multi-hop traversal, PageRank, and GraphRAG queries."""

    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.node_meta: dict[str, NodeRecord] = {}
        self.edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.reverse_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_node(
        self,
        entity: str,
        kind: str = "entity",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register or update an entity node with semantic metadata."""
        if not entity or not entity.strip():
            raise ValueError("Entity name must not be empty")
        self.nodes.add(entity)
        existing = self.node_meta.get(entity)
        if existing is None:
            self.node_meta[entity] = NodeRecord(
                name=entity,
                kind=kind,
                description=description,
                metadata=dict(metadata) if metadata else {},
            )
        else:
            if kind != "entity":
                existing.kind = kind
            if description:
                existing.description = description
            if metadata:
                existing.metadata.update(metadata)

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add a directed typed relation triple (source -[relation]-> target)."""
        if weight <= 0.0:
            raise ValueError("Edge weight must be positive")
        if source not in self.node_meta:
            self.add_node(source)
        if target not in self.node_meta:
            self.add_node(target)

        fwd = {
            "relation": relation,
            "target": target,
            "weight": weight,
            "attributes": dict(attributes) if attributes else {},
        }
        rev = {
            "relation": relation,
            "source": source,
            "weight": weight,
            "attributes": dict(attributes) if attributes else {},
        }
        self.edges[source].append(fwd)
        self.reverse_edges[target].append(rev)

    def multi_hop_search(
        self,
        start_entity: str,
        max_hops: int = 2,
        relation_filter: set[str] | None = None,
        max_paths: int = 200,
    ) -> list[str]:
        """Find simple directed multi-hop paths from start_entity up to max_hops edges."""
        if start_entity not in self.nodes or max_hops < 1 or max_paths < 1:
            return []

        paths: list[str] = []

        def dfs(curr: str, path: str, depth: int, on_path: set[str]) -> None:
            if depth >= max_hops or len(paths) >= max_paths:
                return
            for edge in self.edges.get(curr, []):
                if relation_filter and edge["relation"] not in relation_filter:
                    continue
                next_node = edge["target"]
                if next_node in on_path:
                    continue
                next_path = f"{path} -[{edge['relation']}]-> {next_node}"
                paths.append(next_path)
                if len(paths) >= max_paths:
                    return
                on_path.add(next_node)
                dfs(next_node, next_path, depth + 1, on_path)
                on_path.remove(next_node)

        dfs(start_entity, start_entity, 0, {start_entity})
        return paths

    def shortest_path(self, source: str, target: str, max_hops: int = 6) -> list[dict[str, str]]:
        """Find shortest directed path between source and target as a list of relation steps."""
        if source not in self.nodes or target not in self.nodes or source == target:
            return []

        queue: deque[tuple[str, list[dict[str, str]]]] = deque([(source, [])])
        visited: set[str] = {source}

        while queue:
            curr, trail = queue.popleft()
            if len(trail) >= max_hops:
                continue
            for edge in self.edges.get(curr, []):
                nxt = edge["target"]
                step = {"source": curr, "relation": edge["relation"], "target": nxt}
                next_trail = [*trail, step]
                if nxt == target:
                    return next_trail
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, next_trail))
        return []

    def pagerank_centrality(
        self, damping: float = 0.85, iterations: int = 25
    ) -> dict[str, float]:
        """Compute iterative PageRank centrality across all graph nodes."""
        n = len(self.nodes)
        if n == 0:
            return {}

        ranks = dict.fromkeys(self.nodes, 1.0 / n)
        base = (1.0 - damping) / n

        for _ in range(iterations):
            next_ranks = dict.fromkeys(self.nodes, base)
            dangling_sum = 0.0
            for node in self.nodes:
                out_edges = self.edges.get(node, [])
                if not out_edges:
                    dangling_sum += ranks[node]
                    continue
                total_w = sum(e["weight"] for e in out_edges)
                for edge in out_edges:
                    share = damping * ranks[node] * (edge["weight"] / total_w)
                    next_ranks[edge["target"]] += share

            dangling_share = damping * dangling_sum / n
            for node in self.nodes:
                next_ranks[node] += dangling_share
            ranks = next_ranks

        return {node: round(score, 6) for node, score in ranks.items()}

    def search_nodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Rank entities matching a natural-language query using lexical + centrality signals."""
        if not query or not query.strip() or not self.nodes or top_k <= 0:
            return []

        q_lower = query.lower()
        q_tokens = _tokenize(query)
        centrality = self.pagerank_centrality()
        scored: list[dict[str, Any]] = []

        for entity in sorted(self.nodes):
            record = self.node_meta.get(entity, NodeRecord(name=entity))
            entity_text = f"{entity} {record.kind} {record.description}"
            e_tokens = _tokenize(entity_text)

            exact_phrase_bonus = 1.0 if entity.lower() in q_lower else 0.0
            overlap = len(q_tokens & e_tokens)
            token_recall = overlap / len(e_tokens) if e_tokens else 0.0
            query_recall = overlap / len(q_tokens) if q_tokens else 0.0
            lexical = 0.5 * query_recall + 0.5 * token_recall

            if exact_phrase_bonus == 0.0 and overlap == 0:
                continue

            combined = round(0.55 * exact_phrase_bonus + 0.35 * lexical + 0.10 * centrality[entity], 4)
            scored.append(
                {
                    "entity": entity,
                    "kind": record.kind,
                    "description": record.description,
                    "score": combined,
                    "centrality": centrality[entity],
                }
            )

        scored.sort(key=lambda item: (item["score"], item["centrality"]), reverse=True)
        return scored[:top_k]

    def subgraph_query(
        self,
        question: str,
        max_hops: int = 2,
        top_k_seeds: int = 3,
    ) -> dict[str, Any]:
        """Run end-to-end GraphRAG retrieval: entity linking, k-hop expansion, and context synthesis."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty")

        seeds = self.search_nodes(question, top_k=top_k_seeds)
        seed_names = [s["entity"] for s in seeds]

        collected_triples: list[dict[str, Any]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        node_scores: dict[str, float] = {s["entity"]: s["score"] for s in seeds}

        queue: deque[tuple[str, int, float]] = deque(
            (s["entity"], 0, s["score"]) for s in seeds
        )
        visited_depth: dict[str, int] = dict.fromkeys(seed_names, 0)

        while queue:
            curr, depth, base_score = queue.popleft()
            if depth >= max_hops:
                continue

            # Expand both outgoing and incoming relations to preserve multi-hop neighborhood context
            for edge in self.edges.get(curr, []):
                tgt = edge["target"]
                key = (curr, edge["relation"], tgt)
                hop_score = round(base_score * edge["weight"] * (0.65 ** (depth + 1)), 4)
                if key not in seen_triples:
                    seen_triples.add(key)
                    collected_triples.append(
                        {
                            "source": curr,
                            "relation": edge["relation"],
                            "target": tgt,
                            "hop": depth + 1,
                            "score": hop_score,
                        }
                    )
                node_scores[tgt] = max(node_scores.get(tgt, 0.0), hop_score)
                if depth + 1 < visited_depth.get(tgt, max_hops + 1):
                    visited_depth[tgt] = depth + 1
                    queue.append((tgt, depth + 1, base_score * edge["weight"]))

            for rev in self.reverse_edges.get(curr, []):
                src = rev["source"]
                key = (src, rev["relation"], curr)
                hop_score = round(base_score * rev["weight"] * (0.55 ** (depth + 1)), 4)
                if key not in seen_triples:
                    seen_triples.add(key)
                    collected_triples.append(
                        {
                            "source": src,
                            "relation": rev["relation"],
                            "target": curr,
                            "hop": depth + 1,
                            "score": hop_score,
                        }
                    )
                node_scores[src] = max(node_scores.get(src, 0.0), hop_score)

        connecting_paths: list[list[dict[str, str]]] = []
        for i, src_seed in enumerate(seed_names):
            for tgt_seed in seed_names[i + 1 :]:
                for a, b in ((src_seed, tgt_seed), (tgt_seed, src_seed)):
                    path = self.shortest_path(a, b, max_hops=max_hops + 2)
                    if path:
                        connecting_paths.append(path)

        collected_triples.sort(key=lambda t: t["score"], reverse=True)
        context_lines = [
            f"{t['source']} -[{t['relation']}]-> {t['target']} (hop={t['hop']}, score={t['score']})"
            for t in collected_triples
        ]
        return {
            "question": question,
            "seed_entities": seeds,
            "triples": collected_triples,
            "connecting_paths": connecting_paths,
            "synthesized_context": "\n".join(context_lines),
        }

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the knowledge graph."""
        edge_count = sum(len(items) for items in self.edges.values())
        return {
            "node_count": len(self.nodes),
            "edge_count": edge_count,
            "top_entities": sorted(
                self.pagerank_centrality().items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5],
        }
