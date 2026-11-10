"""FastAPI HTTP interface for Knowledge Graph construction and GraphRAG reasoning."""

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from graph_rag.graph import KnowledgeGraph

app = FastAPI(title="Graph RAG Navigator", version="1.0.0")
kg = KnowledgeGraph()

kg.add_node("Microservice Architecture", kind="pattern", description="Distributed backend architecture")
kg.add_node("API Gateway", kind="component", description="Edge routing, auth, and traffic management")
kg.add_node("Rate Limiting", kind="capability", description="Token bucket quota protection")
kg.add_node("NexusMQ", kind="broker", description="High-throughput mmap commit-log broker")
kg.add_node("Append-Only Commit Log", kind="storage", description="Immutable sequential event log")

kg.add_relation("Microservice Architecture", "uses", "API Gateway")
kg.add_relation("API Gateway", "implements", "Rate Limiting")
kg.add_relation("API Gateway", "routes_to", "NexusMQ")
kg.add_relation("NexusMQ", "provides", "Append-Only Commit Log")


class TripleCreateReq(BaseModel):
    """Payload to add a directed typed edge and optional metadata."""

    source: str = Field(..., min_length=1, max_length=200)
    relation: str = Field(..., min_length=1, max_length=100)
    target: str = Field(..., min_length=1, max_length=200)
    weight: float = Field(default=1.0, gt=0.0, le=100.0)


class QueryReq(BaseModel):
    """Multi-hop traversal request anchored at a starting entity."""

    entity: str = Field(..., min_length=1, max_length=200)
    max_hops: int = Field(default=2, ge=1, le=6)


class GraphRAGQueryReq(BaseModel):
    """Natural-language GraphRAG question payload."""

    question: str = Field(..., min_length=2, max_length=1000)
    max_hops: int = Field(default=2, ge=1, le=5)
    top_k_seeds: int = Field(default=3, ge=1, le=10)


@app.get("/health", tags=["Monitoring"])
def health_check() -> dict[str, Any]:
    """Liveness check returning graph counters."""
    return {"status": "healthy", **kg.stats()}


@app.post("/api/v1/triples", status_code=201, tags=["Graph"])
def add_triple(req: TripleCreateReq) -> dict[str, Any]:
    """Add a relation triple to the knowledge graph."""
    kg.add_relation(req.source, req.relation, req.target, weight=req.weight)
    return {"status": "added", **kg.stats()}


@app.post("/api/v1/traverse", tags=["GraphRAG"])
def traverse(req: QueryReq) -> dict[str, Any]:
    """Execute multi-hop graph traversal from a given entity."""
    paths = kg.multi_hop_search(req.entity, req.max_hops)
    return {"entity": req.entity, "multi_hop_paths": paths}


@app.post("/api/v1/query", tags=["GraphRAG"])
def graph_rag_query(req: GraphRAGQueryReq) -> dict[str, Any]:
    """Execute natural-language entity linking + k-hop subgraph synthesis."""
    return kg.subgraph_query(
        question=req.question,
        max_hops=req.max_hops,
        top_k_seeds=req.top_k_seeds,
    )


@app.get("/api/v1/path", tags=["GraphRAG"])
def explain_shortest_path(
    source: str = Query(..., min_length=1),
    target: str = Query(..., min_length=1),
    max_hops: int = Query(default=6, ge=1, le=10),
) -> dict[str, Any]:
    """Explain the shortest relation path connecting source and target entities."""
    if source not in kg.nodes or target not in kg.nodes:
        raise HTTPException(status_code=404, detail="Source or target entity not found")
    steps = kg.shortest_path(source, target, max_hops=max_hops)
    return {"source": source, "target": target, "hops": len(steps), "steps": steps}


@app.get("/api/v1/stats", tags=["Graph"])
def get_stats() -> dict[str, Any]:
    """Return node/edge totals and top PageRank entities."""
    return kg.stats()
