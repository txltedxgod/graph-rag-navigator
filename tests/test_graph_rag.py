from fastapi.testclient import TestClient

from graph_rag.api import app
from graph_rag.graph import KnowledgeGraph


def test_graph_traversal() -> None:
    kg = KnowledgeGraph()
    kg.add_relation("Auth", "verifies", "JWT")
    kg.add_relation("JWT", "contains", "Roles")
    paths = kg.multi_hop_search("Auth", max_hops=2)
    assert len(paths) == 2


def test_diamond_dag_and_cycle_safety() -> None:
    kg = KnowledgeGraph()
    kg.add_relation("Gateway", "calls", "Auth")
    kg.add_relation("Gateway", "calls", "Billing")
    kg.add_relation("Auth", "reads", "Postgres")
    kg.add_relation("Billing", "reads", "Postgres")
    # Introduce cycle back edge
    kg.add_relation("Postgres", "notifies", "Gateway")

    paths = kg.multi_hop_search("Gateway", max_hops=3)
    # Diamond graph must preserve both routes to Postgres without revisiting Gateway infinitely
    postgres_paths = [p for p in paths if p.endswith("-> Postgres")]
    assert len(postgres_paths) == 2
    assert any("Auth" in p for p in postgres_paths)
    assert any("Billing" in p for p in postgres_paths)


def test_shortest_path_and_pagerank() -> None:
    kg = KnowledgeGraph()
    kg.add_relation("API Gateway", "routes_to", "Order Service")
    kg.add_relation("Order Service", "publishes_to", "Kafka")
    kg.add_relation("Kafka", "streams_to", "Analytics")

    steps = kg.shortest_path("API Gateway", "Analytics")
    assert len(steps) == 3
    assert steps[0]["target"] == "Order Service"
    assert steps[-1]["target"] == "Analytics"

    ranks = kg.pagerank_centrality()
    assert abs(sum(ranks.values()) - 1.0) < 1e-3
    assert ranks["Analytics"] > ranks["API Gateway"]


def test_subgraph_query_and_api_endpoints() -> None:
    kg = KnowledgeGraph()
    kg.add_node("PostgreSQL", kind="database", description="ACID relational database storage")
    kg.add_node("Connection Pool", kind="infra", description="PgBouncer pool manager")
    kg.add_node("Kubernetes", kind="orchestrator", description="Container scheduling platform")
    kg.add_relation("Connection Pool", "buffers", "PostgreSQL")
    kg.add_relation("Kubernetes", "runs", "Connection Pool")

    res = kg.subgraph_query("How does Kubernetes connect to PostgreSQL?", max_hops=2)
    seed_names = {s["entity"] for s in res["seed_entities"]}
    assert "PostgreSQL" in seed_names
    assert "Kubernetes" in seed_names
    assert len(res["connecting_paths"]) >= 1
    assert "PostgreSQL" in res["synthesized_context"]

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    query_resp = client.post(
        "/api/v1/query",
        json={"question": "How does API Gateway use NexusMQ?", "max_hops": 2},
    )
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert data["triples"]
    assert data["connecting_paths"]

    path_resp = client.get(
        "/api/v1/path",
        params={"source": "Microservice Architecture", "target": "Append-Only Commit Log"},
    )
    assert path_resp.status_code == 200
    assert path_resp.json()["hops"] == 3

    missing_path = client.get("/api/v1/path", params={"source": "Missing", "target": "NexusMQ"})
    assert missing_path.status_code == 404
