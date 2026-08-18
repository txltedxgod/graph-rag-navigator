from graph_rag.graph import KnowledgeGraph

def test_graph_traversal():
    kg = KnowledgeGraph()
    kg.add_relation("Auth", "verifies", "JWT")
    kg.add_relation("JWT", "contains", "Roles")
    paths = kg.multi_hop_search("Auth", max_hops=2)
    assert len(paths) == 2
