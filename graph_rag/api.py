from fastapi import FastAPI
from pydantic import BaseModel
from graph_rag.graph import KnowledgeGraph

app = FastAPI(title="Graph RAG Navigator", version="0.1.0")
kg = KnowledgeGraph()

# Seed default knowledge graph
kg.add_relation("Microservice Architecture", "uses", "API Gateway")
kg.add_relation("API Gateway", "implements", "Rate Limiting")
kg.add_relation("API Gateway", "routes_to", "NexusMQ")
kg.add_relation("NexusMQ", "provides", "Append-Only Commit Log")

class QueryReq(BaseModel):
    entity: str
    max_hops: int = 2

@app.post("/api/v1/traverse")
def traverse(req: QueryReq):
    paths = kg.multi_hop_search(req.entity, req.max_hops)
    return {"entity": req.entity, "multi_hop_paths": paths}
