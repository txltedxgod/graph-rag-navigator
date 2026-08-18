from typing import Dict, List, Any, Set

class KnowledgeGraph:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[Dict[str, str]]] = {}

    def add_relation(self, source: str, relation: str, target: str):
        self.nodes.add(source)
        self.nodes.add(target)
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append({"relation": relation, "target": target})

    def multi_hop_search(self, start_entity: str, max_hops: int = 2) -> List[str]:
        if start_entity not in self.nodes:
            return []
        visited = set()
        paths = []

        def dfs(curr, path, depth):
            visited.add(curr)
            if depth >= max_hops:
                return
            for edge in self.edges.get(curr, []):
                next_node = edge["target"]
                p = f"{path} -[{edge['relation']}]-> {next_node}"
                paths.append(p)
                if next_node not in visited:
                    dfs(next_node, p, depth + 1)

        dfs(start_entity, start_entity, 0)
        return paths
