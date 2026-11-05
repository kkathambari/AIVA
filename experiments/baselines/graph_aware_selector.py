"""
AIVA Research Baseline: Graph-Aware Prerequisite Selector
----------------------------------------------------------
Selects concepts strictly following prerequisite topological ordering:
Foundational in-degree=0 root concepts are probed first, followed by downstream modules.
Ignores student knowledge gaps G_c and epistemic uncertainty U_c.
"""

import random
from typing import Dict, List, Optional
import networkx as nx


class GraphAwareSelector:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.graph = nx.DiGraph()
        self.concept_states: Dict[str, dict] = {}
        self.current_turn = 0

    def load_concept_graph(self, nodes: List[dict], relationships: List[dict]):
        self.graph.clear()
        self.concept_states.clear()
        for n in nodes:
            cid = n.get("id") if isinstance(n, dict) else n.id
            ctype = n.get("type", "Concept") if isinstance(n, dict) else getattr(n, "type", "Concept")
            pdepth = n.get("prereq_depth", 0.5) if isinstance(n, dict) else getattr(n, "prereq_depth", 0.5)
            self.graph.add_node(cid, type=ctype, prereq_depth=pdepth)
            self.concept_states[cid] = {
                "attempts": 0,
                "scores": [],
                "last_turn": -1,
                "type": ctype,
                "prereq_depth": pdepth
            }

        for r in relationships:
            src = r.get("source") if isinstance(r, dict) else r.source
            tgt = r.get("target") if isinstance(r, dict) else r.target
            rtype = r.get("type", "dependency") if isinstance(r, dict) else getattr(r, "type", "dependency")
            if src in self.graph.nodes and tgt in self.graph.nodes:
                self.graph.add_edge(src, tgt, type=rtype)

    def update_concept_score(self, concept_id: str, score: float, turn: int):
        if concept_id not in self.concept_states:
            self.graph.add_node(concept_id, type="Concept")
            self.concept_states[concept_id] = {"attempts": 0, "scores": [], "last_turn": -1, "type": "Concept"}
        state = self.concept_states[concept_id]
        state["attempts"] += 1
        state["scores"].append(float(score))
        state["last_turn"] = turn
        self.current_turn = max(self.current_turn, turn)

    def select_next_concept(self, turn: int) -> dict:
        if not self.concept_states:
            return {"selected_concept": "General Architecture", "strategy": "graph_aware"}

        # Filter candidate nodes: prioritize unprobed nodes whose prerequisites have already been probed
        unprobed = [cid for cid, s in self.concept_states.items() if s["attempts"] == 0]

        eligible = []
        for cid in unprobed:
            # Check predecessors (prerequisites)
            preds = list(self.graph.predecessors(cid)) if cid in self.graph else []
            # Node is eligible if it has no predecessors or all predecessors have been probed
            if not preds or all(self.concept_states[p]["attempts"] > 0 for p in preds if p in self.concept_states):
                eligible.append(cid)

        if not eligible:
            # If all prerequisites are satisfied or all unprobed are blocked, fall back to any unprobed or least attempted
            eligible = unprobed if unprobed else list(self.concept_states.keys())

        # Rank eligible by topological out-degree (foundational importance)
        ranked = []
        for cid in eligible:
            out_deg = self.graph.out_degree(cid) if cid in self.graph else 0
            attempts = self.concept_states[cid]["attempts"]
            pdepth = self.concept_states[cid].get("prereq_depth", 0.5)
            # Prioritize low attempts, high out_degree, low prerequisite depth (foundations first)
            priority = (out_deg * 2.0) - (pdepth * 1.5) - (attempts * 3.0)
            ranked.append({"concept_id": cid, "priority": priority})

        self.rng.shuffle(ranked)
        ranked.sort(key=lambda x: x["priority"], reverse=True)
        winner = ranked[0]

        return {
            "selected_concept": winner["concept_id"],
            "strategy": "graph_aware",
            "score_metrics": {"concept_id": winner["concept_id"], "priority": round(winner["priority"], 4)}
        }
