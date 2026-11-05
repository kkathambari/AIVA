"""
AIVA Research Baseline: Random Concept Selector
------------------------------------------------
Selects a uniformly random concept from the eligible concept graph at each turn.
Used as the unguided lower-bound baseline.
"""

import random
from typing import Dict, List, Optional
import networkx as nx


class RandomSelector:
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
            self.graph.add_node(cid, type=ctype)
            self.concept_states[cid] = {
                "attempts": 0,
                "scores": [],
                "last_turn": -1,
                "type": ctype
            }

        for r in relationships:
            src = r.get("source") if isinstance(r, dict) else r.source
            tgt = r.get("target") if isinstance(r, dict) else r.target
            if src in self.graph.nodes and tgt in self.graph.nodes:
                self.graph.add_edge(src, tgt)

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
        candidates = list(self.concept_states.keys())
        if not candidates:
            return {"selected_concept": "General Architecture", "strategy": "random"}

        chosen = self.rng.choice(candidates)
        return {
            "selected_concept": chosen,
            "strategy": "random",
            "score_metrics": {"concept_id": chosen, "strategy": "random"}
        }
