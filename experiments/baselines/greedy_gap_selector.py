"""
AIVA Research Baseline: Greedy Knowledge-Gap Selector
------------------------------------------------------
Selects the concept with the largest estimated knowledge gap:
    c* = argmax_c G_c
Where G_c = 1.0 if unassessed, else max(0.0, 1.0 - mean(scores_c)).
Ignores prerequisite structure, topological centrality, and uncertainty.
"""

import random
from typing import Dict, List, Optional
import networkx as nx


class GreedyGapSelector:
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
        if not self.concept_states:
            return {"selected_concept": "General Architecture", "strategy": "greedy_gap"}

        candidates = []
        for cid, state in self.concept_states.items():
            if state["attempts"] == 0:
                gap = 1.0
            else:
                gap = max(0.0, 1.0 - (sum(state["scores"]) / len(state["scores"])))

            # Penalize immediately previous concept to prevent infinite 1-turn loops on failure
            if state["last_turn"] == turn - 1:
                gap *= 0.5

            candidates.append({"concept_id": cid, "gap": gap, "attempts": state["attempts"]})

        # Shuffle candidates first to randomize tie-breaking deterministically with self.rng
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda x: x["gap"], reverse=True)
        winner = candidates[0]

        return {
            "selected_concept": winner["concept_id"],
            "strategy": "greedy_gap",
            "score_metrics": {"concept_id": winner["concept_id"], "G_c": round(winner["gap"], 4)}
        }
