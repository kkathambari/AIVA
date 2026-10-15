"""
AIVA Research Core: Unresolved Concept Selection (UCS) Engine
--------------------------------------------------------------
Implements Knowledge-Gap-Aware question sequencing across a directed project concept graph:

    S(c) = w_1 * G_c + w_2 * P_c + w_3 * U_c - w_4 * R_c

Components:
1. G_c (Estimated Knowledge Gap):
   - If unassessed: G_c = 1.0 (maximal candidate gap).
   - If assessed: G_c = max(0.0, 1.0 - mean(scores_c)).

2. P_c (Prerequisite Topological Importance):
   - Normalized structural centrality in the project knowledge graph:
     P_c = (out_degree(c) + 0.5 * in_degree(c)) / max_degree_norm.
   - Concepts that are foundational prerequisites for multiple downstream modules receive high priority.

3. U_c (Epistemic Mastery Uncertainty):
   - Captures uncertainty about whether the student genuinely understands or guessed:
     If unassessed: U_c = 1.0.
     If assessed: U_c = (1.0 / (1.0 + attempts_c)) + 0.5 * (1.0 - 2.0 * abs(mean_score_c - 0.5)).
   - Uncertainty is highest for unprobed concepts or borderline scores near 0.5.

4. R_c (Recency & Frequency Redundancy Penalty):
   - Penalizes circular re-examination of the same concept:
     R_c = (attempts_c / (current_turn + 1.0)) + (0.5 if last_turn == current_turn - 1 else 0.0).

Weights:
- w_1 (Gap Weight) = 0.35
- w_2 (Prerequisite Weight) = 0.25
- w_3 (Uncertainty Weight) = 0.25
- w_4 (Redundancy Weight) = 0.15
(Sum = 1.00)
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import math

class UCSEngine:
    def __init__(
        self,
        w_gap: float = 0.35,
        w_prereq: float = 0.25,
        w_uncertainty: float = 0.25,
        w_redundancy: float = 0.15
    ):
        self.w1 = w_gap
        self.w2 = w_prereq
        self.w3 = w_uncertainty
        self.w4 = w_redundancy
        
        self.graph = nx.DiGraph()
        # concept_id -> {"attempts": int, "scores": List[float], "last_turn": int, "type": str}
        self.concept_states: Dict[str, dict] = {}
        self.current_turn = 0

    def load_concept_graph(self, nodes: List[dict], relationships: List[dict]):
        """Populates the directed NetworkX concept graph and initializes state tracking."""
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
            rtype = r.get("type", "RELATES_TO") if isinstance(r, dict) else getattr(r, "type", "RELATES_TO")
            if src in self.graph.nodes and tgt in self.graph.nodes:
                self.graph.add_edge(src, tgt, type=rtype)

    def update_concept_score(self, concept_id: str, score: float, turn: int):
        """Records student performance on a probed concept."""
        if concept_id not in self.concept_states:
            # Add dynamic concept if not pre-seeded in graph
            self.graph.add_node(concept_id, type="Concept")
            self.concept_states[concept_id] = {
                "attempts": 0,
                "scores": [],
                "last_turn": -1,
                "type": "Concept"
            }
            
        state = self.concept_states[concept_id]
        state["attempts"] += 1
        state["scores"].append(max(0.0, min(1.0, float(score))))
        state["last_turn"] = turn
        self.current_turn = max(self.current_turn, turn)

    def calculate_metrics(self, concept_id: str, turn: int) -> dict:
        """Computes G_c, P_c, U_c, R_c, and composite S(c) for a concept."""
        state = self.concept_states.get(concept_id, {
            "attempts": 0, "scores": [], "last_turn": -1, "type": "Concept"
        })
        attempts = state["attempts"]
        scores = state["scores"]
        last_turn = state["last_turn"]
        
        # 1. G_c: Knowledge Gap
        if attempts == 0:
            G_c = 1.0
        else:
            mean_score = sum(scores) / len(scores)
            G_c = max(0.0, 1.0 - mean_score)
            
        # 2. P_c: Prerequisite Importance (Centrality + Domain Prerequisite Depth)
        base_pdepth = state.get("prereq_depth", 0.5)
        if self.graph.number_of_edges() > 0:
            out_deg = self.graph.out_degree(concept_id) if concept_id in self.graph else 0
            in_deg = self.graph.in_degree(concept_id) if concept_id in self.graph else 0
            deg_sum = out_deg * 1.5 + in_deg * 0.5
            max_deg = max(1.0, max(dict(self.graph.degree()).values(), default=1.0) * 1.5)
            topo_centrality = min(1.0, deg_sum / max_deg)
            P_c = 0.6 * base_pdepth + 0.4 * topo_centrality
        else:
            P_c = base_pdepth

        # 3. U_c: Epistemic Uncertainty
        if attempts == 0:
            U_c = 1.0
        else:
            mean_score = sum(scores) / len(scores)
            sample_uncertainty = 1.0 / (1.0 + attempts)
            # Borderline margin uncertainty (scores near 0.5 are ambiguous)
            margin_uncertainty = 1.0 - 2.0 * abs(mean_score - 0.5)
            U_c = min(1.0, 0.6 * sample_uncertainty + 0.4 * max(0.0, margin_uncertainty))

        # 4. R_c: Redundancy Penalty
        if attempts == 0:
            R_c = 0.0
        else:
            frequency_penalty = attempts / max(1.0, float(turn + 1))
            immediate_recency = 0.5 if (last_turn == turn - 1) else 0.0
            R_c = min(1.0, frequency_penalty + immediate_recency)

        # Composite Selection Score S(c)
        S_c = (self.w1 * G_c) + (self.w2 * P_c) + (self.w3 * U_c) - (self.w4 * R_c)

        return {
            "concept_id": concept_id,
            "S_c": round(S_c, 4),
            "G_c": round(G_c, 4),
            "P_c": round(P_c, 4),
            "U_c": round(U_c, 4),
            "R_c": round(R_c, 4),
            "attempts": attempts,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "type": state["type"]
        }

    def select_next_concept(self, turn: int) -> dict:
        """
        Selects the optimal concept to probe in the upcoming turn:
            c* = argmax_c S(c)
        Also recommends the best academic persona matching the concept's facet.
        """
        if not self.concept_states:
            return {
                "selected_concept": "System Architecture",
                "score_metrics": {},
                "recommended_persona": "Examiner"
            }
            
        ranked_candidates = []
        for cid in self.concept_states.keys():
            metrics = self.calculate_metrics(cid, turn)
            ranked_candidates.append(metrics)
            
        ranked_candidates.sort(key=lambda x: x["S_c"], reverse=True)
        winner = ranked_candidates[0]
        
        # Persona matching based on concept domain
        ctype = winner["type"].lower()
        cid_l = winner["concept_id"].lower()
        
        if any(w in cid_l or w in ctype for w in ["math", "algorithm", "complexity", "theory", "loss", "fusion"]):
            recommended_persona = "Professor"
        elif any(w in cid_l or w in ctype for w in ["docker", "kubernetes", "aws", "deploy", "scale", "database", "fastapi"]):
            recommended_persona = "Industry Expert"
        elif any(w in cid_l or w in ctype for w in ["dqn", "reinforcement", "trade-off", "limitation", "xgboost"]):
            recommended_persona = "Critic"
        else:
            recommended_persona = "Examiner"

        return {
            "selected_concept": winner["concept_id"],
            "selection_score": winner["S_c"],
            "score_metrics": winner,
            "recommended_persona": recommended_persona,
            "ranked_candidates_top3": ranked_candidates[:3]
        }

    def get_coverage_summary(self) -> dict:
        """Returns macro-level assessment efficiency and syllabus coverage metrics."""
        total_concepts = len(self.concept_states)
        if total_concepts == 0:
            return {
                "total_concepts": 0,
                "assessed_count": 0,
                "mastered_count": 0,
                "unresolved_count": 0,
                "coverage_pct": 0.0,
                "mastery_pct": 0.0
            }
            
        assessed_concepts = [c for c, s in self.concept_states.items() if s["attempts"] > 0]
        mastered_concepts = [
            c for c, s in self.concept_states.items() 
            if s["attempts"] > 0 and (sum(s["scores"]) / len(s["scores"])) >= 0.70
        ]
        unresolved_concepts = [
            c for c, s in self.concept_states.items()
            if s["attempts"] == 0 or (sum(s["scores"]) / len(s["scores"])) < 0.50
        ]
        
        return {
            "total_concepts": total_concepts,
            "assessed_count": len(assessed_concepts),
            "mastered_count": len(mastered_concepts),
            "unresolved_count": len(unresolved_concepts),
            "coverage_pct": round((len(assessed_concepts) / total_concepts) * 100, 2),
            "mastery_pct": round((len(mastered_concepts) / total_concepts) * 100, 2)
        }
