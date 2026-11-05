"""
AIVA Knowledge Graph Reliability Pipeline
----------------------------------------
Implements:
1. ConceptNormalizer: Alias resolution, canonical deduplication, and lexical normalization.
2. RelationValidator: 5-class constrained relation typing, rule-based directional checks,
   edge confidence calibration, and prerequisite vs general graph separation.
3. DAGCycleDetector: Strict cycle detection and minimum-confidence edge pruning enforced
   ONLY on the prerequisite subgraph (G_prereq), preserving cyclic general relationships.
4. ChunkedDocumentProcessor: Sliding-window chunking preventing text[:10000] truncation.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schemas for Verified Knowledge Graph Extractions
# ---------------------------------------------------------------------------

class VerifiedNode(BaseModel):
    id: str = Field(description="Canonical normalized name of the concept.")
    type: str = Field(default="Concept", description="Category of entity, e.g., Algorithm, Architecture, Tool.")
    aliases: List[str] = Field(default_factory=list, description="Known aliases and acronyms.")
    prereq_depth: float = Field(default=0.5, ge=0.0, le=1.0, description="Estimated pedagogical prerequisite depth.")


class VerifiedRelationship(BaseModel):
    source: str = Field(description="Source canonical concept id.")
    target: str = Field(description="Target canonical concept id.")
    type: str = Field(description="One of: prerequisite, dependency, part_of, implements, evaluates")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Calibrated confidence score [0.0, 1.0].")
    source_is_prerequisite: bool = Field(default=False, description="True if source must precede target.")
    rationale: Optional[str] = Field(default=None, description="Linguistic or rule-based justification.")


class ProcessedGraph(BaseModel):
    nodes: List[VerifiedNode]
    relationships: List[VerifiedRelationship]
    prereq_edges_count: int
    general_edges_count: int
    cycles_detected: int
    cycles_resolved: int


# ---------------------------------------------------------------------------
# 1. Concept Normalizer & Alias Resolver
# ---------------------------------------------------------------------------

class ConceptNormalizer:
    """Normalizes technical concept names, expands acronyms, and deduplicates aliases."""

    COMMON_ACRONYM_MAP = {
        "cnn": "Convolutional Neural Network",
        "convnet": "Convolutional Neural Network",
        "dqn": "Deep Q-Network",
        "mdp": "Markov Decision Process",
        "rag": "Retrieval-Augmented Generation",
        "rrf": "Reciprocal Rank Fusion",
        "bm25": "BM25Okapi",
        "hpa": "Horizontal Pod Autoscaler",
        "jwt": "JSON Web Tokens",
        "k8s": "Kubernetes",
        "nms": "Non-Maximum Suppression",
        "iou": "Intersection over Union",
        "ndcg": "Normalized Discounted Cumulative Gain",
        "td": "Temporal Difference",
        "rl": "Reinforcement Learning",
        "api": "API Gateway",
        "db": "Database"
    }

    @staticmethod
    def clean_text(text: str) -> str:
        """Strips punctuation, trims whitespace, and standardizes casing for comparisons."""
        if not text:
            return ""
        # Strip parenthesized acronyms for the base form, e.g. "Deep Q-Network (DQN)" -> "Deep Q-Network"
        base = re.sub(r"\s*\([^)]*\)", "", text.strip())
        # Lowercase, remove non-alphanumeric chars except space and hyphen
        cleaned = re.sub(r"[^\w\s-]", "", base).strip()
        return " ".join(cleaned.split())

    @classmethod
    def get_canonical_id(cls, raw_name: str, alias_dict: Optional[Dict[str, str]] = None) -> str:
        """Maps raw strings to their canonical name using alias dicts and acronym heuristics."""
        raw_clean = cls.clean_text(raw_name)
        raw_lower = raw_clean.lower()

        # Check for extracted acronyms inside parentheses, e.g. (DQN) -> Deep Q-Network
        acronyms = re.findall(r"\(([^)]+)\)", raw_name)
        for acr in acronyms:
            acr_clean = cls.clean_text(acr).lower()
            if alias_dict and acr_clean in alias_dict:
                return alias_dict[acr_clean]
            if acr_clean in cls.COMMON_ACRONYM_MAP:
                return cls.COMMON_ACRONYM_MAP[acr_clean]

        # 1. Check user/dataset provided alias dictionary
        if alias_dict and raw_lower in alias_dict:
            return alias_dict[raw_lower]

        # 2. Check acronym map
        if raw_lower in cls.COMMON_ACRONYM_MAP:
            return cls.COMMON_ACRONYM_MAP[raw_lower]

        # 3. Handle common noun suffixes (indexes -> index, containers -> container)
        stemmed = raw_lower
        if stemmed.endswith(" indexes"):
            stemmed = stemmed[:-8]
        elif stemmed.endswith(" index"):
            stemmed = stemmed[:-6]
        elif stemmed.endswith("s") and not stemmed.endswith("ss"):
            stemmed = stemmed[:-1]

        if alias_dict and stemmed in alias_dict:
            return alias_dict[stemmed]
        if stemmed in cls.COMMON_ACRONYM_MAP:
            return cls.COMMON_ACRONYM_MAP[stemmed]

        # 4. Capitalize standard title case if raw_name is all lower
        if raw_name.islower():
            return raw_name.title()

        return raw_clean if raw_clean else raw_name.strip()

    @classmethod
    def deduplicate_nodes(
        cls,
        raw_nodes: List[dict],
        custom_aliases: Optional[Dict[str, str]] = None
    ) -> Tuple[List[VerifiedNode], Dict[str, str]]:
        """
        Deduplicates a list of raw nodes, resolves aliases to a single canonical entity,
        and produces a rename mapping for edges.
        """
        # Build composite alias lookup: normalized_alias -> canonical_id
        alias_lookup: Dict[str, str] = {}
        if custom_aliases:
            for k, v in custom_aliases.items():
                alias_lookup[cls.clean_text(k).lower()] = v

        # First pass: map raw nodes into canonical buckets
        canonical_buckets: Dict[str, dict] = {}
        raw_to_canonical: Dict[str, str] = {}

        for n in raw_nodes:
            raw_id = n.get("id") if isinstance(n, dict) else n.id
            raw_type = n.get("type", "Concept") if isinstance(n, dict) else getattr(n, "type", "Concept")
            aliases = n.get("aliases", []) if isinstance(n, dict) else getattr(n, "aliases", [])

            canonical_id = cls.get_canonical_id(raw_id, alias_lookup)
            raw_to_canonical[raw_id] = canonical_id
            raw_to_canonical[raw_id.lower()] = canonical_id

            # Register any explicit aliases declared inside the node
            for a in aliases:
                alias_lookup[cls.clean_text(a).lower()] = canonical_id
                raw_to_canonical[a] = canonical_id
                raw_to_canonical[a.lower()] = canonical_id

            if canonical_id not in canonical_buckets:
                canonical_buckets[canonical_id] = {
                    "id": canonical_id,
                    "type": raw_type,
                    "aliases": set(aliases),
                    "prereq_depth": n.get("prereq_depth", 0.5) if isinstance(n, dict) else getattr(n, "prereq_depth", 0.5)
                }
            else:
                # Merge aliases and retain highest specificity type
                canonical_buckets[canonical_id]["aliases"].update(aliases)
                if raw_id != canonical_id:
                    canonical_buckets[canonical_id]["aliases"].add(raw_id)

        # Build output nodes
        verified_nodes: List[VerifiedNode] = []
        for cid, data in canonical_buckets.items():
            verified_nodes.append(VerifiedNode(
                id=cid,
                type=data["type"],
                aliases=sorted(list(data["aliases"])),
                prereq_depth=data["prereq_depth"]
            ))

        return verified_nodes, raw_to_canonical


# ---------------------------------------------------------------------------
# 2. Relation Validator & Direction Rule Checker
# ---------------------------------------------------------------------------

class RelationValidator:
    """
    Validates relationship types against constrained 5-class taxonomy,
    applies linguistic rule checks for passive-voice direction inversion,
    and assigns calibrated edge confidence scores.
    """

    ALLOWED_TYPES = {"prerequisite", "dependency", "part_of", "implements", "evaluates"}

    # Inverted linguistic predicates where passive voice indicates reverse dependency
    PASSIVE_INVERSION_PATTERNS = [
        "required_by", "needed_by", "controlled_by", "used_by",
        "indexed_by", "implemented_by", "evaluated_by", "subsystem_of"
    ]

    # Explicit forward prerequisite keywords
    FORWARD_PREREQ_KEYWORDS = [
        "prerequisite", "foundational", "precedes", "required_for", "basis_for"
    ]

    @classmethod
    def validate_and_orient_edge(
        cls,
        source: str,
        target: str,
        rel_type: str,
        node_ids: Set[str],
        base_confidence: float = 0.8
    ) -> Optional[VerifiedRelationship]:
        """
        Validates an edge, normalizes its relationship type, applies rule-based directional
        correction for passive inversions, and assigns calibrated confidence.
        """
        if not source or not target or source == target:
            # Self-loops or empty edges are immediately rejected
            return None

        if source not in node_ids or target not in node_ids:
            # Reject edges referencing dangling/unregistered nodes
            return None

        rel_lower = rel_type.lower().strip()
        confidence = max(0.1, min(1.0, float(base_confidence)))
        is_prereq = False
        final_type = "dependency"
        rationale = f"Original predicate: {rel_type}"

        # Rule 1: Check for passive voice inversion
        # e.g., source="Adaptive Difficulty", rel="CONTROLLED_BY", target="Deep Q-Network"
        # Should be oriented as: Deep Q-Network -> Adaptive Difficulty
        if any(pat in rel_lower for pat in cls.PASSIVE_INVERSION_PATTERNS):
            source, target = target, source
            confidence *= 0.95
            rationale += " [Rule: Inverted passive predicate detected; swapped source and target]"

        # Rule 2: Classify into the 5-class taxonomy
        if any(kw in rel_lower for kw in cls.FORWARD_PREREQ_KEYWORDS):
            final_type = "prerequisite"
            is_prereq = True
            confidence = min(1.0, confidence * 1.1)
        elif "part_of" in rel_lower or "component_of" in rel_lower or "subsystem" in rel_lower:
            final_type = "part_of"
        elif "implements" in rel_lower or "acts_as" in rel_lower or "realizes" in rel_lower:
            final_type = "implements"
        elif "evaluates" in rel_lower or "measures" in rel_lower or "tests" in rel_lower:
            final_type = "evaluates"
        elif rel_lower in cls.ALLOWED_TYPES:
            final_type = rel_lower
            if final_type == "prerequisite":
                is_prereq = True
        else:
            # Map generic predicates ('USES', 'CALLS', 'STREAMS_TO', 'POOLS_INTO') to 'dependency'
            final_type = "dependency"

        return VerifiedRelationship(
            source=source,
            target=target,
            type=final_type,
            confidence=round(confidence, 3),
            source_is_prerequisite=is_prereq,
            rationale=rationale
        )


# ---------------------------------------------------------------------------
# 3. DAG Cycle Detector (Enforces DAG ONLY on Prerequisite Subgraph)
# ---------------------------------------------------------------------------

class DAGCycleDetector:
    """
    Enforces strict Directed Acyclic Graph (DAG) constraints ONLY on G_prereq.
    General semantic relationships (G_general) are allowed to contain cycles
    representing valid bidirectional dependencies or runtime feedback loops.
    """

    @classmethod
    def process_and_enforce_dag(
        cls,
        nodes: List[VerifiedNode],
        relationships: List[VerifiedRelationship],
        min_confidence_threshold: float = 0.50
    ) -> Tuple[List[VerifiedRelationship], int, int]:
        """
        1. Filters edges below minimum confidence threshold.
        2. Separates relationships into G_prereq and G_general.
        3. Enforces DAG strictly on G_prereq by identifying cycles and pruning the
           feedback arc with the lowest confidence score.
        4. Recombines G_prereq and G_general into the final verified relationship list.

        Returns: (verified_relationships, cycles_detected, cycles_resolved)
        """
        # Step 1: Filter edges by confidence
        filtered_edges = [
            r for r in relationships
            if r.confidence >= min_confidence_threshold
        ]

        # Step 2: Subgraph separation
        prereq_edges: List[VerifiedRelationship] = []
        general_edges: List[VerifiedRelationship] = []

        for r in filtered_edges:
            if r.type == "prerequisite":
                prereq_edges.append(r)
            else:
                general_edges.append(r)

        # Step 3: Build DiGraph for prerequisite edges only
        prereq_graph = nx.DiGraph()
        for n in nodes:
            prereq_graph.add_node(n.id)

        # Map (u, v) -> VerifiedRelationship to quickly retrieve confidence
        edge_map: Dict[Tuple[str, str], VerifiedRelationship] = {}
        for r in prereq_edges:
            # Handle duplicate parallel edges: keep the one with higher confidence
            key = (r.source, r.target)
            if key not in edge_map or r.confidence > edge_map[key].confidence:
                edge_map[key] = r
                prereq_graph.add_edge(r.source, r.target, weight=r.confidence)

        cycles_detected = 0
        cycles_resolved = 0

        # Step 4: Iteratively find and break cycles in G_prereq
        while True:
            try:
                cycle = nx.find_cycle(prereq_graph, orientation="original")
            except nx.NetworkXNoCycle:
                break

            cycles_detected += 1
            # Cycle is a list of tuples: (u, v, 'forward')
            # Find the edge in the cycle with the lowest confidence score
            min_edge = None
            min_conf = float("inf")

            for edge in cycle:
                u, v = edge[0], edge[1]
                conf = edge_map.get((u, v), VerifiedRelationship(
                    source=u, target=v, type="prerequisite", confidence=0.5
                )).confidence
                if conf < min_conf:
                    min_conf = conf
                    min_edge = (u, v)

            if min_edge:
                # Remove the lowest-confidence feedback edge from G_prereq
                prereq_graph.remove_edge(min_edge[0], min_edge[1])
                # Demote this edge to 'dependency' instead of dropping it entirely,
                # so the semantic connection is not lost in general knowledge
                removed_rel = edge_map.pop(min_edge)
                demoted_rel = VerifiedRelationship(
                    source=removed_rel.source,
                    target=removed_rel.target,
                    type="dependency",
                    confidence=removed_rel.confidence * 0.8,
                    source_is_prerequisite=False,
                    rationale=f"Demoted from prerequisite to dependency to resolve cycle in G_prereq (conf={removed_rel.confidence})"
                )
                general_edges.append(demoted_rel)
                cycles_resolved += 1
            else:
                break

        # Step 5: Compute refined prereq_depth for nodes using the verified DAG
        if prereq_graph.number_of_edges() > 0:
            # Calculate topological depth: distance from root (in-degree=0 nodes)
            in_degrees = dict(prereq_graph.in_degree())
            roots = [n for n, deg in in_degrees.items() if deg == 0]
            depth_map: Dict[str, int] = {n: 0 for n in prereq_graph.nodes()}

            for root in roots:
                # BFS/Shortest path to determine topological hierarchy depth
                lengths = nx.single_source_shortest_path_length(prereq_graph, root)
                for target_node, length in lengths.items():
                    depth_map[target_node] = max(depth_map[target_node], length)

            max_depth = max(depth_map.values()) if depth_map and max(depth_map.values()) > 0 else 1
            for n in nodes:
                if n.id in depth_map:
                    # Normalized depth in [0.1, 0.9]
                    norm_depth = 0.1 + 0.8 * (depth_map[n.id] / max_depth)
                    n.prereq_depth = round(norm_depth, 3)

        # Recombine resolved prerequisite edges and general edges
        final_relationships = list(edge_map.values()) + general_edges
        return final_relationships, cycles_detected, cycles_resolved


# ---------------------------------------------------------------------------
# 4. Chunked Document Processor
# ---------------------------------------------------------------------------

class ChunkedDocumentProcessor:
    """Splits technical documents into sliding semantic windows without truncating."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 500) -> List[str]:
        """Splits long text into overlapping chunks, respecting paragraph boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Find nearest paragraph or sentence break before end
            if end < text_len:
                break_pos = text.rfind("\n\n", start, end)
                if break_pos == -1 or break_pos < start + (chunk_size // 2):
                    break_pos = text.rfind(". ", start, end)
                if break_pos != -1 and break_pos > start + (chunk_size // 2):
                    end = break_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break
            start = end - overlap

        return chunks
