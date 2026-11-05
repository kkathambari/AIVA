import networkx as nx
from pyvis.network import Network
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
import json
import time

from core.kg_pipeline import (
    ConceptNormalizer,
    RelationValidator,
    DAGCycleDetector,
    ChunkedDocumentProcessor,
    VerifiedNode,
    VerifiedRelationship
)


class Node(BaseModel):
    id: str = Field(description="Canonical name or identifier of the entity, e.g., 'Convolutional Neural Network', 'PostgreSQL'")
    type: str = Field(default="Concept", description="Category: 'Mathematical Foundation', 'Algorithm', 'Model Architecture', 'System Component', 'Framework / Tool', 'Protocol / Standard', 'Evaluation Metric / Methodology'")
    aliases: List[str] = Field(default_factory=list, description="Common acronyms, synonyms, and lexical variants, e.g., ['CNN', 'ConvNet']")
    prereq_depth: float = Field(default=0.5, ge=0.0, le=1.0, description="Pedagogical depth in curriculum hierarchy (0.0=foundational, 1.0=specialized).")


class Relationship(BaseModel):
    source: str = Field(description="Source entity id")
    target: str = Field(description="Target entity id")
    type: str = Field(
        default="dependency",
        description="Constrained relationship taxonomy: 'prerequisite', 'dependency', 'part_of', 'implements', 'evaluates'"
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score of relationship extraction [0.0, 1.0]")
    source_is_prerequisite: bool = Field(default=False, description="Set strictly to True if source MUST be understood before target.")


class GraphExtraction(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]


class KGBuilder:
    def __init__(self, api_key: str = None, min_edge_confidence: float = 0.50):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.graph = nx.DiGraph()
        self.prereq_graph = nx.DiGraph()
        self.generation_time = 0.0
        self.min_edge_confidence = min_edge_confidence

    def _extract_chunk(self, chunk_text: str, structured_llm) -> Optional[GraphExtraction]:
        """Extracts candidate nodes and typed relationships from a single document chunk."""
        prompt = f"""
        Analyze the following text from a student's technical project report.
        Extract the key technical concepts, tools, methodologies, algorithms, and their relationships.
        
        Strict Guidelines:
        1. Entity Canonicalization:
           - Use full canonical names for entities (e.g., 'Convolutional Neural Network' instead of 'CNN', 'Deep Q-Network' instead of 'DQN').
           - Include all acronyms and aliases in the 'aliases' field.
        2. Relationship Taxonomy & Direction:
           - 'prerequisite': Understanding source is strictly required before understanding target (e.g. Convolution -> Feature Map).
           - 'dependency': Operational runtime or invocation dependency (e.g. FastAPI -> PostgreSQL).
           - 'part_of': Architectural subcomponent or module containment (e.g. Pooling -> Convolutional Neural Network).
           - 'implements': Concrete realization of an abstract pattern/algorithm (e.g. ChromaDB -> Vector Database).
           - 'evaluates': Metric or benchmark assessing performance (e.g. IoU -> Non-Maximum Suppression).
        3. Do NOT extract inverted passive relationships (e.g., if A is controlled by B, set source=B and target=A).
        4. Focus on the 15-25 most critical technical components.
        
        Text:
        {chunk_text}
        """
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            print(f"Error extracting chunk: {e}")
            return None

    def extract_graph_from_text(self, text: str, custom_aliases: Optional[Dict[str, str]] = None) -> Optional[GraphExtraction]:
        """
        Full multi-stage pipeline:
        1. Sliding window chunking (no text[:10000] arbitrary truncation).
        2. LLM extraction with constrained 5-class relationship taxonomy.
        3. Concept normalization and alias deduplication.
        4. Rule-based direction inversion correction and edge confidence calibration.
        5. Prerequisite DAG enforcement (cycles pruned ONLY on G_prereq, preserving general cyclic relationships).
        6. Calculation of calibrated prerequisite depth.
        """
        start_time = time.time()
        structured_llm = self.llm.with_structured_output(GraphExtraction)

        # Step 1: Chunking
        chunks = ChunkedDocumentProcessor.chunk_text(text, chunk_size=6000, overlap=600)
        raw_nodes: List[dict] = []
        raw_relationships: List[dict] = []

        for chunk in chunks:
            res = self._extract_chunk(chunk, structured_llm)
            if res:
                for n in res.nodes:
                    raw_nodes.append(n.model_dump())
                for r in res.relationships:
                    raw_relationships.append(r.model_dump())

        if not raw_nodes:
            self.generation_time = round(time.time() - start_time, 2)
            return None

        # Step 2: Concept Normalization & Alias Deduplication
        verified_nodes, raw_to_canonical = ConceptNormalizer.deduplicate_nodes(raw_nodes, custom_aliases)
        node_id_set = {n.id for n in verified_nodes}

        # Step 3: Edge Validation & Direction Inversion Rules
        candidate_edges: List[VerifiedRelationship] = []
        for r in raw_relationships:
            raw_src = r.get("source", "")
            raw_tgt = r.get("target", "")
            src_canon = raw_to_canonical.get(raw_src, raw_to_canonical.get(raw_src.lower(), raw_src))
            tgt_canon = raw_to_canonical.get(raw_tgt, raw_to_canonical.get(raw_tgt.lower(), raw_tgt))

            v_rel = RelationValidator.validate_and_orient_edge(
                source=src_canon,
                target=tgt_canon,
                rel_type=r.get("type", "dependency"),
                node_ids=node_id_set,
                base_confidence=r.get("confidence", 0.8)
            )
            if v_rel:
                candidate_edges.append(v_rel)

        # Step 4: Enforce DAG strictly on G_prereq (preserving cyclic general relationships)
        final_rels, cycles_det, cycles_res = DAGCycleDetector.process_and_enforce_dag(
            nodes=verified_nodes,
            relationships=candidate_edges,
            min_confidence_threshold=self.min_edge_confidence
        )

        self.generation_time = round(time.time() - start_time, 2)

        # Convert back to GraphExtraction format for system compatibility
        output_nodes = [
            Node(
                id=n.id,
                type=n.type,
                aliases=n.aliases,
                prereq_depth=n.prereq_depth
            ) for n in verified_nodes
        ]
        output_rels = [
            Relationship(
                source=r.source,
                target=r.target,
                type=r.type,
                confidence=r.confidence,
                source_is_prerequisite=r.source_is_prerequisite
            ) for r in final_rels
        ]

        result = GraphExtraction(nodes=output_nodes, relationships=output_rels)

        # Save validated kg_data.json
        kg_data = {
            "nodes": [node.model_dump() for node in result.nodes],
            "relationships": [rel.model_dump() for rel in result.relationships],
            "validation_summary": {
                "total_nodes": len(result.nodes),
                "total_edges": len(result.relationships),
                "prereq_edges": sum(1 for r in result.relationships if r.type == "prerequisite"),
                "cycles_detected": cycles_det,
                "cycles_resolved": cycles_res,
                "execution_time_sec": self.generation_time
            }
        }
        with open("kg_data.json", "w", encoding="utf-8") as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)

        return result

    def build_networkx_graph(self, extraction: GraphExtraction):
        """Constructs NetworkX graphs separating prerequisite hierarchy from general topology."""
        self.graph.clear()
        self.prereq_graph.clear()

        concept_types = set()

        for node in extraction.nodes:
            self.graph.add_node(
                node.id,
                label=node.id,
                title=f"{node.type} (Depth: {node.prereq_depth})",
                group=node.type,
                prereq_depth=node.prereq_depth
            )
            self.prereq_graph.add_node(node.id, prereq_depth=node.prereq_depth)
            concept_types.add(node.type)

        prereq_count = 0
        general_count = 0

        for rel in extraction.relationships:
            if rel.source in self.graph.nodes and rel.target in self.graph.nodes:
                self.graph.add_edge(
                    rel.source,
                    rel.target,
                    label=rel.type,
                    title=f"{rel.type} (conf: {rel.confidence:.2f})",
                    confidence=rel.confidence,
                    type=rel.type
                )
                if rel.type == "prerequisite":
                    self.prereq_graph.add_edge(
                        rel.source,
                        rel.target,
                        confidence=rel.confidence
                    )
                    prereq_count += 1
                else:
                    general_count += 1

        is_prereq_dag = nx.is_directed_acyclic_graph(self.prereq_graph)
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        graph_density = round(nx.density(self.graph), 4) if num_nodes > 1 else 0.0

        metadata = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "prereq_edges": prereq_count,
            "general_edges": general_count,
            "prereq_is_dag": is_prereq_dag,
            "num_concepts": len(concept_types),
            "graph_density": graph_density,
            "generation_time_sec": self.generation_time
        }

        with open("graph_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def generate_html_visualization(self, output_file="kg_visualization.html"):
        net = Network(height="600px", width="100%", bgcolor="#0b0f19", font_color="white", directed=True)
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        net.from_nx(self.graph)
        net.save_graph(output_file)
        return output_file
