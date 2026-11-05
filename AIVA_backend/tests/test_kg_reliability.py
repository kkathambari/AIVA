"""
Unit and Integration Tests for Knowledge Graph Reliability Pipeline
-------------------------------------------------------------------
Tests:
1. Concept Normalization and Alias Deduplication.
2. Direction Inversion Rules and Calibrated Confidence.
3. DAG Cycle Detection: Prunes feedback arc on G_prereq while preserving cycles in G_general.
4. Confidence Thresholding (UCS safety invariant).
5. Topological Prerequisite Depth Calibration.
"""

import os
import sys
import unittest
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.kg_pipeline import (
    ConceptNormalizer,
    RelationValidator,
    DAGCycleDetector,
    ChunkedDocumentProcessor,
    VerifiedNode,
    VerifiedRelationship
)
from core.knowledge_graph import KGBuilder, GraphExtraction, Node, Relationship


class TestConceptNormalizer(unittest.TestCase):
    def test_acronym_expansion(self):
        self.assertEqual(ConceptNormalizer.get_canonical_id("DQN"), "Deep Q-Network")
        self.assertEqual(ConceptNormalizer.get_canonical_id("cnn"), "Convolutional Neural Network")
        self.assertEqual(ConceptNormalizer.get_canonical_id("RRF"), "Reciprocal Rank Fusion")
        self.assertEqual(ConceptNormalizer.get_canonical_id("HPA"), "Horizontal Pod Autoscaler")

    def test_deduplicate_nodes_merges_synonyms(self):
        raw_nodes = [
            {"id": "CNN", "type": "Model Architecture", "aliases": ["ConvNet"]},
            {"id": "Convolutional Neural Network", "type": "Model Architecture", "aliases": []},
            {"id": "Convolution", "type": "Mathematical Foundation", "aliases": ["2D Convolution"]}
        ]
        verified_nodes, mapping = ConceptNormalizer.deduplicate_nodes(raw_nodes)

        node_ids = [n.id for n in verified_nodes]
        self.assertIn("Convolutional Neural Network", node_ids)
        self.assertNotIn("CNN", node_ids)
        self.assertEqual(len(node_ids), 2)

        self.assertEqual(mapping["CNN"], "Convolutional Neural Network")
        self.assertEqual(mapping["cnn"], "Convolutional Neural Network")
        self.assertEqual(mapping["Convolutional Neural Network"], "Convolutional Neural Network")


class TestRelationValidator(unittest.TestCase):
    def test_passive_inversion_swap(self):
        node_ids = {"Deep Q-Network", "Adaptive Difficulty"}
        rel = RelationValidator.validate_and_orient_edge(
            source="Adaptive Difficulty",
            target="Deep Q-Network",
            rel_type="CONTROLLED_BY",
            node_ids=node_ids,
            base_confidence=0.85
        )
        self.assertIsNotNone(rel)
        self.assertEqual(rel.source, "Deep Q-Network")
        self.assertEqual(rel.target, "Adaptive Difficulty")
        self.assertEqual(rel.type, "dependency")
        self.assertIn("Inverted passive predicate detected", rel.rationale)

    def test_prerequisite_keyword_mapping(self):
        node_ids = {"Convolution", "Feature Map"}
        rel = RelationValidator.validate_and_orient_edge(
            source="Convolution",
            target="Feature Map",
            rel_type="PREREQUISITE_FOR",
            node_ids=node_ids,
            base_confidence=0.8
        )
        self.assertIsNotNone(rel)
        self.assertEqual(rel.source, "Convolution")
        self.assertEqual(rel.target, "Feature Map")
        self.assertEqual(rel.type, "prerequisite")
        self.assertTrue(rel.source_is_prerequisite)
        self.assertGreaterEqual(rel.confidence, 0.8)

    def test_rejects_self_loops_and_dangling_nodes(self):
        node_ids = {"FastAPI", "Docker"}
        self.assertIsNone(RelationValidator.validate_and_orient_edge("FastAPI", "FastAPI", "USES", node_ids))
        self.assertIsNone(RelationValidator.validate_and_orient_edge("FastAPI", "UnknownNode", "USES", node_ids))


class TestDAGCycleDetector(unittest.TestCase):
    def test_prunes_lowest_confidence_edge_in_prereq_cycle(self):
        nodes = [
            VerifiedNode(id="A", type="Concept"),
            VerifiedNode(id="B", type="Concept"),
            VerifiedNode(id="C", type="Concept")
        ]
        # 3-cycle in prerequisite: A -> B (0.9), B -> C (0.8), C -> A (0.35)
        rels = [
            VerifiedRelationship(source="A", target="B", type="prerequisite", confidence=0.9, source_is_prerequisite=True),
            VerifiedRelationship(source="B", target="C", type="prerequisite", confidence=0.8, source_is_prerequisite=True),
            VerifiedRelationship(source="C", target="A", type="prerequisite", confidence=0.35, source_is_prerequisite=True)
        ]

        final_rels, det, res = DAGCycleDetector.process_and_enforce_dag(
            nodes=nodes, relationships=rels, min_confidence_threshold=0.30
        )

        self.assertEqual(det, 1)
        self.assertEqual(res, 1)

        prereq_g = nx.DiGraph()
        for r in final_rels:
            if r.type == "prerequisite":
                prereq_g.add_edge(r.source, r.target)

        self.assertTrue(nx.is_directed_acyclic_graph(prereq_g))
        self.assertNotIn(("C", "A"), prereq_g.edges())

    def test_preserves_cycles_in_general_relationships(self):
        """Technical lead directive: Do not delete cycles in non-prerequisite edges."""
        nodes = [
            VerifiedNode(id="ServiceA", type="Tool"),
            VerifiedNode(id="ServiceB", type="Tool")
        ]
        rels = [
            VerifiedRelationship(source="ServiceA", target="ServiceB", type="dependency", confidence=0.9),
            VerifiedRelationship(source="ServiceB", target="ServiceA", type="dependency", confidence=0.85)
        ]

        final_rels, det, res = DAGCycleDetector.process_and_enforce_dag(
            nodes=nodes, relationships=rels, min_confidence_threshold=0.50
        )

        self.assertEqual(det, 0)
        self.assertEqual(len(final_rels), 2)
        sources_targets = {(r.source, r.target) for r in final_rels}
        self.assertIn(("ServiceA", "ServiceB"), sources_targets)
        self.assertIn(("ServiceB", "ServiceA"), sources_targets)

    def test_filters_low_confidence_edges(self):
        nodes = [VerifiedNode(id="A", type="Concept"), VerifiedNode(id="B", type="Concept")]
        rels = [
            VerifiedRelationship(source="A", target="B", type="dependency", confidence=0.35),
            VerifiedRelationship(source="A", target="B", type="prerequisite", confidence=0.85, source_is_prerequisite=True)
        ]
        final_rels, _, _ = DAGCycleDetector.process_and_enforce_dag(
            nodes=nodes, relationships=rels, min_confidence_threshold=0.50
        )
        self.assertEqual(len(final_rels), 1)
        self.assertEqual(final_rels[0].confidence, 0.85)


class TestChunkedDocumentProcessor(unittest.TestCase):
    def test_sliding_window_does_not_truncate(self):
        text = "Chapter 1 Introduction\n\n" + ("Deep learning and artificial intelligence algorithms. " * 300)
        chunks = ChunkedDocumentProcessor.chunk_text(text, chunk_size=2000, overlap=200)
        self.assertGreater(len(chunks), 1)
        self.assertIn("Deep learning and artificial intelligence algorithms.", chunks[-1])


class TestNetworkXGraphBuildIntegration(unittest.TestCase):
    def test_builder_constructs_dag_metadata(self):
        builder = KGBuilder(min_edge_confidence=0.50)
        extraction = GraphExtraction(
            nodes=[
                Node(id="Convolution", type="Mathematical Foundation", prereq_depth=0.1),
                Node(id="Feature Map", type="System Component", prereq_depth=0.5),
                Node(id="Pooling", type="Algorithm", prereq_depth=0.7)
            ],
            relationships=[
                Relationship(source="Convolution", target="Feature Map", type="prerequisite", confidence=0.9, source_is_prerequisite=True),
                Relationship(source="Feature Map", target="Pooling", type="dependency", confidence=0.8)
            ]
        )
        builder.build_networkx_graph(extraction)

        self.assertEqual(builder.graph.number_of_nodes(), 3)
        self.assertEqual(builder.graph.number_of_edges(), 2)
        self.assertEqual(builder.prereq_graph.number_of_edges(), 1)
        self.assertTrue(nx.is_directed_acyclic_graph(builder.prereq_graph))


if __name__ == "__main__":
    unittest.main()
