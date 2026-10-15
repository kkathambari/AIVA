"""
AIVA Research Test Suite: Deterministic End-to-End HTTP Integration & Session Isolation
---------------------------------------------------------------------------------------
Validates complete FastAPI application workflows offline with 100% mocked external dependencies:
- Zero Gemini API calls, zero network dependencies, zero quota consumption.
- Tests multi-session state isolation (concurrent sessions cannot contaminate each other).
- Tests full HTTP lifecycle: upload -> turn 1 -> student answer -> turn 2 -> session -> analytics.
- Tests concept continuity, provenance logging, and error / edge case resilience.
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch
import concurrent.futures

# Set testing environment variable
os.environ["TESTING"] = "true"
os.environ["GOOGLE_API_KEY"] = "mock_key_for_deterministic_testing"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import main
from main import app, get_or_create_session, AnswerEvaluation, session_store
from core.knowledge_graph import GraphExtraction, Node, Relationship


class TestDeterministicEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicit deterministic initialization of global backend dependencies
        main.rag_engine = MagicMock()
        main.rag_engine.index_document.return_value = 5
        main.rag_engine.query.return_value = ["Context chunk 1", "Context chunk 2"]
        main.analytics_engine = main.AnalyticsEngine()
        main.adaptive_engine = main.AdaptiveEngine()
        
        main.multi_agent_panel = MagicMock()
        def mock_run_panel(**kwargs):
            target = kwargs.get("target_concept") or "System Architecture"
            agent = kwargs.get("recommended_persona") or kwargs.get("agent_type") or "Examiner"
            return {
                "question": f"Explain the design trade-offs and implementation details of '{target}' in your project.",
                "selected_agent": agent,
                "routing_reason": f"Deterministic routing to {agent} for concept {target}."
            }
        main.multi_agent_panel.run_panel.side_effect = mock_run_panel
        
        main.voice_engine = MagicMock()
        main.evaluator_llm = MagicMock()
        main.tutor_llm = MagicMock()
        cls.client = TestClient(app)

    def setUp(self):
        # Clear mock session store for clean test isolation
        session_store.clear()
        main.rag_engine.index_document.return_value = 5
        main.rag_engine.query.return_value = ["Context chunk 1", "Context chunk 2"]

    def _create_mock_kg_extraction(self, prefix: str):
        nodes = [
            Node(id=f"{prefix}_RootArchitecture", type="Architecture"),
            Node(id=f"{prefix}_AsyncWorker", type="Theory"),
            Node(id=f"{prefix}_DatabasePersistence", type="Production")
        ]
        relationships = [
            Relationship(source=f"{prefix}_RootArchitecture", target=f"{prefix}_AsyncWorker", type="POWERS"),
            Relationship(source=f"{prefix}_AsyncWorker", target=f"{prefix}_DatabasePersistence", type="WRITES_TO")
        ]
        return GraphExtraction(nodes=nodes, relationships=relationships)

    @patch("core.knowledge_graph.KGBuilder.extract_graph_from_text")
    @patch("core.parser.DocumentParser.extract_text")
    def test_multi_session_isolation(self, mock_extract_text, mock_extract_kg):
        """
        Critical Test: Proves that Session A and Session B maintain strictly isolated state.
        Uploading to Session A cannot contaminate Session B's concept graph, questions, or scores.
        """
        mock_extract_text.return_value = "Sample technical report documentation."
        
        # 1. Setup Session Alpha
        mock_extract_kg.return_value = self._create_mock_kg_extraction("Alpha")
        resp_alpha = self.client.post(
            "/upload_and_build_kg",
            files={"file": ("alpha_report.txt", b"Alpha Report Content", "text/plain")},
            data={"session_id": "session_alpha"}
        )
        self.assertEqual(resp_alpha.status_code, 200)
        self.assertEqual(resp_alpha.json()["session_id"], "session_alpha")

        # 2. Setup Session Beta
        mock_extract_kg.return_value = self._create_mock_kg_extraction("Beta")
        resp_beta = self.client.post(
            "/upload_and_build_kg",
            files={"file": ("beta_report.txt", b"Beta Report Content", "text/plain")},
            data={"session_id": "session_beta"}
        )
        self.assertEqual(resp_beta.status_code, 200)
        self.assertEqual(resp_beta.json()["session_id"], "session_beta")

        # 3. Assert Session Alpha state contains only Alpha concepts
        state_alpha = self.client.get("/session?session_id=session_alpha").json()
        alpha_nodes = [n["id"] for n in state_alpha["nodes"]]
        self.assertIn("Alpha_RootArchitecture", alpha_nodes)
        self.assertNotIn("Beta_RootArchitecture", alpha_nodes)

        # 4. Assert Session Beta state contains only Beta concepts
        state_beta = self.client.get("/session?session_id=session_beta").json()
        beta_nodes = [n["id"] for n in state_beta["nodes"]]
        self.assertIn("Beta_RootArchitecture", beta_nodes)
        self.assertNotIn("Alpha_RootArchitecture", beta_nodes)

        # 5. Execute Turn 1 in Session Alpha
        turn1_alpha = self.client.post(
            "/multi_agent_question",
            json={
                "session_id": "session_alpha",
                "conversation_history": [],
                "agent_type": "Examiner",
                "question_selection_strategy": "ucs"
            }
        ).json()
        self.assertEqual(turn1_alpha["session_id"], "session_alpha")
        self.assertTrue(turn1_alpha["selected_concept"].startswith("Alpha_"))

        # 6. Verify Session Beta's audit trail and asked questions remain completely empty
        audit_beta = self.client.get("/audit_trail?session_id=session_beta").json()
        self.assertEqual(audit_beta["total_turns"], 0)
        self.assertEqual(len(audit_beta["audit_trail"]), 0)

        # 7. Verify Session Alpha's audit trail has exactly 1 turn
        audit_alpha = self.client.get("/audit_trail?session_id=session_alpha").json()
        self.assertEqual(audit_alpha["total_turns"], 1)
        self.assertEqual(audit_alpha["audit_trail"][0]["session_id"], "session_alpha")

    @patch("core.knowledge_graph.KGBuilder.extract_graph_from_text")
    @patch("core.parser.DocumentParser.extract_text")
    def test_concurrent_session_safety(self, mock_extract_text, mock_extract_kg):
        """
        Verifies thread/async concurrency safety when multiple clients submit viva turns simultaneously.
        Uses isolated client instances per thread and a strict timeout to guarantee deadlock freedom.
        """
        mock_extract_text.return_value = "Concurrent testing report."
        
        # Initialize 3 distinct sessions
        sessions = ["conc_sess_1", "conc_sess_2", "conc_sess_3"]
        for sid in sessions:
            mock_extract_kg.return_value = self._create_mock_kg_extraction(sid)
            resp = self.client.post(
                "/upload_and_build_kg",
                files={"file": (f"{sid}.txt", b"Content", "text/plain")},
                data={"session_id": sid}
            )
            self.assertEqual(resp.status_code, 200)

        def make_turn_request(sid, turn_idx):
            # Use dedicated TestClient instance to avoid cross-thread transport contention
            thread_client = TestClient(app)
            return thread_client.post(
                "/multi_agent_question",
                json={
                    "session_id": sid,
                    "conversation_history": [{"role": "user", "content": f"Answer from turn {turn_idx}"}] if turn_idx > 0 else [],
                    "agent_type": "Examiner",
                    "question_selection_strategy": "ucs"
                }
            )

        # Dispatch 6 concurrent requests across the sessions with a strict 10-second timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(make_turn_request, sessions[i % len(sessions)], i)
                for i in range(6)
            ]
            # Enforce timeout on each future to prevent any test execution hanging
            results = [f.result(timeout=10.0) for f in futures]

        for resp in results:
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("selected_concept", data)
            self.assertIn(data["session_id"], sessions)

        # Verify that each session's state is preserved and correctly updated
        for sid in sessions:
            audit = self.client.get(f"/audit_trail?session_id={sid}").json()
            self.assertEqual(audit["session_id"], sid)
            # Exactly 2 turns dispatched per session (6 total / 3 sessions)
            self.assertEqual(audit["total_turns"], 2)

    @patch("core.knowledge_graph.KGBuilder.extract_graph_from_text")
    @patch("core.parser.DocumentParser.extract_text")
    def test_complete_viva_http_lifecycle(self, mock_extract_text, mock_extract_kg):
        """
        Validates complete end-to-end HTTP viva workflow:
        Upload -> Turn 1 (UCS selection) -> Turn 2 (Answer evaluation & mastery update) -> Session check -> Analytics.
        """
        mock_extract_text.return_value = "AIVA platform technical specification."
        mock_extract_kg.return_value = self._create_mock_kg_extraction("Viva")
        
        sid = "viva_lifecycle_session"
        
        # 1. Upload & Build KG
        upload_resp = self.client.post(
            "/upload_and_build_kg",
            files={"file": ("project_spec.txt", b"Specification Content", "text/plain")},
            data={"session_id": sid}
        )
        self.assertEqual(upload_resp.status_code, 200)
        
        # 2. Turn 1 (First Question)
        turn1_resp = self.client.post(
            "/multi_agent_question",
            json={
                "session_id": sid,
                "conversation_history": [],
                "agent_type": "Examiner",
                "question_selection_strategy": "ucs"
            }
        )
        self.assertEqual(turn1_resp.status_code, 200)
        t1_data = turn1_resp.json()
        q1_text = t1_data["question"]
        concept_q1 = t1_data["selected_concept"]
        self.assertIsNotNone(concept_q1)
        self.assertEqual(t1_data["session_id"], sid)
        
        # Mock evaluator structured response for Turn 2
        mock_eval_result = AnswerEvaluation(
            correctness=0.85,
            completeness=0.80,
            evidence_alignment=0.75,
            reasoning_depth=0.80,
            score=0.81,
            feedback="Strong demonstration of architectural fundamentals.",
            concept=concept_q1
        )
        
        # 3. Turn 2: Student submits defense answer to Question 1
        with patch.object(main.evaluator_llm, "invoke", return_value=mock_eval_result):
            turn2_resp = self.client.post(
                "/multi_agent_question",
                json={
                    "session_id": sid,
                    "conversation_history": [
                        {"role": "agent", "agent": "Examiner", "content": q1_text},
                        {"role": "user", "content": "I implemented asynchronous worker loops to handle concurrent student sessions."}
                    ],
                    "agent_type": "Examiner",
                    "question_selection_strategy": "ucs"
                }
            )
        self.assertEqual(turn2_resp.status_code, 200)
        t2_data = turn2_resp.json()
        
        # Verify evaluation occurred and preserved provenance
        self.assertIsNotNone(t2_data["evaluation"])
        self.assertEqual(t2_data["evaluation"]["concept"], concept_q1)
        self.assertEqual(t2_data["evaluation_provenance"]["execution_mode"], "live_model")
        self.assertEqual(t2_data["evaluation_provenance"]["concept_validation_status"], "matched_exact")
        
        # Verify next concept was selected without circular trapping
        concept_q2 = t2_data["selected_concept"]
        self.assertIsNotNone(concept_q2)
        
        # 4. Check Session State
        session_state = self.client.get(f"/session?session_id={sid}").json()
        self.assertEqual(len(session_state["turn_audit_trail"]), 2)
        self.assertEqual(session_state["ucs_summary"]["assessed_count"], 1)
        
        # 5. Check Analytics endpoint
        analytics_resp = self.client.post("/analytics", json={"session_id": sid})
        self.assertEqual(analytics_resp.status_code, 200)
        analytics_data = analytics_resp.json()
        self.assertIn("readiness_prediction", analytics_data)
        self.assertIn("coverage", analytics_data)

    def test_negative_and_edge_cases(self):
        """
        Tests malformed documents, missing sessions, and evaluation fallbacks.
        """
        # 1. Empty document upload returns 400
        with patch("core.parser.DocumentParser.extract_text", return_value="   "):
            resp = self.client.post(
                "/upload_and_build_kg",
                files={"file": ("empty.txt", b"", "text/plain")}
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Could not extract text", resp.json()["detail"])

        # 2. Reset on a non-existent session safely initializes and resets
        resp_reset = self.client.post("/reset_examination?session_id=brand_new_session")
        self.assertEqual(resp_reset.status_code, 200)
        self.assertEqual(resp_reset.json()["session_id"], "brand_new_session")

        # 3. LLM failure fallback maintains concept continuity and records fallback provenance
        sess = get_or_create_session("fallback_test_session")
        sess.ucs_engine.load_concept_graph([{"id": "Concept_X", "type": "Theory"}], [])
        sess.turn_audit_trail.append({"selected_concept": "Concept_X"})
        
        with patch.object(main.evaluator_llm, "invoke", side_effect=RuntimeError("Mock API Quota Exhausted")):
            with patch("core.rag_engine.RAGEngine.query", return_value=[]):
                resp_fallback = self.client.post(
                    "/multi_agent_question",
                    json={
                        "session_id": "fallback_test_session",
                        "conversation_history": [
                            {"role": "agent", "content": "Question about Concept_X"},
                            {"role": "user", "content": "My brief answer."}
                        ],
                        "agent_type": "Examiner",
                        "question_selection_strategy": "ucs"
                    }
                )
        self.assertEqual(resp_fallback.status_code, 200)
        fb_data = resp_fallback.json()
        # Proved that evaluation fell back gracefully without crashing
        self.assertEqual(fb_data["evaluation_provenance"]["execution_mode"], "deterministic_fallback")
        self.assertEqual(fb_data["evaluation"]["concept"], "Concept_X")


if __name__ == "__main__":
    unittest.main()
