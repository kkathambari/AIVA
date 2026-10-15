from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from core.parser import DocumentParser
from core.knowledge_graph import KGBuilder
from core.adaptive_engine import AdaptiveEngine
from core.multi_agent import MultiAgentPanel
from core.analytics_engine import AnalyticsEngine
from core.rag_engine import RAGEngine
from core.voice_engine import VoiceEngine
from core.ucs_engine import UCSEngine
import shutil
import os
import json
import asyncio
import threading
import datetime
import uuid
from typing import List, Optional, Dict
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="AIVA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_engine = None
analytics_engine = None
adaptive_engine = None
multi_agent_panel = None
voice_engine = None
evaluator_llm = None
tutor_llm = None

class AnswerEvaluation(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0, default=0.5, description="Technical accuracy of the student's answer (0.0 to 1.0).")
    completeness: float = Field(ge=0.0, le=1.0, default=0.5, description="Degree to which all essential components of the question are addressed (0.0 to 1.0).")
    evidence_alignment: float = Field(ge=0.0, le=1.0, default=0.5, description="Consistency with facts and architectural details in the retrieved project report context (0.0 to 1.0).")
    reasoning_depth: float = Field(ge=0.0, le=1.0, default=0.5, description="Student's ability to articulate 'why' specific choices/methods were used and explain trade-offs (0.0 to 1.0).")
    score: float = Field(ge=0.0, le=1.0, default=0.5, description="Composite score (0.0 to 1.0) synthesized from the 4 rubric dimensions.")
    feedback: str = Field(description="Actionable diagnostic feedback explaining strengths, omissions, or misconceptions.")
    concept: str = Field(description="Primary technical concept tested (1-3 words).")

@app.on_event("startup")
def startup_event():
    global rag_engine, analytics_engine, adaptive_engine, multi_agent_panel, voice_engine, evaluator_llm, tutor_llm
    rag_engine = RAGEngine()
    analytics_engine = AnalyticsEngine()
    adaptive_engine = AdaptiveEngine()
    multi_agent_panel = MultiAgentPanel()
    voice_engine = VoiceEngine()
    
    # Initialize the evaluator LLM using Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_retries=1, timeout=10).with_structured_output(AnswerEvaluation)
    tutor_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_retries=1, timeout=10)
    
    # Load existing KG data from file if present to restore session
    if os.path.exists("kg_data.json"):
        try:
            with open("kg_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                session.graph_nodes = data.get("nodes", [])
                session.graph_relationships = data.get("relationships", [])
                session.document_name = data.get("document_name", "")
            if session.graph_nodes:
                session.ucs_engine.load_concept_graph(session.graph_nodes, session.graph_relationships)
            print(f"AIVA Backend: Restored {len(session.graph_nodes)} nodes from kg_data.json (UCS concept graph initialized)")
        except Exception as e:
            print(f"Error loading kg_data.json on startup: {e}")
            
    print("AIVA Backend: All services and ML models loaded successfully.")


class AsyncThreadLock:
    """Thread-safe and asyncio-compatible lock for concurrent requests across event loops and OS threads."""
    def __init__(self):
        self._lock = threading.RLock()

    async def __aenter__(self):
        self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

class VivaSession:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.created_at = datetime.datetime.now().isoformat()
        self.lock = AsyncThreadLock()
        self.ucs_engine = UCSEngine()
        self.reset()
        
    def reset(self):
        self.chat_history = ""
        self.conversation_history = []
        self.difficulty_history = [1] # Initial difficulty is 1 (Easy)
        self.performance_data = []
        self.asked_questions = []
        self.current_difficulty = 1 # 0 to 4
        self.graph_nodes = []
        self.graph_relationships = []
        self.overall_coverage = 0.0
        self.last_fluency_report = None
        self.document_name = ""
        self.turn_audit_trail = []
        self.ucs_engine = UCSEngine()

session_store: Dict[str, VivaSession] = {}
session_store_lock = threading.Lock()
DEFAULT_SESSION_ID = "default"

def get_or_create_session(session_id: Optional[str] = None) -> VivaSession:
    sid = session_id.strip() if (session_id and isinstance(session_id, str) and session_id.strip()) else DEFAULT_SESSION_ID
    with session_store_lock:
        if sid not in session_store:
            session_store[sid] = VivaSession(session_id=sid)
        return session_store[sid]

# Global session alias for backwards compatibility
session = get_or_create_session(DEFAULT_SESSION_ID)


@app.get("/")
def read_root():
    return {"message": "AIVA Backend is running!"}

@app.post("/upload_and_build_kg")
async def upload_and_build_kg(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    target_sid = session_id or x_session_id or str(uuid.uuid4())[:8]
    sess = get_or_create_session(target_sid)
    
    tmp_path = f"temp_{target_sid}_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        async with sess.lock:
            # Reset target session for a fresh examination
            sess.reset()
            sess.document_name = file.filename
            
            # Extract Text
            parser = DocumentParser()
            text = parser.extract_text(tmp_path)
            
            if not text.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from the document.")

            # Build Knowledge Graph
            kg_builder = KGBuilder()
            extraction = kg_builder.extract_graph_from_text(text)
            
            if not extraction:
                raise HTTPException(status_code=500, detail="Failed to extract knowledge graph using AI.")

            # Build NetworkX and visualize
            kg_builder.build_networkx_graph(extraction)
            output_html = kg_builder.generate_html_visualization(f"kg_visualization_{target_sid}.html")
            
            # Store in session state
            sess.graph_nodes = [node.model_dump() for node in extraction.nodes]
            sess.graph_relationships = [rel.model_dump() for rel in extraction.relationships]
            
            # Initialize the Unresolved Concept Selection (UCS) engine with the extracted concept graph
            sess.ucs_engine.load_concept_graph(sess.graph_nodes, sess.graph_relationships)
            
            # Index document into Vector DB scoped to target_sid
            chunks_indexed = rag_engine.index_document(text, document_id=file.filename, session_id=target_sid)
            
            # Read graph metadata
            graph_metadata = {}
            if os.path.exists("graph_metadata.json"):
                with open("graph_metadata.json", "r", encoding="utf-8") as f:
                    graph_metadata = json.load(f)
                    
            return {
                "message": "Knowledge Graph built and document indexed successfully!",
                "session_id": target_sid,
                "document_name": sess.document_name,
                "nodes": sess.graph_nodes,
                "relationships": sess.graph_relationships,
                "visualization_path": output_html,
                "chunks_indexed": chunks_indexed,
                "metadata": graph_metadata
            }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.get("/kg_visualization", response_class=HTMLResponse)
async def get_kg_visualization(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    target_sid = session_id or x_session_id or DEFAULT_SESSION_ID
    viz_file = f"kg_visualization_{target_sid}.html"
    if os.path.exists(viz_file):
        with open(viz_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    elif os.path.exists("kg_visualization.html"):
        with open("kg_visualization.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>No visualization found. Please upload a document first.</h1>", status_code=404)

@app.get("/session")
async def get_session(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    target_sid = session_id or x_session_id or DEFAULT_SESSION_ID
    sess = get_or_create_session(target_sid)
    
    # Read graph metadata from file if exists
    graph_metadata = {}
    if os.path.exists("graph_metadata.json"):
        try:
            with open("graph_metadata.json", "r", encoding="utf-8") as f:
                graph_metadata = json.load(f)
        except Exception:
            pass
            
    # Count chunks indexed
    chunks_count = len(sess.graph_nodes) * 2 if sess.graph_nodes else 0
            
    return {
        "session_id": target_sid,
        "active": len(sess.graph_nodes) > 0,
        "document_name": sess.document_name,
        "nodes": sess.graph_nodes,
        "relationships": sess.graph_relationships,
        "chunks_indexed": chunks_count,
        "metadata": graph_metadata,
        "chat_history": sess.conversation_history,
        "current_difficulty": sess.current_difficulty,
        "difficulty_history": sess.difficulty_history,
        "performance_data": sess.performance_data,
        "last_fluency_report": sess.last_fluency_report,
        "turn_audit_trail": sess.turn_audit_trail,
        "ucs_summary": sess.ucs_engine.get_coverage_summary(),
        "ucs_weights": sess.ucs_engine.get_weights()
    }

@app.post("/reset_examination")
async def reset_examination(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    target_sid = session_id or x_session_id or DEFAULT_SESSION_ID
    sess = get_or_create_session(target_sid)
    async with sess.lock:
        sess.chat_history = ""
        sess.conversation_history = []
        sess.difficulty_history = [1]
        sess.performance_data = []
        sess.asked_questions = []
        sess.current_difficulty = 1
        sess.overall_coverage = 0.0
        sess.last_fluency_report = None
        sess.turn_audit_trail = []
        sess.ucs_engine = UCSEngine()
        return {"message": f"Examination for session '{target_sid}' reset successfully.", "session_id": target_sid}

@app.get("/audit_trail")
async def get_audit_trail(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Returns the complete, reproducible audit log of examiner routing and turn decisions for the specified session."""
    target_sid = session_id or x_session_id or DEFAULT_SESSION_ID
    sess = get_or_create_session(target_sid)
    return {
        "session_id": target_sid,
        "total_turns": len(sess.turn_audit_trail),
        "audit_trail": sess.turn_audit_trail
    }

@app.get("/research_metadata")
async def get_research_metadata():
    """Returns experimental configurations, baseline settings, and synthetic model qualifications."""
    return {
        "framework": "AIVA (Adaptive Intelligent Viva Assessment)",
        "multi_agent_personas": ["Professor", "Industry Expert", "Critic", "Examiner"],
        "routing_strategies_supported": ["keyword_balanced", "round_robin", "ucs_guided"],
        "question_selection_strategies_supported": ["ucs", "difficulty_only"],
        "adaptive_controllers_supported": ["dqn", "rule_based", "fixed"],
        "rubric_dimensions": [
            {"name": "correctness", "weight": 0.35, "description": "Technical veracity"},
            {"name": "completeness", "weight": 0.25, "description": "Core facet coverage"},
            {"name": "evidence_alignment", "weight": 0.20, "description": "Report context support"},
            {"name": "reasoning_depth", "weight": 0.20, "description": "Trade-offs and architectural rationale"}
        ],
        "rag_configuration": {
            "retrieval_architecture": "Hybrid Reciprocal Rank Fusion",
            "k_parameter": 60,
            "sparse_engine": "BM25Okapi with technical tokenization",
            "dense_engine": "Google Gemini Embedding 2 (cosine similarity)"
        },
        "synthetic_models_qualification": {
            "is_synthetic_simulation": True,
            "models": ["XGBoost Weakness Classifier", "Logistic Regression Readiness Predictor"],
            "provenance": "Rule-parameterized synthetic demonstration models for latency and architectural verification. Empirical human validation pending Phase 2 data collection."
        }
    }

class DifficultyRequest(BaseModel):
    current_difficulty: int
    last_score: float


@app.post("/next_difficulty")
async def next_difficulty(request: DifficultyRequest):
    try:
        result = adaptive_engine.get_next_difficulty(request.current_difficulty, request.last_score)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MultiAgentRequest(BaseModel):
    conversation_history: List[dict]
    agent_type: str
    difficulty: int = 5
    is_final_turn: bool = False
    routing_strategy: Optional[str] = "keyword_balanced"  # Supports 'keyword_balanced', 'round_robin', 'ucs_guided'
    controller_type: Optional[str] = "dqn"  # Supports 'dqn', 'rule_based', or 'fixed'
    question_selection_strategy: Optional[str] = "ucs"  # Supports 'ucs' (default, knowledge-gap-aware) or 'difficulty_only'
    session_id: Optional[str] = None

@app.post("/multi_agent_question")
async def generate_multi_agent_question(
    request: MultiAgentRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    try:
        target_sid = request.session_id or x_session_id or DEFAULT_SESSION_ID
        sess = get_or_create_session(target_sid)
        
        async with sess.lock:
            # 1. Update the local conversation history
            sess.conversation_history = request.conversation_history
            
            # Build formatted chat history string preserving persona tags
            chat_history_str = ""
            for msg in request.conversation_history:
                if msg.get("role") == "user":
                    role_name = "Student"
                else:
                    agent_name = msg.get("agent")
                    if agent_name:
                        role_name = f"Examiner ({agent_name})"
                    else:
                        role_name = "Examiner"
                chat_history_str += f"{role_name}: {msg.get('content', '')}\n"
                
            sess.chat_history = chat_history_str
            
            # 2. Check if this is the start of the session (no student answers yet)
            student_answers = [m for m in request.conversation_history if m.get("role") == "user"]
            
            evaluation = None
            action_taken = "KEEP"
            eval_provenance = {
                "execution_mode": "not_applicable",
                "provider": "google",
                "model": "gemini-2.5-flash",
                "concept_validation_status": "initial_turn"
            }
            
            # If student has answered, evaluate their last answer
            if len(student_answers) > 0:
                last_student_answer = student_answers[-1]["content"]
                
                # 2.5 Calculate fluency of the answer
                sess.last_fluency_report = analytics_engine.analyze_fluency(last_student_answer)
                
                # Find the previous question asked by the agent
                agent_questions = [m for m in request.conversation_history[:-1] if m.get("role") != "user"]
                prev_question = agent_questions[-1]["content"] if agent_questions else "Introduction"
                
                # Identify the concept probed in the previous turn
                prev_audit = sess.turn_audit_trail[-1] if sess.turn_audit_trail else {}
                prev_concept = prev_audit.get("selected_concept")

                # Retrieve relevant context from RAG scoped to this session
                context_chunks = rag_engine.query(last_student_answer, n_results=3, session_id=target_sid)
                retrieved_context = "\n\n".join(context_chunks)
                
                # Run 4-Dimensional Rubric Evaluation using structured output LLM
                prompt = f"""
                You are an academic viva examination evaluator assessing a university student's defense answer.
                Evaluate the student's answer along these four distinct academic rubric dimensions (each scored 0.0 to 1.0):
                1. correctness: Technical accuracy and domain correctness.
                2. completeness: Extent to which the response addresses the core components of the examiner's question.
                3. evidence_alignment: Grounding in and consistency with the provided project report context.
                4. reasoning_depth: Depth of explanation articulate 'why' specific choices/methods were made and trade-offs.

                Previous Question Asked: {prev_question}
                Target Concept Probed: {prev_concept or 'General Architecture'}
                Student Answer: {last_student_answer}
                Retrieved Report Context:
                {retrieved_context}
                """
                is_live_eval = True
                eval_error = None
                try:
                    eval_result = evaluator_llm.invoke(prompt)
                    composite_score = analytics_engine.calculate_rubric_composite(
                        correctness=eval_result.correctness,
                        completeness=eval_result.completeness,
                        evidence_alignment=eval_result.evidence_alignment,
                        reasoning_depth=eval_result.reasoning_depth
                    )
                    suggested_concept = eval_result.concept
                    evaluation = {
                        "correctness": eval_result.correctness,
                        "completeness": eval_result.completeness,
                        "evidence_alignment": eval_result.evidence_alignment,
                        "reasoning_depth": eval_result.reasoning_depth,
                        "score": composite_score,
                        "feedback": eval_result.feedback,
                        "concept": suggested_concept
                    }
                except Exception as e:
                    is_live_eval = False
                    eval_error = str(e)
                    print(f"Error in LLM evaluation (using fallback): {repr(e)}")
                    suggested_concept = None
                    evaluation = {
                        "correctness": 0.5,
                        "completeness": 0.5,
                        "evidence_alignment": 0.5,
                        "reasoning_depth": 0.5,
                        "score": 0.5,
                        "feedback": "Evaluation fallback due to API rate limit or error.",
                        "concept": prev_concept or "General Architecture"
                    }
                
                # Scientific Concept Reconciliation: Verify evaluated concept against concept graph
                validated_concept, val_status = sess.ucs_engine.reconcile_concept(
                    suggested_concept=suggested_concept,
                    fallback_concept=prev_concept
                )
                evaluation["concept"] = validated_concept
                
                eval_provenance = {
                    "execution_mode": "live_model" if is_live_eval else "deterministic_fallback",
                    "provider": "google",
                    "model": "gemini-2.5-flash",
                    "concept_validation_status": val_status,
                    "target_probed_concept": prev_concept,
                    "evaluated_concept": validated_concept,
                    "error": eval_error
                }
                    
                # 3. Call adaptive engine (supporting dqn, rule_based, or fixed baseline)
                controller = request.controller_type or "dqn"
                rl_decision = adaptive_engine.get_next_difficulty(
                    current_difficulty=sess.current_difficulty,
                    last_answer_score=evaluation["score"],
                    controller_type=controller
                )
                action_taken = rl_decision["action_taken"]
                sess.current_difficulty = rl_decision["new_difficulty_level"]
                sess.difficulty_history.append(sess.current_difficulty)
                
                # Calculate normalized answer length
                answer_length = min(len(last_student_answer) / 300.0, 1.0)
                
                # Get node coverage for this concept
                coverage_dict = analytics_engine.calculate_coverage(sess.graph_nodes, last_student_answer)
                concept_coverage = coverage_dict["concept_breakdown"].get(validated_concept, 0.0) / 100.0
                
                # 4. Store performance data and update UCS concept state
                performance_entry = {
                    "concept": validated_concept,
                    "score": evaluation["score"],
                    "difficulty": sess.current_difficulty,
                    "coverage": concept_coverage,
                    "answer_length": answer_length,
                    "confidence": 0.8
                }
                sess.performance_data.append(performance_entry)
                
                # Update UCS engine with evaluated concept mastery
                sess.ucs_engine.update_concept_score(
                    concept_id=validated_concept,
                    score=evaluation["score"],
                    turn=len(sess.asked_questions)
                )
                
            # 5. Question Selection Strategy & Context Retrieval
            q_strategy = request.question_selection_strategy or "ucs"
            selected_concept = None
            ucs_metrics = {}
            recommended_persona = None
            
            if q_strategy == "ucs":
                # Lazily populate concept graph if nodes exist but engine state is empty
                if len(sess.ucs_engine.concept_states) == 0 and len(sess.graph_nodes) > 0:
                    sess.ucs_engine.load_concept_graph(sess.graph_nodes, sess.graph_relationships)
                    
                ucs_decision = sess.ucs_engine.select_next_concept(turn=len(sess.asked_questions))
                selected_concept = ucs_decision["selected_concept"]
                ucs_metrics = ucs_decision["score_metrics"]
                recommended_persona = ucs_decision["recommended_persona"]
                
                # Knowledge-gap grounded query: anchor RAG search to targeted concept scoped to session
                query_text = f"{selected_concept} overview architecture implementation algorithms"
                next_context_chunks = rag_engine.query(query_text, n_results=3, session_id=target_sid)
                next_context = "\n\n".join(next_context_chunks)
            else:
                # Baseline: Difficulty-only conversational local retrieval
                query_text = student_answers[-1]["content"] if student_answers else "Overview of the project report"
                next_context_chunks = rag_engine.query(query_text, n_results=3, session_id=target_sid)
                next_context = "\n\n".join(next_context_chunks)
            
            # 6. Route to selected agent and generate the next question
            difficulty_mapping = {
                0: "Fundamentals (Very Easy)",
                1: "Easy",
                2: "Medium",
                3: "Hard",
                4: "Research Level (Very Hard)"
            }
            difficulty_name = difficulty_mapping[sess.current_difficulty]
            
            # Run Multi Agent Panel with traceable routing and optional UCS guidance
            strategy = request.routing_strategy or "keyword_balanced"
            is_live_q = True
            q_error = None
            try:
                panel_result = multi_agent_panel.run_panel(
                    context=next_context,
                    chat_history=sess.chat_history,
                    difficulty=difficulty_name,
                    agent_type=request.agent_type,
                    asked_questions=sess.asked_questions,
                    is_final_turn=request.is_final_turn,
                    routing_strategy=strategy,
                    target_concept=selected_concept,
                    recommended_persona=recommended_persona
                )
                next_question = panel_result["question"]
                actual_agent = panel_result.get("selected_agent", "Examiner")
                routing_reason = panel_result.get("routing_reason", "Automated selection")
            except Exception as e:
                is_live_q = False
                q_error = str(e)
                print(f"Error in Multi-Agent Panel: {repr(e)}")
                next_question = f"In your implementation of '{selected_concept or 'System Architecture'}', how does it handle performance and reliability trade-offs?"
                actual_agent = recommended_persona or request.agent_type
                routing_reason = f"Persona fallback due to API: {e}"

            sess.asked_questions.append(next_question)
            
            q_provenance = {
                "execution_mode": "live_model" if is_live_q else "deterministic_fallback",
                "provider": "google",
                "model": "gemini-2.5-flash",
                "error": q_error
            }
            
            # 7. Record Turn Audit Trail for scientific evaluation and reproducibility
            turn_audit_entry = {
                "session_id": target_sid,
                "turn_index": len(sess.asked_questions),
                "selected_agent": actual_agent,
                "routing_reason": routing_reason,
                "routing_strategy": strategy,
                "controller_type": request.controller_type or "dqn",
                "question_selection_strategy": q_strategy,
                "selected_concept": selected_concept,
                "ucs_metrics": ucs_metrics,
                "ucs_weights": sess.ucs_engine.get_weights(),
                "question": next_question,
                "difficulty_level": sess.current_difficulty,
                "difficulty_name": difficulty_name,
                "evaluation": evaluation,
                "evaluation_provenance": eval_provenance,
                "question_generation_provenance": q_provenance,
                "action_taken": action_taken,
                "retrieved_context_snippets": next_context_chunks[:2]
            }
            sess.turn_audit_trail.append(turn_audit_entry)
                
            # 8. Compute updated analytics
            coverage_data = analytics_engine.calculate_coverage(sess.graph_nodes, sess.chat_history)
            sess.overall_coverage = coverage_data["overall_coverage_percent"]
            
            # Weaknesses (XGBoost)
            weaknesses = analytics_engine.detect_weaknesses(sess.performance_data)
            
            # Difficulty trend (slope of difficulty history)
            if len(sess.difficulty_history) > 1:
                diff_trend = (sess.difficulty_history[-1] - sess.difficulty_history[0]) / 4.0
            else:
                diff_trend = 0.0
                
            # Readiness (Logistic Regression)
            avg_score = sum(p["score"] for p in sess.performance_data) / len(sess.performance_data) if sess.performance_data else 0.5
            fluency_val = sess.last_fluency_report.get("fluency_score", 100.0) if sess.last_fluency_report else 100.0
            readiness_data = analytics_engine.predict_readiness(
                coverage=sess.overall_coverage,
                avg_score=avg_score,
                difficulty_trend=diff_trend,
                weakness_count=len(weaknesses),
                fluency_score=fluency_val
            )
            
            return {
                "session_id": target_sid,
                "agent": actual_agent,
                "routing_reason": routing_reason,
                "routing_strategy": strategy,
                "question_selection_strategy": q_strategy,
                "selected_concept": selected_concept,
                "ucs_metrics": ucs_metrics,
                "question": next_question,
                "evaluation": evaluation,
                "new_difficulty": sess.current_difficulty,
                "action_taken": action_taken,
                "evaluation_provenance": eval_provenance,
                "question_generation_provenance": q_provenance,
                "analytics": {
                    "coverage": coverage_data,
                    "weaknesses": {
                        "weaknesses": weaknesses
                    },
                    "readiness_prediction": readiness_data,
                    "difficulty_history": sess.difficulty_history,
                    "fluency": sess.last_fluency_report,
                    "ucs_summary": sess.ucs_engine.get_coverage_summary()
                }
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class AnalyticsRequest(BaseModel):
    graph_nodes: Optional[list] = None
    chat_history: Optional[str] = None
    performance_data: Optional[list] = None
    session_id: Optional[str] = None

@app.post("/analytics")
async def get_analytics_post(
    request: Optional[AnalyticsRequest] = None,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    session_id: Optional[str] = Query(None)
):
    try:
        target_sid = (request.session_id if request else None) or session_id or x_session_id or DEFAULT_SESSION_ID
        sess = get_or_create_session(target_sid)
        
        nodes = (request.graph_nodes if request and request.graph_nodes else None) or sess.graph_nodes
        chat_hist = (request.chat_history if request and request.chat_history else None) or sess.chat_history
        perf = (request.performance_data if request and request.performance_data else None) or sess.performance_data
        
        coverage_data = analytics_engine.calculate_coverage(nodes, chat_hist)
        weaknesses = analytics_engine.detect_weaknesses(perf)
        
        if len(sess.difficulty_history) > 1:
            diff_trend = (sess.difficulty_history[-1] - sess.difficulty_history[0]) / 4.0
        else:
            diff_trend = 0.0
            
        avg_score = sum(p["score"] for p in perf) / len(perf) if perf else 0.5
        fluency_val = sess.last_fluency_report.get("fluency_score", 100.0) if sess.last_fluency_report else 100.0
        readiness_data = analytics_engine.predict_readiness(
            coverage=coverage_data["overall_coverage_percent"],
            avg_score=avg_score,
            difficulty_trend=diff_trend,
            weakness_count=len(weaknesses),
            fluency_score=fluency_val
        )
        
        return {
            "session_id": target_sid,
            "coverage": coverage_data,
            "weaknesses": {
                "weaknesses": weaknesses
            },
            "readiness_prediction": readiness_data,
            "difficulty_history": sess.difficulty_history,
            "fluency": sess.last_fluency_report,
            "ucs_summary": sess.ucs_engine.get_coverage_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics_get(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    return await get_analytics_post(request=None, x_session_id=x_session_id, session_id=session_id)

class RAGRequest(BaseModel):
    query: str

@app.post("/rag_query")
async def rag_query(request: RAGRequest):
    try:
        results = rag_engine.query(request.query)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TutorRequest(BaseModel):
    question: str
    conversation_history: List[dict] = []

@app.post("/tutor_answer")
async def get_tutor_answer(request: TutorRequest):
    try:
        global rag_engine, tutor_llm
        if not rag_engine:
            raise HTTPException(status_code=500, detail="RAG Engine is not initialized. Please upload a document first.")
            
        # Retrieve relevant context from RAG
        context_chunks = rag_engine.query(request.question, n_results=4)
        retrieved_context = "\n\n".join(context_chunks)
        
        # Build chat history string
        chat_history_str = ""
        for msg in request.conversation_history:
            role_name = "Student" if msg["role"] == "user" else "Tutor"
            chat_history_str += f"{role_name}: {msg['content']}\n"
            
        # Call LLM to answer the question
        prompt = f"""
        You are a helpful and knowledgeable tutor for a student's academic project.
        Your task is to answer the student's questions based ONLY on the provided project context.
        
        If the answer is not contained within the provided context, politely inform the student that you cannot answer the question based on the uploaded document. Do not hallucinate or guess information outside the context.
        
        Project Context:
        {retrieved_context}
        
        Conversation History:
        {chat_history_str}
        
        Student's Question: {request.question}
        
        Provide a clear, accurate, and concise answer to the student's question.
        Tutor:
        """
        
        from langchain_core.messages import HumanMessage
        response = tutor_llm.invoke([HumanMessage(content=prompt)])
        return {"answer": response.content.strip()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        output_file = "response_audio.mp3"
        voice_engine.text_to_speech(request.text, output_file)
        return FileResponse(output_file, media_type="audio/mpeg", filename="response.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    tmp_audio = f"temp_{file.filename}"
    try:
        with open(tmp_audio, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        text = voice_engine.speech_to_text(tmp_audio)
        return {"transcription": text}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)
