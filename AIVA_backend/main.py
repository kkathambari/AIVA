from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from core.parser import DocumentParser
from core.knowledge_graph import KGBuilder
from core.adaptive_engine import AdaptiveEngine
from core.multi_agent import MultiAgentPanel
from core.analytics_engine import AnalyticsEngine
from core.rag_engine import RAGEngine
from core.voice_engine import VoiceEngine
import shutil
import os
import json
from typing import List, Optional
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
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0).with_structured_output(AnswerEvaluation)
    tutor_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # Load existing KG data from file if present to restore session
    if os.path.exists("kg_data.json"):
        try:
            with open("kg_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                session.graph_nodes = data.get("nodes", [])
                session.graph_relationships = data.get("relationships", [])
                session.document_name = data.get("document_name", "")
            print(f"AIVA Backend: Restored {len(session.graph_nodes)} nodes from kg_data.json")
        except Exception as e:
            print(f"Error loading kg_data.json on startup: {e}")
            
    print("AIVA Backend: All services and ML models loaded successfully.")


class VivaSession:
    def __init__(self):
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

session = VivaSession()


@app.get("/")
def read_root():
    return {"message": "AIVA Backend is running!"}

@app.post("/upload_and_build_kg")
async def upload_and_build_kg(file: UploadFile = File(...)):
    tmp_path = f"temp_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Reset session for a new examination
        session.reset()
        session.document_name = file.filename
        
        # Extract Text
        parser = DocumentParser()
        text = parser.extract_text(tmp_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        # Clear old embeddings from vector store
        rag_engine.clear()

        # Build Knowledge Graph
        kg_builder = KGBuilder()
        extraction = kg_builder.extract_graph_from_text(text)
        
        if not extraction:
            raise HTTPException(status_code=500, detail="Failed to extract knowledge graph using AI.")

        # Build NetworkX and visualize
        kg_builder.build_networkx_graph(extraction)
        output_html = kg_builder.generate_html_visualization("kg_visualization.html")
        
        # Store in session state
        session.graph_nodes = [node.model_dump() for node in extraction.nodes]
        session.graph_relationships = [rel.model_dump() for rel in extraction.relationships]
        
        # Save kg_data.json with document_name for persistence across restarts
        try:
            with open("kg_data.json", "w", encoding="utf-8") as f:
                json.dump({
                    "document_name": session.document_name,
                    "nodes": session.graph_nodes,
                    "relationships": session.graph_relationships
                }, f, ensure_ascii=False, indent=2)
        except Exception as err:
            print(f"Error persisting kg_data.json: {err}")

        # Index document into Vector DB
        chunks_indexed = rag_engine.index_document(text, document_id=file.filename)
        
        # Read graph metadata
        graph_metadata = {}
        if os.path.exists("graph_metadata.json"):
            with open("graph_metadata.json", "r", encoding="utf-8") as f:
                graph_metadata = json.load(f)
                
        return {
            "message": "Knowledge Graph built and document indexed successfully!",
            "document_name": session.document_name,
            "nodes": session.graph_nodes,
            "relationships": session.graph_relationships,
            "visualization_path": output_html,
            "chunks_indexed": chunks_indexed,
            "metadata": graph_metadata
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/kg_visualization", response_class=HTMLResponse)
async def get_kg_visualization():
    if os.path.exists("kg_visualization.html"):
        with open("kg_visualization.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>No visualization found. Please upload a document first.</h1>", status_code=404)

@app.get("/session")
async def get_session():
    # Read graph metadata from file if exists
    graph_metadata = {}
    if os.path.exists("graph_metadata.json"):
        try:
            with open("graph_metadata.json", "r", encoding="utf-8") as f:
                graph_metadata = json.load(f)
        except Exception:
            pass
            
    # Count chunks indexed
    chunks_count = 0
    if os.path.exists("chroma_db/bm25_chunks.json"):
        try:
            with open("chroma_db/bm25_chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
                chunks_count = len(chunks)
        except Exception:
            pass
            
    return {
        "active": len(session.graph_nodes) > 0,
        "document_name": session.document_name,
        "nodes": session.graph_nodes,
        "relationships": session.graph_relationships,
        "chunks_indexed": chunks_count if chunks_count > 0 else (len(session.graph_nodes) * 2 if session.graph_nodes else 0),
        "metadata": graph_metadata,
        "chat_history": session.conversation_history,
        "current_difficulty": session.current_difficulty,
        "difficulty_history": session.difficulty_history,
        "performance_data": session.performance_data,
        "last_fluency_report": session.last_fluency_report,
        "turn_audit_trail": session.turn_audit_trail
    }

@app.post("/reset_examination")
async def reset_examination():
    session.chat_history = ""
    session.conversation_history = []
    session.difficulty_history = [1]
    session.performance_data = []
    session.asked_questions = []
    session.current_difficulty = 1
    session.overall_coverage = 0.0
    session.last_fluency_report = None
    session.turn_audit_trail = []
    return {"message": "Examination reset successfully."}

@app.get("/audit_trail")
async def get_audit_trail():
    """Returns the complete, reproducible audit log of examiner routing and turn decisions."""
    return {
        "total_turns": len(session.turn_audit_trail),
        "audit_trail": session.turn_audit_trail
    }

@app.get("/research_metadata")
async def get_research_metadata():
    """Returns experimental configurations, baseline settings, and synthetic model qualifications."""
    return {
        "framework": "AIVA (Adaptive Intelligent Viva Assessment)",
        "multi_agent_personas": ["Professor", "Industry Expert", "Critic", "Examiner"],
        "routing_strategies_supported": ["keyword_balanced", "round_robin"],
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
    routing_strategy: Optional[str] = "keyword_balanced"  # Supports 'keyword_balanced' or 'round_robin'
    controller_type: Optional[str] = "dqn"  # Supports 'dqn', 'rule_based', or 'fixed'

@app.post("/multi_agent_question")
async def generate_multi_agent_question(request: MultiAgentRequest):
    try:
        global session
        
        # 1. Update the local conversation history
        session.conversation_history = request.conversation_history
        
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
            
        session.chat_history = chat_history_str
        
        # 2. Check if this is the start of the session (no student answers yet)
        student_answers = [m for m in request.conversation_history if m.get("role") == "user"]
        
        evaluation = None
        action_taken = "KEEP"
        
        # If student has answered, evaluate their last answer
        if len(student_answers) > 0:
            last_student_answer = student_answers[-1]["content"]
            
            # 2.5 Calculate fluency of the answer
            session.last_fluency_report = analytics_engine.analyze_fluency(last_student_answer)
            
            # Find the previous question asked by the agent
            agent_questions = [m for m in request.conversation_history[:-1] if m.get("role") != "user"]
            prev_question = agent_questions[-1]["content"] if agent_questions else "Introduction"
            
            # Retrieve relevant context from RAG (using default hybrid RRF)
            context_chunks = rag_engine.query(last_student_answer, n_results=3)
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
            Student Answer: {last_student_answer}
            Retrieved Report Context:
            {retrieved_context}
            """
            try:
                eval_result = evaluator_llm.invoke(prompt)
                composite_score = analytics_engine.calculate_rubric_composite(
                    correctness=eval_result.correctness,
                    completeness=eval_result.completeness,
                    evidence_alignment=eval_result.evidence_alignment,
                    reasoning_depth=eval_result.reasoning_depth
                )
                evaluation = {
                    "correctness": eval_result.correctness,
                    "completeness": eval_result.completeness,
                    "evidence_alignment": eval_result.evidence_alignment,
                    "reasoning_depth": eval_result.reasoning_depth,
                    "score": composite_score,
                    "feedback": eval_result.feedback,
                    "concept": eval_result.concept
                }
            except Exception as e:
                print(f"Error in LLM evaluation: {repr(e)}")
                # Robust fallback evaluation
                evaluation = {
                    "correctness": 0.5,
                    "completeness": 0.5,
                    "evidence_alignment": 0.5,
                    "reasoning_depth": 0.5,
                    "score": 0.5,
                    "feedback": "Evaluation fallback due to API error.",
                    "concept": "General"
                }
                
            # 3. Call adaptive engine (supporting dqn, rule_based, or fixed baseline)
            controller = request.controller_type or "dqn"
            rl_decision = adaptive_engine.get_next_difficulty(
                current_difficulty=session.current_difficulty,
                last_answer_score=evaluation["score"],
                controller_type=controller
            )
            action_taken = rl_decision["action_taken"]
            session.current_difficulty = rl_decision["new_difficulty_level"]
            session.difficulty_history.append(session.current_difficulty)
            
            # Calculate normalized answer length
            answer_length = min(len(last_student_answer) / 300.0, 1.0)
            
            # Get node coverage for this concept
            coverage_dict = analytics_engine.calculate_coverage(session.graph_nodes, last_student_answer)
            concept_coverage = coverage_dict["concept_breakdown"].get(evaluation["concept"], 0.0) / 100.0
            
            # 4. Store performance data for weakness detection
            performance_entry = {
                "concept": evaluation["concept"],
                "score": evaluation["score"],
                "difficulty": session.current_difficulty,
                "coverage": concept_coverage,
                "answer_length": answer_length,
                "confidence": 0.8
            }
            session.performance_data.append(performance_entry)
            
        # 5. Retrieve Context for the Next Question
        query_text = student_answers[-1]["content"] if student_answers else "Overview of the project report"
        next_context_chunks = rag_engine.query(query_text, n_results=3)
        next_context = "\n\n".join(next_context_chunks)
        
        # 6. Route to selected agent and generate the next question
        difficulty_mapping = {
            0: "Fundamentals (Very Easy)",
            1: "Easy",
            2: "Medium",
            3: "Hard",
            4: "Research Level (Very Hard)"
        }
        difficulty_name = difficulty_mapping[session.current_difficulty]
        
        # Run Multi Agent Panel with traceable routing
        strategy = request.routing_strategy or "keyword_balanced"
        try:
            panel_result = multi_agent_panel.run_panel(
                context=next_context,
                chat_history=session.chat_history,
                difficulty=difficulty_name,
                agent_type=request.agent_type,
                asked_questions=session.asked_questions,
                is_final_turn=request.is_final_turn,
                routing_strategy=strategy
            )
            next_question = panel_result["question"]
            actual_agent = panel_result.get("selected_agent", "Examiner")
            routing_reason = panel_result.get("routing_reason", "Automated selection")
        except Exception as e:
            print(f"Error in Multi-Agent Panel: {repr(e)}")
            next_question = "I apologize, but I am having trouble generating the next question. Could you please elaborate on your previous point?"
            actual_agent = request.agent_type
            routing_reason = f"Fallback error: {e}"

        session.asked_questions.append(next_question)
        
        # 7. Record Turn Audit Trail for scientific evaluation and reproducibility
        turn_audit_entry = {
            "turn_index": len(session.asked_questions),
            "selected_agent": actual_agent,
            "routing_reason": routing_reason,
            "routing_strategy": strategy,
            "controller_type": request.controller_type or "dqn",
            "question": next_question,
            "difficulty_level": session.current_difficulty,
            "difficulty_name": difficulty_name,
            "evaluation": evaluation,
            "action_taken": action_taken,
            "retrieved_context_snippets": next_context_chunks[:2]
        }
        session.turn_audit_trail.append(turn_audit_entry)
            
        # 8. Compute updated analytics
        coverage_data = analytics_engine.calculate_coverage(session.graph_nodes, session.chat_history)
        session.overall_coverage = coverage_data["overall_coverage_percent"]
        
        # Weaknesses (XGBoost)
        weaknesses = analytics_engine.detect_weaknesses(session.performance_data)
        
        # Difficulty trend (slope of difficulty history)
        if len(session.difficulty_history) > 1:
            diff_trend = (session.difficulty_history[-1] - session.difficulty_history[0]) / 4.0
        else:
            diff_trend = 0.0
            
        # Readiness (Logistic Regression)
        avg_score = sum(p["score"] for p in session.performance_data) / len(session.performance_data) if session.performance_data else 0.5
        fluency_val = session.last_fluency_report.get("fluency_score", 100.0) if session.last_fluency_report else 100.0
        readiness_data = analytics_engine.predict_readiness(
            coverage=session.overall_coverage,
            avg_score=avg_score,
            difficulty_trend=diff_trend,
            weakness_count=len(weaknesses),
            fluency_score=fluency_val
        )
        
        return {
            "agent": actual_agent,
            "routing_reason": routing_reason,
            "routing_strategy": strategy,
            "question": next_question,
            "evaluation": evaluation,
            "new_difficulty": session.current_difficulty,
            "action_taken": action_taken,
            "analytics": {
                "coverage": coverage_data,
                "weaknesses": {
                    "weaknesses": weaknesses
                },
                "readiness_prediction": readiness_data,
                "difficulty_history": session.difficulty_history,
                "fluency": session.last_fluency_report
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AnalyticsRequest(BaseModel):
    graph_nodes: list = None
    chat_history: str = None
    performance_data: list = None

@app.post("/analytics")
async def get_analytics_post(request: AnalyticsRequest = None):
    try:
        coverage_data = analytics_engine.calculate_coverage(session.graph_nodes, session.chat_history)
        weaknesses = analytics_engine.detect_weaknesses(session.performance_data)
        
        if len(session.difficulty_history) > 1:
            diff_trend = (session.difficulty_history[-1] - session.difficulty_history[0]) / 4.0
        else:
            diff_trend = 0.0
            
        avg_score = sum(p["score"] for p in session.performance_data) / len(session.performance_data) if session.performance_data else 0.5
        fluency_val = session.last_fluency_report.get("fluency_score", 100.0) if session.last_fluency_report else 100.0
        readiness_data = analytics_engine.predict_readiness(
            coverage=session.overall_coverage,
            avg_score=avg_score,
            difficulty_trend=diff_trend,
            weakness_count=len(weaknesses),
            fluency_score=fluency_val
        )
        
        return {
            "coverage": coverage_data,
            "weaknesses": {
                "weaknesses": weaknesses
            },
            "readiness_prediction": readiness_data,
            "difficulty_history": session.difficulty_history,
            "fluency": session.last_fluency_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics_get():
    return await get_analytics_post()

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
