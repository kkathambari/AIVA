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
from typing import List
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="VivaForge V2 Backend")

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
    score: float = Field(description="Score between 0.0 and 1.0. 1.0 is perfect, 0.0 is completely incorrect or empty.")
    feedback: str = Field(description="Brief explanation of the score and what was missing or incorrect.")
    concept: str = Field(description="The primary concept tested by the question, e.g. 'Random Forest', 'FastAPI'. Must be short (1-3 words).")

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
    print("VivaForge Backend: All services and ML models loaded successfully.")

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

session = VivaSession()

@app.get("/")
def read_root():
    return {"message": "VivaForge Backend is running!"}

@app.post("/upload_and_build_kg")
async def upload_and_build_kg(file: UploadFile = File(...)):
    tmp_path = f"temp_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Reset session for a new examination
        session.reset()
        
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
        output_html = kg_builder.generate_html_visualization("kg_visualization.html")
        
        # Store in session state
        session.graph_nodes = [node.model_dump() for node in extraction.nodes]
        session.graph_relationships = [rel.model_dump() for rel in extraction.relationships]
        
        # Index document into Vector DB
        chunks_indexed = rag_engine.index_document(text, document_id=file.filename)
        
        # Read graph metadata
        graph_metadata = {}
        if os.path.exists("graph_metadata.json"):
            with open("graph_metadata.json", "r", encoding="utf-8") as f:
                graph_metadata = json.load(f)
                
        return {
            "message": "Knowledge Graph built and document indexed successfully!",
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

def determine_agent_spoke(chat_history: str) -> str:
    # Extract last student message
    last_student_msg = ""
    lines = chat_history.strip().split("\n")
    for line in reversed(lines):
        if any(line.lower().startswith(prefix) for prefix in ["student:", "user:", "student response:"]):
            parts = line.split(":", 1)
            if len(parts) > 1:
                last_student_msg = parts[1].strip()
            break
            
    text_to_scan = last_student_msg.lower()
    
    prof_keywords = ["math", "formula", "algorithm", "complexity", "theorem", "proof", "equation", "analysis", "theory", "time complexity", "big-o", "space complexity", "derivation"]
    industry_keywords = ["deploy", "docker", "aws", "cloud", "scale", "latency", "production", "kubernetes", "database", "ci/cd", "pipeline", "dockerfile", "server", "hosting", "architecture", "microservice", "infrastructure"]
    critic_keywords = ["limit", "fail", "drawback", "bottleneck", "weakness", "error", "alternative", "compromise", "flaw", "shortcoming", "disadvantage", "constraint"]
    
    if any(kw in text_to_scan for kw in prof_keywords):
        return "Professor"
    elif any(kw in text_to_scan for kw in industry_keywords):
        return "Industry Expert"
    elif any(kw in text_to_scan for kw in critic_keywords):
        return "Critic"
    else:
        # Turn balancing: choose who has spoken the least
        counts = {
            "Professor": chat_history.lower().count("professor:"),
            "Industry Expert": chat_history.lower().count("industry expert:"),
            "Critic": chat_history.lower().count("research critic:"),
            "Examiner": chat_history.lower().count("examiner:")
        }
        return min(counts, key=counts.get)

@app.post("/multi_agent_question")
async def generate_multi_agent_question(request: MultiAgentRequest):
    try:
        global session
        
        # 1. Update the local conversation history
        session.conversation_history = request.conversation_history
        
        # Build formatted chat history string
        chat_history_str = ""
        for msg in request.conversation_history:
            role_name = "Student" if msg["role"] == "user" else "Examiner"
            chat_history_str += f"{role_name}: {msg['content']}\n"
            
        session.chat_history = chat_history_str
        
        # 2. Check if this is the start of the session (no student answers yet)
        student_answers = [m for m in request.conversation_history if m["role"] == "user"]
        
        evaluation = {
            "score": 0.5,
            "feedback": "Initial question of the session.",
            "concept": "Introduction"
        }
        action_taken = "KEEP"
        
        # If student has answered, evaluate their last answer
        if len(student_answers) > 0:
            last_student_answer = student_answers[-1]["content"]
            
            # Find the previous question asked by the agent
            agent_questions = [m for m in request.conversation_history[:-1] if m["role"] != "user"]
            prev_question = agent_questions[-1]["content"] if agent_questions else "Introduction"
            
            # Retrieve relevant context from RAG
            context_chunks = rag_engine.query(last_student_answer, n_results=3)
            retrieved_context = "\n\n".join(context_chunks)
            
            # Run Evaluation using Gemini structure
            prompt = f"""
            Previous Question asked: {prev_question}
            Student Answer: {last_student_answer}
            Context from Project: {retrieved_context}
            """
            try:
                eval_result = evaluator_llm.invoke(prompt)
                evaluation = {
                    "score": eval_result.score,
                    "feedback": eval_result.feedback,
                    "concept": eval_result.concept
                }
            except Exception as e:
                print(f"Error in LLM evaluation: {e}")
                # Fallback evaluation
                evaluation = {
                    "score": 0.5,
                    "feedback": "Evaluation fallback due to API error.",
                    "concept": "General"
                }
                
            # 3. Call RL adaptive engine to adjust difficulty based on the score
            rl_decision = adaptive_engine.get_next_difficulty(session.current_difficulty, evaluation["score"])
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
        
        # Run Multi Agent Panel
        panel_result = multi_agent_panel.run_panel(
            context=next_context,
            chat_history=session.chat_history,
            difficulty=difficulty_name,
            agent_type=request.agent_type,
            asked_questions=session.asked_questions
        )
        
        next_question = panel_result["question"]
        session.asked_questions.append(next_question)
        
        # Determine actual agent type in case of Auto routing
        actual_agent = request.agent_type
        if request.agent_type.lower() in ["auto", "panel", "auto (panel)"]:
            actual_agent = determine_agent_spoke(session.chat_history)
            
        # 7. Compute updated analytics
        # Coverage
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
        readiness_data = analytics_engine.predict_readiness(
            coverage=session.overall_coverage,
            avg_score=avg_score,
            difficulty_trend=diff_trend,
            weakness_count=len(weaknesses)
        )
        
        return {
            "agent": actual_agent,
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
                "difficulty_history": session.difficulty_history
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
        readiness_data = analytics_engine.predict_readiness(
            coverage=session.overall_coverage,
            avg_score=avg_score,
            difficulty_trend=diff_trend,
            weakness_count=len(weaknesses)
        )
        
        return {
            "coverage": coverage_data,
            "weaknesses": {
                "weaknesses": weaknesses
            },
            "readiness_prediction": readiness_data,
            "difficulty_history": session.difficulty_history
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
