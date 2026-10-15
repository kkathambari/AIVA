from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os

class AgentState(TypedDict):
    context: str
    chat_history: str
    difficulty: str
    agent_type: str
    routing_strategy: str  # 'keyword_balanced' or 'round_robin'
    selected_agent_node: str
    selected_agent: str
    routing_reason: str
    generated_question: str
    asked_questions: List[str]  # Prevents repeated questions
    is_final_turn: bool

class MultiAgentPanel:
    def __init__(self, api_key: str = None):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_retries=1, timeout=10)
        self.graph = self._build_graph()

    def router_node(self, state: AgentState) -> dict:
        """
        First-class LangGraph router node that computes traceable persona assignment
        and explicit reasoning based on the chosen strategy (keyword_balanced or round_robin).
        """
        agent = state.get("agent_type", "auto").lower()
        strategy = state.get("routing_strategy", "keyword_balanced").lower()
        history = state.get("chat_history", "")
        
        # 1. Direct explicit persona selection
        if agent in ["critic", "research critic"]:
            return {
                "selected_agent_node": "critic",
                "selected_agent": "Critic",
                "routing_reason": "Direct user override: Critic persona requested."
            }
        elif agent in ["industry", "industry expert"]:
            return {
                "selected_agent_node": "industry",
                "selected_agent": "Industry Expert",
                "routing_reason": "Direct user override: Industry Expert persona requested."
            }
        elif agent in ["professor", "university professor"]:
            return {
                "selected_agent_node": "professor",
                "selected_agent": "Professor",
                "routing_reason": "Direct user override: Professor persona requested."
            }
        elif agent in ["examiner", "general examiner"]:
            return {
                "selected_agent_node": "examiner",
                "selected_agent": "Examiner",
                "routing_reason": "Direct user override: General Examiner persona requested."
            }

        # 2. Automated Panel Routing
        # Count previous turns per persona across history by inspecting examiner lines
        counts = {"professor": 0, "industry": 0, "critic": 0, "examiner": 0}
        total_examiner_turns = 0
        for line in history.splitlines():
            line_str = line.strip().lower()
            if not line_str or any(line_str.startswith(p) for p in ["student:", "user:", "student response:"]):
                continue
            total_examiner_turns += 1
            if "professor" in line_str:
                counts["professor"] += 1
            elif "industry" in line_str:
                counts["industry"] += 1
            elif "critic" in line_str:
                counts["critic"] += 1
            else:
                counts["examiner"] += 1

        # Strategy A: Baseline Sequential Round-Robin
        if strategy == "round_robin":
            cycle = ["examiner", "critic", "industry", "professor"]
            chosen_node = cycle[total_examiner_turns % len(cycle)]
            display_names = {
                "examiner": "Examiner",
                "critic": "Critic",
                "industry": "Industry Expert",
                "professor": "Professor"
            }
            return {
                "selected_agent_node": chosen_node,
                "selected_agent": display_names[chosen_node],
                "routing_reason": f"Baseline Round-Robin: Turn #{total_examiner_turns + 1} cycling to {display_names[chosen_node]}."
            }


        # Strategy B: Keyword-Guided with Turn-Balancing Fallback (Default)
        # Extract last student message
        last_student_msg = ""
        lines = history.strip().split("\n")
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
        
        matched_prof = [kw for kw in prof_keywords if kw in text_to_scan]
        matched_ind = [kw for kw in industry_keywords if kw in text_to_scan]
        matched_critic = [kw for kw in critic_keywords if kw in text_to_scan]
        
        if matched_prof:
            return {
                "selected_agent_node": "professor",
                "selected_agent": "Professor",
                "routing_reason": f"Semantic keyword match: detected theoretical keywords [{', '.join(matched_prof[:3])}] in student response."
            }
        elif matched_ind:
            return {
                "selected_agent_node": "industry",
                "selected_agent": "Industry Expert",
                "routing_reason": f"Semantic keyword match: detected production/scaling keywords [{', '.join(matched_ind[:3])}] in student response."
            }
        elif matched_critic:
            return {
                "selected_agent_node": "critic",
                "selected_agent": "Critic",
                "routing_reason": f"Semantic keyword match: detected limitation/trade-off keywords [{', '.join(matched_critic[:3])}] in student response."
            }
        else:
            # Balanced rotation: choose the agent who has spoken the least
            least_node = min(counts, key=counts.get)
            display_names = {
                "examiner": "Examiner",
                "critic": "Critic",
                "industry": "Industry Expert",
                "professor": "Professor"
            }
            return {
                "selected_agent_node": least_node,
                "selected_agent": display_names[least_node],
                "routing_reason": f"Balanced rotation fallback: {display_names[least_node]} has fewest prior turns ({counts[least_node]} turns)."
            }

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define Router and Persona Nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("examiner", self.examiner_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("industry", self.industry_node)
        workflow.add_node("professor", self.professor_node)

        # Entry point is always the traceable router
        workflow.set_entry_point("router")

        # Conditional routing from router to persona
        workflow.add_conditional_edges(
            "router",
            lambda s: s["selected_agent_node"],
            {
                "examiner": "examiner",
                "critic": "critic",
                "industry": "industry",
                "professor": "professor"
            }
        )

        workflow.add_edge("examiner", END)
        workflow.add_edge("critic", END)
        workflow.add_edge("industry", END)
        workflow.add_edge("professor", END)

        return workflow.compile()

    def _generate_response(self, system_prompt: str, state: AgentState) -> dict:
        asked_str = ", ".join(state.get("asked_questions", []))
        
        prompt = f"""
        {system_prompt}
        
        Project Context:
        {state['context']}
        
        Chat History:
        {state['chat_history']}
        
        Difficulty Level: {state['difficulty']}
        """

        if state.get("is_final_turn"):
            prompt += """
            *** URGENT CRITICAL RULE ***
            - THIS IS THE VERY LAST TURN OF THE ENTIRE EXAM.
            - YOU ABSOLUTELY MUST NOT ASK ANOTHER QUESTION. NO MORE QUESTIONS.
            - YOUR ONLY JOB RIGHT NOW IS TO CONCLUDE THE SESSION.
            - Read the student's final answer, give brief feedback on it, and then write a comprehensive, friendly final summary of their performance throughout this chat history.
            - Say "This concludes your Viva test!" at the end.
            """
        else:
            prompt += f"""
            CRITICAL RULE:
            - Do NOT repeat questions or cover topics already asked.
            - Previously asked questions/topics: [{asked_str}]
            - Generate a NEW, distinct viva question related to the project context.
            
            Output only the question. Do not add any greeting, intro, or wrap-up text.
            """
            
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return {"generated_question": response.content.strip()}
        except Exception as e:
            print(f"Warning: MultiAgent LLM invocation error ({e.__class__.__name__}). Using persona fallback.")
            persona = state.get("selected_agent", "Examiner")
            ctx = state.get("context", "").strip()
            
            if ctx:
                first_sent = ctx.split('.')[0].strip()
                grounded_fallbacks = {
                    "Professor": f"Regarding your specification that '{first_sent}', what is the mathematical formulation and asymptotic computational complexity?",
                    "Industry Expert": f"In your implementation of '{first_sent}', how does this scale in production under 50,000 concurrent requests on AWS EKS?",
                    "Critic": f"Given your design choice that '{first_sent}', what critical engineering trade-offs and failure modes did you accept?",
                    "Examiner": f"Based on your report stating that '{first_sent}', walk me through how this component integrates into your architecture."
                }
                fallback_q = grounded_fallbacks.get(persona, f"Based on your report: '{first_sent}', can you explain your implementation?")
            else:
                ungrounded_fallbacks = {
                    "Professor": "Can you explain the theoretical computational complexity and mathematical formulation of your overall system algorithms?",
                    "Industry Expert": "In a high-throughput production environment, how does your architecture scale horizontally and handle node failures?",
                    "Critic": "What critical engineering trade-offs did you accept in your design, and what limitations or failure modes remain?",
                    "Examiner": "Can you elaborate on your general architectural choices, programming frameworks, and development challenges?"
                }
                fallback_q = ungrounded_fallbacks.get(persona, "Can you elaborate on your general architectural design choices?")
                
            return {"generated_question": fallback_q}

    def examiner_node(self, state: AgentState):
        prompt = "You are an Examiner. Ask normal, balanced viva questions assessing the student's overall understanding of the project flow, design, and implementation."
        return self._generate_response(prompt, state)

    def critic_node(self, state: AgentState):
        prompt = "You are a Research Critic. You heavily challenge assumptions. Ask why they chose their specific technique and challenge them to defend why they did not use an alternative modern technique. Probe into design flaws or limitations."
        return self._generate_response(prompt, state)

    def industry_node(self, state: AgentState):
        prompt = "You are an Industry Expert. Ask practical questions about scalability, deployment, serverless architectures, database indexing, API latencies, and handling high volume traffic (e.g. 1 Million daily users) in production."
        return self._generate_response(prompt, state)

    def professor_node(self, state: AgentState):
        prompt = "You are a University Professor. Focus heavily on underlying theory, mathematical formulas, algorithms, time complexity, and academic research foundations of their system."
        return self._generate_response(prompt, state)

    def run_panel(
        self, 
        context: str, 
        chat_history: str, 
        difficulty: str, 
        agent_type: str, 
        asked_questions: List[str] = None, 
        is_final_turn: bool = False,
        routing_strategy: str = "keyword_balanced"
    ) -> dict:
        """
        Executes the compiled multi-agent panel graph and returns both the generated question,
        the genuinely selected agent identity, and the routing rationale.
        """
        if asked_questions is None:
            asked_questions = []
            
        initial_state = {
            "context": context,
            "chat_history": chat_history,
            "difficulty": difficulty,
            "agent_type": agent_type,
            "routing_strategy": routing_strategy,
            "selected_agent_node": "examiner",
            "selected_agent": "Examiner",
            "routing_reason": "Initial default",
            "generated_question": "",
            "asked_questions": asked_questions,
            "is_final_turn": is_final_turn
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "question": final_state.get("generated_question", ""),
            "selected_agent": final_state.get("selected_agent", "Examiner"),
            "routing_reason": final_state.get("routing_reason", "Default routing"),
            "routing_strategy": routing_strategy
        }

