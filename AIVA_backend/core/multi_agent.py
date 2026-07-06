from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

class AgentState(TypedDict):
    context: str
    chat_history: str
    difficulty: str
    agent_type: str
    generated_question: str
    asked_questions: List[str]  # Prevents repeated questions

class MultiAgentPanel:
    def __init__(self, api_key: str = None):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("examiner", self.examiner_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("industry", self.industry_node)
        workflow.add_node("professor", self.professor_node)

        # Define rule-based router function
        def route_to_agent(state: AgentState):
            agent = state["agent_type"].lower()
            
            # If Auto (Panel) mode is selected, route based on response keywords or balanced turns
            if agent in ["auto", "panel", "auto (panel)"]:
                history = state.get("chat_history", "")
                
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
                
                # Rule-based keyword matching
                prof_keywords = ["math", "formula", "algorithm", "complexity", "theorem", "proof", "equation", "analysis", "theory", "time complexity", "big-o", "space complexity", "derivation"]
                industry_keywords = ["deploy", "docker", "aws", "cloud", "scale", "latency", "production", "kubernetes", "database", "ci/cd", "pipeline", "dockerfile", "server", "hosting", "architecture", "microservice", "infrastructure"]
                critic_keywords = ["limit", "fail", "drawback", "bottleneck", "weakness", "error", "alternative", "compromise", "flaw", "shortcoming", "disadvantage", "constraint"]
                
                if any(kw in text_to_scan for kw in prof_keywords):
                    return "professor"
                elif any(kw in text_to_scan for kw in industry_keywords):
                    return "industry"
                elif any(kw in text_to_scan for kw in critic_keywords):
                    return "critic"
                else:
                    # Balanced rotation: choose the agent who has spoken the least
                    history_lower = history.lower()
                    counts = {
                        "professor": history_lower.count("professor:"),
                        "industry": history_lower.count("industry expert:"),
                        "critic": history_lower.count("research critic:"),
                        "examiner": history_lower.count("examiner:")
                    }
                    # Return the agent with minimum turns
                    return min(counts, key=counts.get)
            
            # Direct routing if a specific agent is selected
            if agent == "critic":
                return "critic"
            elif agent in ["industry", "industry expert"]:
                return "industry"
            elif agent == "professor":
                return "professor"
            else:
                return "examiner"

        workflow.set_conditional_entry_point(
            route_to_agent,
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
        
        CRITICAL RULE:
        - Do NOT repeat questions or cover topics already asked.
        - Previously asked questions/topics: [{asked_str}]
        - Generate a NEW, distinct viva question related to the project context.
        
        Output only the question. Do not add any greeting, intro, or wrap-up text.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"generated_question": response.content.strip()}

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

    def run_panel(self, context: str, chat_history: str, difficulty: str, agent_type: str, asked_questions: List[str] = None) -> dict:
        if asked_questions is None:
            asked_questions = []
            
        initial_state = {
            "context": context,
            "chat_history": chat_history,
            "difficulty": difficulty,
            "agent_type": agent_type,
            "generated_question": "",
            "asked_questions": asked_questions
        }
        
        # We want to trace which agent actually spoke!
        # LangGraph compile allows us to inspect the final state which contains the agent selection
        # But we can also determine which node ran by looking at the execution trace or just letting the node populate its type.
        # Since we route using set_conditional_entry_point, we can find out the node that ran by checking the selected node.
        # To make it simple, let's update the node functions to return the agent type as well!
        # Wait, if we return it in the node dict, it updates the state. Let's add "selected_agent" to AgentState.
        
        final_state = self.graph.invoke(initial_state)
        
        # We can find out which agent ran by checking the entry point routing or we can just let each node return its identity.
        # Let's inspect final_state
        return {
            "question": final_state["generated_question"]
        }
