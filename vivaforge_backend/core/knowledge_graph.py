import networkx as nx
from pyvis.network import Network
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List
import os
import json
import time

class Node(BaseModel):
    id: str = Field(description="Name or identifier of the entity, e.g., 'Random Forest', 'Machine Learning'")
    type: str = Field(description="Type of entity, e.g., 'Algorithm', 'Concept', 'Tool'")

class Relationship(BaseModel):
    source: str = Field(description="Source entity id")
    target: str = Field(description="Target entity id")
    type: str = Field(description="Relationship type, e.g., 'USES', 'PART_OF', 'IS_A'")

class GraphExtraction(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

class KGBuilder:
    def __init__(self, api_key: str = None):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.graph = nx.DiGraph()
        self.generation_time = 0.0

    def extract_graph_from_text(self, text: str):
        start_time = time.time()
        structured_llm = self.llm.with_structured_output(GraphExtraction)
        
        prompt = f"""
        Analyze the following text from a student's technical project report.
        Extract the key technical concepts, tools, methodologies, algorithms, and their relationships.
        Focus heavily on identifying architectural components, AI/ML concepts, and systems.
        Do not create too many nodes, focus on the top 20-30 most important technical components.
        
        Text:
        {text[:10000]}  # Extract from the first part of the document usually containing the abstract/intro/architecture
        """
        
        try:
            result = structured_llm.invoke(prompt)
            self.generation_time = round(time.time() - start_time, 2)
            
            # Save kg_data.json
            if result:
                kg_data = {
                    "nodes": [node.model_dump() for node in result.nodes],
                    "relationships": [rel.model_dump() for rel in result.relationships]
                }
                with open("kg_data.json", "w", encoding="utf-8") as f:
                    json.dump(kg_data, f, ensure_ascii=False, indent=2)
                    
            return result
        except Exception as e:
            print(f"Error extracting graph: {e}")
            self.generation_time = round(time.time() - start_time, 2)
            return None

    def build_networkx_graph(self, extraction: GraphExtraction):
        self.graph.clear()
        
        # Track unique concept types
        concept_types = set()
        
        for node in extraction.nodes:
            self.graph.add_node(node.id, label=node.id, title=node.type, group=node.type)
            concept_types.add(node.type)
        
        for rel in extraction.relationships:
            # Only add edges if both nodes exist in the node list to prevent isolated unlabeled nodes
            if rel.source in self.graph.nodes and rel.target in self.graph.nodes:
                self.graph.add_edge(rel.source, rel.target, label=rel.type, title=rel.type)

        # Compute graph metrics
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        num_concepts = len(concept_types)
        graph_density = round(nx.density(self.graph), 4) if num_nodes > 1 else 0.0
        
        metadata = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_concepts": num_concepts,
            "graph_density": graph_density,
            "generation_time_sec": self.generation_time
        }
        
        # Save graph_metadata.json
        with open("graph_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def generate_html_visualization(self, output_file="kg_visualization.html"):
        net = Network(height="600px", width="100%", bgcolor="#0b0f19", font_color="white", directed=True)
        
        # Configure physics for stable visualization
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
