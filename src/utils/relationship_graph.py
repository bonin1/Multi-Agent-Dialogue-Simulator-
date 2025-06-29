"""
NetworkX-based relationship graph for modeling agent interactions
"""
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging


class RelationshipGraph:
    """Graph-based model for agent relationships and social dynamics"""
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for asymmetric relationships
        self.logger = logging.getLogger("RelationshipGraph")
        
        # Relationship type weights
        self.relationship_weights = {
            "trust": 1.0,
            "respect": 0.8,
            "affinity": 0.6,
            "cooperation": 0.7,
            "conflict": -0.5,
            "competition": -0.3,
        }
    
    def add_agent(self, agent_id: str, agent_data: Dict[str, Any]):
        """Add an agent to the relationship graph"""
        # Prepare node attributes, avoiding conflicts with NetworkX reserved parameters
        node_attrs = {
            "agent_name": agent_data.get("name", "Unknown"),
            "role": agent_data.get("role", "unknown"),
            "personality": agent_data.get("personality", {}),
            "created_at": datetime.now(),
        }
        
        # Add any additional data that doesn't conflict with reserved names
        for key, value in agent_data.items():
            if key not in ["name"]:  # Avoid conflicts
                node_attrs[key] = value
        
        self.graph.add_node(agent_id, **node_attrs)
        self.logger.info(f"Added agent {agent_id} to relationship graph")
    
    def update_agent(self, agent_id: str, updates: Dict[str, Any]):
        """Update agent data in the graph"""
        if agent_id in self.graph:
            self.graph.nodes[agent_id].update(updates)
            self.graph.nodes[agent_id]["updated_at"] = datetime.now()
    
    def add_relationship(
        self,
        from_agent: str,
        to_agent: str,
        relationship_type: str,
        strength: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add or update a relationship between agents"""
        if from_agent not in self.graph:
            self.logger.warning(f"Agent {from_agent} not in graph")
            return
        
        if to_agent not in self.graph:
            self.logger.warning(f"Agent {to_agent} not in graph")
            return
        
        # Create edge data
        edge_data = {
            "relationship_type": relationship_type,
            "strength": max(-1.0, min(1.0, strength)),  # Clamp to [-1, 1]
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "interaction_count": 1,
            "metadata": metadata or {}
        }
        
        # Update existing relationship or create new one
        if self.graph.has_edge(from_agent, to_agent):
            existing = self.graph[from_agent][to_agent]
            # Average the strengths and update metadata
            interaction_count = existing.get("interaction_count", 0) + 1
            new_strength = (existing["strength"] + strength) / 2
            
            edge_data.update({
                "strength": new_strength,
                "interaction_count": interaction_count,
                "created_at": existing["created_at"]
            })
        
        self.graph.add_edge(from_agent, to_agent, **edge_data)
        
        self.logger.debug(
            f"Updated relationship: {from_agent} -> {to_agent} "
            f"({relationship_type}, strength: {strength:.2f})"
        )
    
    def get_relationship(self, from_agent: str, to_agent: str) -> Optional[Dict[str, Any]]:
        """Get relationship data between two agents"""
        if self.graph.has_edge(from_agent, to_agent):
            return dict(self.graph[from_agent][to_agent])
        return None
    
    def get_agent_relationships(self, agent_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all relationships for an agent"""
        relationships = {
            "outgoing": [],  # Relationships this agent has with others
            "incoming": []   # Relationships others have with this agent
        }
        
        # Outgoing relationships
        for target in self.graph.successors(agent_id):
            rel_data = dict(self.graph[agent_id][target])
            rel_data["target_agent"] = target
            rel_data["target_name"] = self.graph.nodes[target].get("agent_name", target)
            relationships["outgoing"].append(rel_data)
        
        # Incoming relationships
        for source in self.graph.predecessors(agent_id):
            rel_data = dict(self.graph[source][agent_id])
            rel_data["source_agent"] = source
            rel_data["source_name"] = self.graph.nodes[source].get("agent_name", source)
            relationships["incoming"].append(rel_data)
        
        return relationships
    
    def calculate_influence_score(self, agent_id: str) -> float:
        """Calculate how influential an agent is in the network"""
        if agent_id not in self.graph:
            return 0.0
        
        # Combine different centrality measures
        try:
            # In-degree centrality (how many others trust/respect this agent)
            in_degree = self.graph.in_degree(agent_id, weight="strength")
            
            # PageRank (considers both direct and indirect influence)
            pagerank = nx.pagerank(self.graph, weight="strength").get(agent_id, 0)
            
            # Betweenness centrality (how often agent is between others)
            betweenness = nx.betweenness_centrality(self.graph, weight="strength").get(agent_id, 0)
            
            # Weighted combination
            influence = (
                in_degree * 0.4 +
                pagerank * len(self.graph) * 0.4 +  # Normalize PageRank
                betweenness * 0.2
            )
            
            return max(0.0, min(1.0, influence / len(self.graph)))
            
        except Exception as e:
            self.logger.warning(f"Error calculating influence for {agent_id}: {e}")
            return 0.0
    
    def find_coalitions(self, min_size: int = 2) -> List[List[str]]:
        """Find potential coalitions (strongly connected groups)"""
        try:
            # Find strongly connected components
            components = list(nx.strongly_connected_components(self.graph))
            
            # Filter by size and positive relationships
            coalitions = []
            for component in components:
                if len(component) >= min_size:
                    # Check if relationships within component are mostly positive
                    positive_edges = 0
                    total_edges = 0
                    
                    for agent1 in component:
                        for agent2 in component:
                            if agent1 != agent2 and self.graph.has_edge(agent1, agent2):
                                total_edges += 1
                                if self.graph[agent1][agent2]["strength"] > 0:
                                    positive_edges += 1
                    
                    if total_edges > 0 and positive_edges / total_edges > 0.6:
                        coalitions.append(list(component))
            
            return coalitions
            
        except Exception as e:
            self.logger.warning(f"Error finding coalitions: {e}")
            return []
    
    def detect_conflicts(self, threshold: float = -0.3) -> List[Tuple[str, str, float]]:
        """Detect conflicting relationships"""
        conflicts = []
        
        for source, target, data in self.graph.edges(data=True):
            if data["strength"] < threshold:
                conflicts.append((source, target, data["strength"]))
        
        # Sort by conflict intensity (most negative first)
        conflicts.sort(key=lambda x: x[2])
        return conflicts
    
    def suggest_mediators(self, agent1: str, agent2: str) -> List[str]:
        """Suggest potential mediators for conflicting agents"""
        if agent1 not in self.graph or agent2 not in self.graph:
            return []
        
        mediators = []
        
        # Find agents with positive relationships to both
        for agent in self.graph.nodes():
            if agent in [agent1, agent2]:
                continue
            
            rel1 = self.get_relationship(agent, agent1)
            rel2 = self.get_relationship(agent, agent2)
            
            # Check if agent has positive relationships with both
            if (rel1 and rel1["strength"] > 0.3 and 
                rel2 and rel2["strength"] > 0.3):
                mediators.append(agent)
        
        # Sort by average relationship strength
        def mediator_score(agent):
            rel1 = self.get_relationship(agent, agent1)
            rel2 = self.get_relationship(agent, agent2)
            return (rel1["strength"] + rel2["strength"]) / 2
        
        mediators.sort(key=mediator_score, reverse=True)
        return mediators[:3]  # Top 3 potential mediators
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get overall network metrics"""
        if not self.graph.nodes():
            return {}
        
        try:
            metrics = {
                "node_count": len(self.graph.nodes()),
                "edge_count": len(self.graph.edges()),
                "density": nx.density(self.graph),
                "average_clustering": nx.average_clustering(self.graph),
                "is_connected": nx.is_weakly_connected(self.graph),
            }
            
            # Relationship type distribution
            relationship_types = {}
            for _, _, data in self.graph.edges(data=True):
                rel_type = data.get("relationship_type", "unknown")
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            metrics["relationship_types"] = relationship_types
            
            # Average relationship strength
            strengths = [data["strength"] for _, _, data in self.graph.edges(data=True)]
            if strengths:
                metrics["average_strength"] = sum(strengths) / len(strengths)
                metrics["positive_relationships"] = sum(1 for s in strengths if s > 0)
                metrics["negative_relationships"] = sum(1 for s in strengths if s < 0)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating network metrics: {e}")
            return {"error": str(e)}
    
    def get_shortest_path(self, from_agent: str, to_agent: str) -> Optional[List[str]]:
        """Find shortest path between two agents"""
        try:
            return nx.shortest_path(self.graph, from_agent, to_agent)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for visualization or storage"""
        return {
            "nodes": [
                {
                    "id": node_id,
                    **data
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    **data
                }
                for source, target, data in self.graph.edges(data=True)
            ],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "metrics": self.get_network_metrics()
            }
        }
    
    def import_graph_data(self, data: Dict[str, Any]):
        """Import graph data from exported format"""
        self.graph.clear()
        
        # Add nodes
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        
        # Add edges
        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            self.graph.add_edge(source, target, **edge)
        
        self.logger.info(f"Imported graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
    
    def visualize_graph(self, output_file: Optional[str] = None) -> Optional[str]:
        """Create a visualization of the relationship graph"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            if not self.graph.nodes():
                return None
            
            # Create layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            plt.figure(figsize=(12, 8))
            
            # Draw nodes
            node_colors = []
            node_sizes = []
            
            for node in self.graph.nodes():
                # Color by role
                role = self.graph.nodes[node].get("role", "unknown")
                role_colors = {
                    "doctor": "lightblue",
                    "engineer": "lightgreen",
                    "spy": "lightcoral",
                    "rebel": "orange",
                    "diplomat": "lightpink",
                    "unknown": "lightgray"
                }
                node_colors.append(role_colors.get(role, "lightgray"))
                
                # Size by influence
                influence = self.calculate_influence_score(node)
                node_sizes.append(300 + influence * 500)
            
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8
            )
            
            # Draw edges
            positive_edges = []
            negative_edges = []
            
            for source, target, data in self.graph.edges(data=True):
                if data["strength"] > 0:
                    positive_edges.append((source, target))
                else:
                    negative_edges.append((source, target))
            
            # Draw positive relationships in green
            if positive_edges:
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=positive_edges,
                    edge_color="green",
                    alpha=0.6,
                    width=2
                )
            
            # Draw negative relationships in red
            if negative_edges:
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=negative_edges,
                    edge_color="red",
                    alpha=0.6,
                    width=2,
                    style="dashed"
                )
            
            # Draw labels
            labels = {node: self.graph.nodes[node].get("agent_name", node) for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            # Create legend
            role_patches = [
                mpatches.Patch(color="lightblue", label="Doctor"),
                mpatches.Patch(color="lightgreen", label="Engineer"),
                mpatches.Patch(color="lightcoral", label="Spy"),
                mpatches.Patch(color="orange", label="Rebel"),
                mpatches.Patch(color="lightpink", label="Diplomat"),
            ]
            
            relationship_patches = [
                mpatches.Patch(color="green", label="Positive Relationship"),
                mpatches.Patch(color="red", label="Negative Relationship"),
            ]
            
            plt.legend(handles=role_patches + relationship_patches, loc="upper left", bbox_to_anchor=(1, 1))
            
            plt.title("Agent Relationship Network")
            plt.axis("off")
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()
                return output_file
            else:
                plt.show()
                return None
                
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
            return None
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return None
    
    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for an agent"""
        if agent_id not in self.graph:
            return {}
        
        relationships = self.get_agent_relationships(agent_id)
        influence = self.calculate_influence_score(agent_id)
        
        # Count relationship types
        positive_out = sum(1 for rel in relationships["outgoing"] if rel["strength"] > 0)
        negative_out = sum(1 for rel in relationships["outgoing"] if rel["strength"] < 0)
        positive_in = sum(1 for rel in relationships["incoming"] if rel["strength"] > 0)
        negative_in = sum(1 for rel in relationships["incoming"] if rel["strength"] < 0)
        
        return {
            "agent_data": dict(self.graph.nodes[agent_id]),
            "influence_score": influence,
            "relationship_summary": {
                "outgoing_positive": positive_out,
                "outgoing_negative": negative_out,
                "incoming_positive": positive_in,
                "incoming_negative": negative_in,
                "total_connections": len(relationships["outgoing"]) + len(relationships["incoming"])
            },
            "relationships": relationships
        }
