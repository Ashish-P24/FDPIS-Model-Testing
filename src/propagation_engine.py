import os
import json
import networkx as nx
import pandas as pd
import numpy as np

class FDPISPropagationEngine:
    """
    FDPIS Network-Aware Delay Propagation Engine (Layer 4).
    Models aircraft continuity, absorbable slack, and downstream cascade effects using Breadth-First Search (BFS).
    """
    def __init__(self, turnaround_buffer_threshold=45):
        """
        turnaround_buffer_threshold: Standard minimum ground turnaround required (minutes)
        Default = 45 minutes for narrow-body commercial operations.
        """
        self.min_turnaround = turnaround_buffer_threshold
        self.graph = nx.DiGraph()
        
    def build_network_graph(self, flights_df):
        """
        Constructs a directed weighted graph G = (V, E) for a given operating day.
        Nodes (V): Flight records
        Edges (E): Aircraft rotation continuity edges (weight = 0.95)
        """
        self.graph.clear()
        
        # Ensure proper time order per aircraft
        df = flights_df.copy()
        df["sched_dep_time_min"] = (df["CRS_DEP_TIME"] // 100) * 60 + (df["CRS_DEP_TIME"] % 100)
        df["sched_arr_time_min"] = (df["CRS_ARR_TIME"] // 100) * 60 + (df["CRS_ARR_TIME"] % 100)
        
        # Sort by TAIL_NUM and departure time
        df = df.sort_values(["TAIL_NUM", "sched_dep_time_min"]).reset_index(drop=True)
        
        # Add nodes
        for _, row in df.iterrows():
            flight_id = str(row.get("flight_id", f"{row['OP_UNIQUE_CARRIER']}_{row['ORIGIN']}_{row['DEST']}_{row['CRS_DEP_TIME']}"))
            self.graph.add_node(
                flight_id,
                carrier=row["OP_UNIQUE_CARRIER"],
                origin=row["ORIGIN"],
                dest=row["DEST"],
                tail_num=row["TAIL_NUM"],
                crs_dep_time=int(row["CRS_DEP_TIME"]),
                crs_arr_time=int(row["CRS_ARR_TIME"]),
                dep_min=int(row["sched_dep_time_min"]),
                arr_min=int(row["sched_arr_time_min"]),
                distance=float(row["DISTANCE"]),
                primary_delay_prob=float(row.get("delay_prob", 0.0)),
                predicted_delay_min=float(row.get("predicted_delay", 0.0))
            )
            
        # Add edges along aircraft rotation chains
        for tail, group in df.groupby("TAIL_NUM"):
            if tail == "UNKNOWN" or pd.isna(tail):
                continue
            legs = group.to_dict(orient="records")
            for i in range(len(legs) - 1):
                curr_flight = legs[i]
                next_flight = legs[i+1]
                
                # Check spatial continuity (curr dest == next origin)
                is_connected = (curr_flight["DEST"] == next_flight["ORIGIN"])
                
                # Calculate scheduled ground gap
                ground_gap = next_flight["sched_dep_time_min"] - curr_flight["sched_arr_time_min"]
                if ground_gap < -720:  # Next day rollover handling
                    ground_gap += 1440
                    
                # Absorbable slack = scheduled gap - minimum required turnaround
                absorbable_slack = max(0, ground_gap - self.min_turnaround)
                
                curr_id = str(curr_flight.get("flight_id", f"{curr_flight['OP_UNIQUE_CARRIER']}_{curr_flight['ORIGIN']}_{curr_flight['DEST']}_{curr_flight['CRS_DEP_TIME']}"))
                next_id = str(next_flight.get("flight_id", f"{next_flight['OP_UNIQUE_CARRIER']}_{next_flight['ORIGIN']}_{next_flight['DEST']}_{next_flight['CRS_DEP_TIME']}"))
                
                self.graph.add_edge(
                    curr_id,
                    next_id,
                    edge_type="AIRCRAFT_ROTATION",
                    weight=0.95,
                    ground_gap=ground_gap,
                    absorbable_slack=absorbable_slack,
                    is_spatially_continuous=is_connected
                )
                
        return len(self.graph.nodes), len(self.graph.edges)
        
    def propagate_cascade(self, root_flight_id, initial_delay_minutes=None, max_depth=4):
        """
        Executes Breadth-First Search (BFS) starting from root_flight_id.
        Computes residual delay transmission across downstream flights.
        """
        if root_flight_id not in self.graph:
            return None
            
        root_data = self.graph.nodes[root_flight_id]
        if initial_delay_minutes is None:
            initial_delay_minutes = root_data.get("predicted_delay_min", 30.0)
            
        # BFS Queue: (current_node_id, incoming_delay, depth, parent_id)
        from collections import deque
        queue = deque([(root_flight_id, float(initial_delay_minutes), 0, None)])
        
        visited = set([root_flight_id])
        cascade_tree = {
            "root_flight": root_flight_id,
            "root_carrier": root_data["carrier"],
            "root_route": f"{root_data['origin']} -> {root_data['dest']}",
            "root_delay": float(initial_delay_minutes),
            "affected_flights_count": 0,
            "max_cascade_depth": 0,
            "nodes": [],
            "edges": [],
            "recommendations": []
        }
        
        # Add root node to tree
        cascade_tree["nodes"].append({
            "id": root_flight_id,
            "label": f"{root_data['carrier']} {root_data['origin']}->{root_data['dest']}",
            "origin": root_data["origin"],
            "dest": root_data["dest"],
            "tail_num": root_data["tail_num"],
            "depth": 0,
            "incoming_delay": float(initial_delay_minutes),
            "residual_delay": float(initial_delay_minutes),
            "status": "ROOT_ORIGIN"
        })
        
        total_affected = 0
        max_d = 0
        
        while queue:
            curr_id, incoming_delay, depth, parent_id = queue.popleft()
            if depth > max_d:
                max_d = depth
                
            if depth >= max_depth:
                continue
                
            # Explore outbound edges
            for nxt_id in self.graph.successors(curr_id):
                edge_data = self.graph.edges[curr_id, nxt_id]
                nxt_node = self.graph.nodes[nxt_id]
                
                # Propagation physics:
                # Transmitted delay = (incoming_delay * edge_weight) - absorbable_slack
                transmitted_delay = incoming_delay * edge_data["weight"]
                residual_delay = max(0.0, transmitted_delay - edge_data["absorbable_slack"])
                
                edge_info = {
                    "source": curr_id,
                    "target": nxt_id,
                    "type": edge_data["edge_type"],
                    "slack_minutes": edge_data["absorbable_slack"],
                    "ground_gap": edge_data["ground_gap"],
                    "transmitted_delay": round(transmitted_delay, 1),
                    "residual_delay": round(residual_delay, 1),
                    "is_absorbed": bool(residual_delay <= 0)
                }
                cascade_tree["edges"].append(edge_info)
                
                # If residual delay > 0, cascade hits downstream flight!
                if residual_delay > 0:
                    total_affected += 1
                    status = "HIGH_CASCADE_RISK" if residual_delay >= 30 else "MODERATE_CASCADE_RISK"
                    
                    cascade_tree["nodes"].append({
                        "id": nxt_id,
                        "label": f"{nxt_node['carrier']} {nxt_node['origin']}->{nxt_node['dest']}",
                        "origin": nxt_node["origin"],
                        "dest": nxt_node["dest"],
                        "tail_num": nxt_node["tail_num"],
                        "depth": depth + 1,
                        "incoming_delay": round(transmitted_delay, 1),
                        "residual_delay": round(residual_delay, 1),
                        "slack_absorbed": round(min(transmitted_delay, edge_data["absorbable_slack"]), 1),
                        "status": status
                    })
                    
                    # Rule-based Recommendation Generation (Layer 5)
                    if residual_delay >= 90:
                        cascade_tree["recommendations"].append({
                            "flight_id": nxt_id,
                            "rule": "Crew Duty Hour Protection (FDTL)",
                            "severity": "CRITICAL",
                            "action": f"Alert standby crew at {nxt_node['origin']}; begin immediate positioning. Residual delay {residual_delay:.0f}m risks duty exceedance."
                        })
                    elif residual_delay >= 60:
                        cascade_tree["recommendations"].append({
                            "flight_id": nxt_id,
                            "rule": "Ground Turnaround Compression",
                            "severity": "HIGH",
                            "action": f"Pre-allocate rapid turn ground crew and priority baggage unloading at {nxt_node['origin']}."
                        })
                    elif residual_delay >= 30:
                        cascade_tree["recommendations"].append({
                            "flight_id": nxt_id,
                            "rule": "Passenger Connection Protection",
                            "severity": "MEDIUM",
                            "action": f"Review connecting passengers for {nxt_node['dest']}; prepare gate escort or alternate routing options."
                        })
                        
                    if nxt_id not in visited:
                        visited.add(nxt_id)
                        queue.append((nxt_id, residual_delay, depth + 1, curr_id))
                else:
                    # Delay fully absorbed by buffer
                    cascade_tree["nodes"].append({
                        "id": nxt_id,
                        "label": f"{nxt_node['carrier']} {nxt_node['origin']}->{nxt_node['dest']}",
                        "origin": nxt_node["origin"],
                        "dest": nxt_node["dest"],
                        "tail_num": nxt_node["tail_num"],
                        "depth": depth + 1,
                        "incoming_delay": round(transmitted_delay, 1),
                        "residual_delay": 0.0,
                        "slack_absorbed": round(transmitted_delay, 1),
                        "status": "DELAY_ABSORBED"
                    })
                    
        cascade_tree["affected_flights_count"] = total_affected
        cascade_tree["max_cascade_depth"] = max_d
        return cascade_tree
