#!/usr/bin/env python3
"""
Build Knowledge Graph from ACN Data
Extracts entities (Events, Dates, Locations) and relationships from ingested chunks.
"""
import json
import os
import networkx as nx
import re
from pathlib import Path
from dateutil.parser import parse as parse_date
import pickle

# Reuse Config to get paths - REMOVED to avoid torch import
# from ingest_acn_data import Config

def build_graph():
    print("Building Knowledge Graph...")
    
    # Initialize Graph
    G = nx.MultiDiGraph()
    
    # Load chunks from the JSON files created during ingestion
    # (Since we can't easily iterate ChromaDB, we iterate the source chunks if available, 
    #  or we can reload from the ChromaDB if we had a method. 
    #  Let's assume we can scan the 'acn_data/chunks' directory which we saw earlier)
    
    chunks_dir = Path("acn_data/raw")
    if not chunks_dir.exists():
        print(f"Error: Chunks directory {chunks_dir} not found. Run ingestion first.")
        return

    print(f"Scanning chunks in {chunks_dir}...")
    
    count = 0
    for chunk_file in chunks_dir.glob("*.json"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        content = data.get("content", "")
        url = data.get("url", "")
        title = data.get("title", "")
        content_type = data.get("content_type", "general")
        
        # --- Entity Extraction Logic (Heuristic/Regex based for speed) ---
        
        # 1. Extract EVENTS
        # If content_type is event or title creates an event context
        is_event_page = content_type == "event"
        
        if is_event_page:
            event_name = title.strip()
            # Clean event name
            if len(event_name) > 100: event_name = event_name[:100] + "..."
            
            # Add Event Node
            G.add_node(event_name, type="Event", url=url)
            
            # Extract Dates
            # Pattern: Month DD, YYYY or Month DD-DD, YYYY
            date_matches = re.findall(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:[-–]\d{1,2})?,\s+(20\d{2})', content)
            
            for match in date_matches:
                month, year = match[0], match[1]
                year_node = year
                
                # Link Event -> Year
                G.add_node(year_node, type="Year")
                G.add_edge(event_name, year_node, relation="HAPPENS_IN")
                
                # Can also add specific date node if needed
                
            # Extract Locations
            # Heuristic: "City, State" lookup or common big cities
            # Simple list of ACN hubs
            common_locs = ["Dallas", "Chicago", "Las Vegas", "Orlando", "Virtual", "Online", "Webinar", "Calgary"]
            for loc in common_locs:
                if loc in content:
                    G.add_node(loc, type="Location")
                    G.add_edge(event_name, loc, relation="LOCATED_AT")
                    
            # Extract Topics (Keywords)
            keywords = ["Epic", "EZLynx", "Indio", "Applied Pay", "Security", "AI", "Roundtable"]
            for kw in keywords:
                if kw in content:
                    G.add_node(kw, type="Topic")
                    G.add_edge(event_name, kw, relation="COVERS_TOPIC")

            count += 1

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Save Graph
    output_dir = Path("acn_data/graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as ID-mapped JSON for easy loading
    graph_data = nx.node_link_data(G)
    with open(output_dir / "acn_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2)
        
    print(f"Graph saved to {output_dir / 'acn_graph.json'}")

if __name__ == "__main__":
    build_graph()
