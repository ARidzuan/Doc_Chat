from typing import List, Set


class QueryRouter:
    """Routes queries to appropriate document collections based on keywords and intent"""
    
    def __init__(self):
        # Keywords 
        self.routing_map = {
            "studio_apis": [
                "api", "function", "method", "class", "endpoint", "reference",
                "sdk", "interface", "parameter", "return", "library"
            ],
            "simulation": [
                "simulation", "scenario", "run", "execute", "simulate",
                "configuration", "setup simulation", "test", "replay"
            ],
            "vehicle": [
                "vehicle", "car", "dynamics", "wheel", "tire", "tyre", "suspension",
                "brake", "steering", "drivetrain", "engine", "chassis", "callas", "sensors"
            ],
            "models": [
                "model", "3d", "asset", "mesh", "object", "import",
                "export", "geometry", "shape"
            ],
            "terrain": [
                "terrain", "road", "environment", "landscape", "surface",
                "ground", "elevation", "path", "route", "network"
            ],
            "studio": [
                "studio", "interface", "ui", "gui", "workspace", "editor",
                "window", "menu", "toolbar", "panel","configuration"
            ],
            "analysis": [
                "analysis", "analyze", "data", "metrics", "report", "results",
                "statistics", "graph", "chart", "export data"
            ],
            "compute": [
                "compute", "calculation", "algorithm", "process", "module",
                "plugin", "extension", "cloud", "parallel", "performance"
            ],
            "unreal": [
                "unreal", "ue4", "ue5", "blueprint", "rendering",
                "visualization", "graphics"
            ],
        }
    
    def route(self, query: str, top_n: int = 3) -> List[str]:
        """
        Route query to most relevant collections.
        Returns list of collection names to search, limited to top_n.
        """
        query_lower = query.lower()
        scores = {}
        
        # Score each collection based on keyword matches
        for collection, keywords in self.routing_map.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[collection] = score
        
        # Sort by score and return top N
        if scores:
            sorted_colls = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [coll for coll, _ in sorted_colls[:top_n]]
        
        # Default: search in most common collections
        return ["general", "studio_apis", "studio"]

    def route_with_fallback(self, query: str, available_collections: List[str]) -> List[str]:
        """
        Route query and ensure returned collections exist in available_collections.
        Falls back to all collections if no routing matches.

        Args:
            query: The user's search query
            available_collections: List of actually available collection names

        Returns:
            List of collection names to search
        """
        routed = self.route(query, top_n=3)

        # Filter to only existing collections
        valid = [c for c in routed if c in available_collections]

        # If no valid collections after filtering, use all available
        if not valid:
            return available_collections

        return valid