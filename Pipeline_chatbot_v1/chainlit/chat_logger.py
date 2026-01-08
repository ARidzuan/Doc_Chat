"""
Simple Chat History Logger
Saves all conversations to JSON files for later analysis
No database required - just flat JSON files
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import uuid


class ChatLogger:
    """Simple logger that saves chat history to JSON files"""

    def __init__(self, log_dir: str = "./data/chat_logs"):
        """
        Initialize chat logger

        Args:
            log_dir: Directory to save chat logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create session ID for this chat session
        self.session_id = str(uuid.uuid4())
        self.session_file = os.path.join(
            log_dir,
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.session_id[:8]}.json"
        )

        # Initialize session data
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "messages": [],
            "metadata": {}
        }

        print(f"[ChatLogger] Logging to: {self.session_file}")

    def log_user_message(self, message: str, metadata: Dict = None):
        """
        Log a user message

        Args:
            message: User's message text
            metadata: Additional metadata (user_id, timestamp, etc.)
        """
        entry = {
            "type": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.session_data["messages"].append(entry)
        self._save()

    def log_assistant_message(
        self,
        message: str,
        sources: List[str] = None,
        metadata: Dict = None
    ):
        """
        Log an assistant message

        Args:
            message: Assistant's response text
            sources: List of source documents used
            metadata: Additional metadata (model, tokens, latency, etc.)
        """
        entry = {
            "type": "assistant",
            "content": message,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.session_data["messages"].append(entry)
        self._save()

    def log_error(self, error: str, context: Dict = None):
        """
        Log an error

        Args:
            error: Error message
            context: Additional context about the error
        """
        entry = {
            "type": "error",
            "content": error,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        self.session_data["messages"].append(entry)
        self._save()

    def set_metadata(self, key: str, value: Any):
        """
        Set session metadata

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.session_data["metadata"][key] = value
        self._save()

    def end_session(self):
        """Mark session as ended"""
        self.session_data["end_time"] = datetime.now().isoformat()
        self.session_data["metadata"]["total_messages"] = len(self.session_data["messages"])
        self.session_data["metadata"]["user_messages"] = sum(
            1 for m in self.session_data["messages"] if m["type"] == "user"
        )
        self.session_data["metadata"]["assistant_messages"] = sum(
            1 for m in self.session_data["messages"] if m["type"] == "assistant"
        )
        self._save()
        print(f"[ChatLogger] Session ended. Total messages: {len(self.session_data['messages'])}")

    def _save(self):
        """Save session data to file"""
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ChatLogger] Error saving log: {e}")

    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        return {
            "session_id": self.session_id,
            "file": self.session_file,
            "start_time": self.session_data["start_time"],
            "message_count": len(self.session_data["messages"]),
            "user_messages": sum(1 for m in self.session_data["messages"] if m["type"] == "user"),
            "assistant_messages": sum(1 for m in self.session_data["messages"] if m["type"] == "assistant"),
        }


class ChatLogAnalyzer:
    """Analyze chat logs for performance metrics"""

    def __init__(self, log_dir: str = "./data/chat_logs"):
        self.log_dir = log_dir

    def list_sessions(self) -> List[str]:
        """List all chat session files"""
        if not os.path.exists(self.log_dir):
            return []

        return [
            f for f in os.listdir(self.log_dir)
            if f.startswith("chat_") and f.endswith(".json")
        ]

    def load_session(self, filename: str) -> Dict:
        """Load a chat session from file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_session(self, session_data: Dict) -> Dict:
        """Analyze a single session for metrics"""
        messages = session_data["messages"]

        user_msgs = [m for m in messages if m["type"] == "user"]
        assistant_msgs = [m for m in messages if m["type"] == "assistant"]

        # Calculate response times if timestamps available
        response_times = []
        for i, user_msg in enumerate(user_msgs):
            # Find next assistant message
            user_time = datetime.fromisoformat(user_msg["timestamp"])
            for assist_msg in assistant_msgs:
                assist_time = datetime.fromisoformat(assist_msg["timestamp"])
                if assist_time > user_time:
                    response_time = (assist_time - user_time).total_seconds()
                    response_times.append(response_time)
                    break

        # Calculate session duration
        start_time = datetime.fromisoformat(session_data["start_time"])
        end_time = session_data.get("end_time")
        duration = None
        if end_time:
            end_time = datetime.fromisoformat(end_time)
            duration = (end_time - start_time).total_seconds()

        return {
            "session_id": session_data["session_id"],
            "start_time": session_data["start_time"],
            "duration_seconds": duration,
            "total_messages": len(messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else None,
            "min_response_time": min(response_times) if response_times else None,
            "max_response_time": max(response_times) if response_times else None,
            "errors": len([m for m in messages if m["type"] == "error"]),
            "metadata": session_data.get("metadata", {})
        }

    def analyze_all_sessions(self) -> List[Dict]:
        """Analyze all chat sessions"""
        sessions = self.list_sessions()
        analyses = []

        for session_file in sessions:
            try:
                session_data = self.load_session(session_file)
                analysis = self.analyze_session(session_data)
                analysis["file"] = session_file
                analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing {session_file}: {e}")

        return analyses

    def generate_report(self, output_file: str = None) -> Dict:
        """Generate comprehensive analysis report"""
        analyses = self.analyze_all_sessions()

        if not analyses:
            return {"error": "No sessions found"}

        # Aggregate statistics
        total_sessions = len(analyses)
        total_messages = sum(a["total_messages"] for a in analyses)
        total_user_messages = sum(a["user_messages"] for a in analyses)
        total_assistant_messages = sum(a["assistant_messages"] for a in analyses)

        response_times = [a["avg_response_time"] for a in analyses if a["avg_response_time"]]

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_user_messages": total_user_messages,
                "total_assistant_messages": total_assistant_messages,
                "avg_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0,
                "avg_response_time_seconds": sum(response_times) / len(response_times) if response_times else None,
            },
            "sessions": analyses
        }

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Report saved to: {output_file}")

        return report


if __name__ == "__main__":
    # Example usage and analysis
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # Analyze existing logs
        analyzer = ChatLogAnalyzer()
        report = analyzer.generate_report("./data/chat_analysis_report.json")

        print("\n" + "="*60)
        print("CHAT LOG ANALYSIS REPORT")
        print("="*60)
        print(f"Total Sessions: {report['summary']['total_sessions']}")
        print(f"Total Messages: {report['summary']['total_messages']}")
        print(f"Avg Messages/Session: {report['summary']['avg_messages_per_session']:.1f}")
        if report['summary']['avg_response_time_seconds']:
            print(f"Avg Response Time: {report['summary']['avg_response_time_seconds']:.2f}s")
        print("="*60)

        for session in report['sessions']:
            print(f"\nSession: {session['file']}")
            print(f"  Messages: {session['total_messages']}")
            if session['avg_response_time']:
                print(f"  Avg Response Time: {session['avg_response_time']:.2f}s")
    else:
        print("Usage: python chat_logger.py analyze")
