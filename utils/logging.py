"""Logging utilities for the application."""
import json
import os
from datetime import datetime
from typing import Dict, Optional

# Portable path: log next to the app root (works when deployed)
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "debug.log")


def log(hypothesis_id: str, location: str, message: str, data: Optional[Dict] = None):
    """Log an event to the debug log file."""
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": location,
            "message": message,
            "data": data or {}
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass

