import json
import os
from datetime import datetime

LOG_FILE = "rag_logs.jsonl"  # JSON Lines format

def save_log(entry):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **entry
    }
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if os.path.dirname(LOG_FILE) else None
    
    # Append as JSON line
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
