import json
import hashlib
from datetime import datetime
from typing import Dict, Any

def generate_canonical_json(data: Dict[str, Any]) -> str:
    """
    Generates a deterministic canonical JSON string from a dictionary.
    Keys are sorted to ensure identical data produces identical JSON.
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'))

def generate_evidence_record(session_id: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the evidence record and its SHA-256 hash.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Construct the base record
    record = {
        "session_id": session_id,
        "timestamp": timestamp,
        "voice_analysis": analysis_data.get("voice_analysis", {}),
        "speaker_analysis": analysis_data.get("speaker_analysis"), # Can be None
        "transcript": analysis_data.get("transcript", ""),
        "intent": analysis_data.get("intent", {}),
        "action_context": analysis_data.get("action_context", {}),
        "risk": analysis_data.get("risk", {}),
        "policy_decision": analysis_data.get("policy_decision", {})
    }
    
    canonical_json = generate_canonical_json(record)
    record_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    return {
        "evidence_record": record,
        "hash": record_hash,
        "algorithm": "SHA-256"
    }
