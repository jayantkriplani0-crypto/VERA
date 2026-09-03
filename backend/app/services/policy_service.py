def evaluate_policy(risk_analysis: dict, action_context_analysis: dict) -> dict:
    """
    Evaluates the fused risk score and specific action contexts to determine a final policy decision.
    Rules:
    - low risk → allow
    - medium risk → warn
    - high risk → verify
    - critical risk → block
    
    Escalation:
    - Critical actions (OTP, financial, remote access, account takeover) escalate the decision by one level.
    """
    risk_level = risk_analysis.get("risk_level", "low")
    
    # Base decision
    base_mapping = {
        "low": "allow",
        "medium": "warn",
        "high": "verify",
        "critical": "block"
    }
    
    decision_hierarchy = ["allow", "warn", "verify", "block"]
    
    base_decision = base_mapping.get(risk_level, "allow")
    current_level_idx = decision_hierarchy.index(base_decision)
    
    # Check for critical actions
    signals = action_context_analysis.get("signals", [])
    critical_signals = {
        "auth_credential_request", 
        "financial_transaction_request", 
        "risky_digital_action", 
        "account_modification_request"
    }
    
    found_critical = any(sig in critical_signals for sig in signals)
    
    escalated = False
    final_decision = base_decision
    reason = f"Base risk level is {risk_level}."
    
    if found_critical and current_level_idx < len(decision_hierarchy) - 1:
        # Escalate decision
        final_decision = decision_hierarchy[current_level_idx + 1]
        escalated = True
        reason = f"Escalated from {base_decision} to {final_decision} due to critical action signals."
        
    return {
        "decision": final_decision,
        "escalated": escalated,
        "reason": reason
    }
