import re

def analyze_action_context(transcript: str, intent_analysis: dict) -> dict:
    """
    Lightweight deterministic/rule-based detector for action and context sensitivity.
    Accepts the raw transcript and the output from the intent service.
    Returns action_category, action_risk_score, context_risk_score, signals, and confidence.
    """
    text = transcript.lower()
    signals = []
    action_risk_score = 0.0
    context_risk_score = 0.0
    
    intent_signals = intent_analysis.get("signals", [])
    
    def check_keywords(keywords):
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b')
        return bool(pattern.search(text))
    
    # --- Action Analysis ---
    
    # 1. OTP/Password/PIN requests
    if "request_auth_code" in intent_signals or check_keywords(["otp", "password", "pin", "verification code"]):
        signals.append("auth_credential_request")
        action_risk_score += 0.8
        
    # 2. Money/payment/UPI/bank transfer requests
    if "request_payment" in intent_signals or check_keywords(["upi", "bank transfer", "wire", "pay now", "send money"]):
        signals.append("financial_transaction_request")
        action_risk_score += 0.7
        
    # 3. Account or credential changes
    if check_keywords(["change password", "reset password", "update details", "verify account"]):
        signals.append("account_modification_request")
        action_risk_score += 0.6
        
    # 4. Requests to click/open links or install software
    if "suspicious_action" in intent_signals or check_keywords(["click", "open the link", "download", "install", "anydesk"]):
        signals.append("risky_digital_action")
        action_risk_score += 0.6
        
    # --- Context Analysis ---
    
    # 1. Urgency / High Pressure
    if "urgency" in intent_signals or "threat" in intent_signals:
        signals.append("high_pressure_context")
        context_risk_score += 0.8
        
    # 2. Financial/Irreversible Impact
    if "financial_transaction_request" in signals:
        signals.append("financial_impact_context")
        context_risk_score += 0.7
        
    # 3. Credential Sensitivity
    if "auth_credential_request" in signals or "account_modification_request" in signals:
        signals.append("credential_sensitivity_context")
        context_risk_score += 0.7
        
    # 4. Unusual Verification Bypass or Isolation
    if "secrecy_isolation" in intent_signals or check_keywords(["bypass", "skip verification", "don't hang up"]):
        signals.append("verification_bypass_isolation_context")
        context_risk_score += 0.6
        
    # Cap scores at 1.0
    action_risk_score = min(1.0, action_risk_score)
    
    # Heuristic: No action/context risk is added unless an actual sensitive action is detected.
    if action_risk_score == 0.0:
        context_risk_score = 0.0
    else:
        context_risk_score = min(1.0, context_risk_score)
    
    # Determine Action Category
    if action_risk_score >= 0.7:
        action_category = "high_risk_action"
    elif action_risk_score >= 0.4:
        action_category = "medium_risk_action"
    else:
        action_category = "low_risk_action"
        
    # Simple confidence
    confidence = min(1.0, 0.6 + (len(signals) * 0.1)) if signals else 0.9
    
    return {
        "action_category": action_category,
        "action_risk_score": action_risk_score,
        "context_risk_score": context_risk_score,
        "signals": list(set(signals)),  # Deduplicate just in case
        "confidence": confidence
    }
