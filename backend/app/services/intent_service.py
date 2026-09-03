import re

def analyze_intent(transcript: str) -> dict:
    """
    Lightweight deterministic/rule-based detector for social engineering intents.
    Returns a social_engineering_score (0.0 to 1.0), signals, category, and confidence.
    """
    text = transcript.lower()
    
    signals = []
    score = 0.0
    
    def check_keywords(keywords):
        # Use word boundaries to prevent substring false positives (e.g., "fine" in "define")
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b')
        return bool(pattern.search(text))
    
    # 1. Urgency / Time Pressure
    urgency_keywords = ["immediate", "urgent", "right now", "quickly", "before it's too late", "expires", "action required"]
    if check_keywords(urgency_keywords):
        signals.append("urgency")
        score += 0.3
        
    # 2. Threats / Consequences
    threat_keywords = ["arrest", "police", "suspended", "blocked", "legal action", "warrant", "penalty"]
    if check_keywords(threat_keywords):
        signals.append("threat")
        score += 0.4
        
    # 3. Requests for OTP/Password/PIN/Sensitive Info
    auth_keywords = ["otp", "password", "pin", "verification code", "social security", "one time password", "access code"]
    if check_keywords(auth_keywords):
        signals.append("request_auth_code")
        score += 0.6
        
    # 4. Requests for Money/Payment
    payment_keywords = ["wire transfer", "gift card", "crypto", "bitcoin", "pay now", "send money", "western union", "bank details"]
    if check_keywords(payment_keywords):
        signals.append("request_payment")
        score += 0.5
        
    # 5. Impersonation Claims
    impersonation_keywords = ["irs", "tax agency", "tech support", "microsoft support", "bank fraud department", "fbi", "government"]
    if check_keywords(impersonation_keywords):
        signals.append("impersonation_claim")
        score += 0.3
        
    # 6. Secrecy / Isolation Requests
    secrecy_keywords = ["don't tell anyone", "keep this private", "secret", "stay on the line", "do not hang up"]
    if check_keywords(secrecy_keywords):
        signals.append("secrecy_isolation")
        score += 0.3
        
    # 7. Suspicious Links / Actions
    action_keywords = ["download this", "click the link", "install anydesk", "teamviewer", "remote access", "open this website"]
    if check_keywords(action_keywords):
        signals.append("suspicious_action")
        score += 0.4

    # Cap score at 1.0
    final_score = min(1.0, score)
    
    # Determine Category
    if final_score >= 0.7:
        category = "high_risk_scam"
    elif final_score >= 0.4:
        category = "suspicious_request"
    elif final_score > 0:
        category = "mildly_suspicious"
    else:
        category = "benign"
        
    # Mock confidence based on how many signals triggered
    confidence = min(1.0, 0.5 + (len(signals) * 0.1)) if signals else 0.9
    
    return {
        "intent_category": category,
        "social_engineering_score": final_score,
        "signals": signals,
        "confidence": confidence
    }
