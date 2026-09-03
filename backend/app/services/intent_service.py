_analyzer = None

def get_model():
    global _analyzer
    if _analyzer is None:
        from conversation.analyzer import ConversationAnalyzer
        _analyzer = ConversationAnalyzer()
    return _analyzer

def analyze_intent(transcript: str) -> dict:
    if not transcript:
        return {
            "intent_category": "benign",
            "social_engineering_score": 0.0,
            "signals": [],
            "confidence": 0.9
        }

    analyzer = get_model()
    try:
        # Note: ConversationOutput is a Pydantic model
        result = analyzer.analyze(transcript)
        
        signals = []
        if result.interaction.urgency: signals.append("urgency")
        if result.interaction.secrecy: signals.append("secrecy_isolation")
        if result.interaction.authority_pressure: signals.append("impersonation_claim")
        if result.interaction.threat: signals.append("threat")
        if result.interaction.credential_request: signals.append("request_auth_code")
        if result.intent.label == "financial_transfer": signals.append("request_payment")
        
        score_map = {"high": 0.9, "medium": 0.6, "low": 0.1}
        score = score_map.get(result.pressure, 0.1)
        
        if score >= 0.7: category = "high_risk_scam"
        elif score >= 0.4: category = "suspicious_request"
        elif score > 0.1: category = "mildly_suspicious"
        else: category = "benign"
        
        return {
            "intent_category": category,
            "social_engineering_score": score,
            "signals": signals,
            "confidence": result.emotion.confidence
        }
    except Exception as e:
        import logging
        logging.getLogger("vera.intent_service").error(f"Intent analysis failed: {e}")
        return {
            "intent_category": "benign",
            "social_engineering_score": 0.0,
            "signals": [],
            "confidence": 0.0
        }
