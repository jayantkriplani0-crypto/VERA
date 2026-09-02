def calculate_risk(voice_analysis: dict = None, speaker_analysis: dict = None, intent_analysis: dict = None, action_context_analysis: dict = None) -> dict:
    """
    Deterministically fuses individual risk scores into a unified overall risk score.
    """
    signals = []
    total_weight = 0.0
    weighted_sum = 0.0
    total_confidence = 0.0
    confidence_weight = 0.0
    
    # 1. Voice Integrity (Weight: 40%)
    if voice_analysis:
        vi_score = voice_analysis.get("voice_integrity_score", 0.0)
        vi_conf = voice_analysis.get("confidence", 1.0)
        
        weighted_sum += vi_score * 0.40
        total_weight += 0.40
        total_confidence += vi_conf * 0.40
        confidence_weight += 0.40
        
        if vi_score > 0.6:
            signals.append("high_voice_manipulation_probability")
            
    # 2. Speaker Consistency (Weight: 20%)
    if speaker_analysis and "speaker_similarity_score" in speaker_analysis:
        sim_score = speaker_analysis.get("speaker_similarity_score")
        if sim_score is not None:
            # Lower similarity = higher risk
            speaker_risk = max(0.0, 1.0 - sim_score)
            sp_conf = speaker_analysis.get("confidence", 1.0)
            
            weighted_sum += speaker_risk * 0.20
            total_weight += 0.20
            total_confidence += sp_conf * 0.20
            confidence_weight += 0.20
            
            if not speaker_analysis.get("match", True):
                signals.append("speaker_mismatch")
            
    # 3. Intent/Interaction (Weight: 15%)
    if intent_analysis:
        in_score = intent_analysis.get("social_engineering_score", 0.0)
        in_conf = intent_analysis.get("confidence", 1.0)
        
        weighted_sum += in_score * 0.15
        total_weight += 0.15
        total_confidence += in_conf * 0.15
        confidence_weight += 0.15
        
        signals.extend(intent_analysis.get("signals", []))
        
    # 4. Action/Context (Weight: 25%)
    if action_context_analysis:
        # Take the maximum of action or context risk
        ac_score = max(
            action_context_analysis.get("action_risk_score", 0.0),
            action_context_analysis.get("context_risk_score", 0.0)
        )
        ac_conf = action_context_analysis.get("confidence", 1.0)
        
        weighted_sum += ac_score * 0.25
        total_weight += 0.25
        total_confidence += ac_conf * 0.25
        confidence_weight += 0.25
        
        # Deduplicate signals
        for sig in action_context_analysis.get("signals", []):
            if sig not in signals:
                signals.append(sig)
                
    # Normalize score based on available components
    overall_risk_score = 0.0
    final_confidence = 0.0
    
    if total_weight > 0:
        overall_risk_score = min(1.0, weighted_sum / total_weight)
        final_confidence = total_confidence / confidence_weight
        
    # Determine risk level
    if overall_risk_score >= 0.75:
        risk_level = "critical"
    elif overall_risk_score >= 0.5:
        risk_level = "high"
    elif overall_risk_score >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"
        
    return {
        "overall_risk_score": overall_risk_score,
        "risk_level": risk_level,
        "contributing_signals": list(set(signals)),
        "confidence": final_confidence
    }
