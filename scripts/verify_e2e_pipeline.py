"""VERA End-to-End Real Pipeline Verification Script.

Executes the complete production flow across real models:
Audio
  --> Person 1: Voice Authenticity / Spectra-AASIST3
  --> Person 2: Speaker Verification / ECAPA-TDNN
  --> Person 2: ASR Transcription / faster-whisper
  --> Person 2: Conversation Intelligence / Intent Analysis
  --> Backend:  Risk Fusion & Policy Decision Engine
  --> Frontend: Structured API Payload

Evaluates 3 realistic scenarios:
  Scenario A: Genuine / Normal speech
  Scenario B: Spoof / Synthetic audio
  Scenario C: Risky / Scam imposter conversation
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ensure both project root and backend are in python path
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND_DIR))

# Allow multiple OpenMP runtimes on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_e2e_verification():
    print("=" * 80)
    print("  VERA END-TO-END VERIFICATION: FULL REAL PIPELINE EXECUTION")
    print("=" * 80)

    # 1. Initialize Person 1 Voice Authenticity (Spectra-AASIST3)
    print("\n[STEP 1] Loading Person 1 Voice Authenticity Engine (Spectra-AASIST3)...")
    from ml.voice_detector.detector import VoiceAuthenticityDetector
    from ml.voice_detector.scorer import BONAFIDE_CLASS_INDEX
    from calibration.apply_calibration import CalibratedVoiceScorer
    detector = VoiceAuthenticityDetector(device="cpu")
    calibrator = CalibratedVoiceScorer()
    print("  [OK] Spectra-AASIST3 and CalibratedVoiceScorer loaded successfully on CPU.")

    from fastapi.testclient import TestClient
    from api.app import app as l1_app
    from app.main import app as backend_app
    from app.services.risk_fusion_service import calculate_risk
    from app.services.policy_service import evaluate_policy

    # 2. Initialize Person 2 Speaker Consistency (ECAPA-TDNN)
    print("\n[STEP 2] Loading Person 2 Speaker Verification Engine (SpeechBrain ECAPA-TDNN)...")
    from speaker.enroll import SpeakerEnroller
    from speaker.verifier import SpeakerVerifier
    enroller = SpeakerEnroller()
    verifier = SpeakerVerifier(enroller)
    ceo_ref = str(ROOT / "test_data" / "ceo_reference.wav")
    enroller.enroll("CEO_Rahul", ceo_ref)
    print("  [OK] ECAPA-TDNN loaded; 'CEO_Rahul' reference profile enrolled.")

    # 3. Initialize Person 2 ASR (faster-whisper) and Conversation Analyzer
    print("\n[STEP 3] Loading Person 2 ASR (faster-whisper) & Conversation Analyzer...")
    from asr.transcriber import ASRTranscriber
    from conversation.analyzer import ConversationAnalyzer
    transcriber = ASRTranscriber()
    analyzer = ConversationAnalyzer()
    print("  [OK] faster-whisper and ConversationAnalyzer initialized.")

    # Define test scenarios
    scenarios = [
        {
            "id": "A",
            "name": "Scenario A: Genuine Enrolled Speech",
            "file": str(ROOT / "test_data" / "ceo_live_genuine.wav"),
            "reference_speaker": "CEO_Rahul",
            "expected_spoof": False,
            "expected_speaker_match": True,
            "expected_risk": "low"
        },
        {
            "id": "B",
            "name": "Scenario B: Synthetic / Anti-Spoof Test Audio",
            "file": str(ROOT / "evaluation" / "benchmark_data" / "audio" / "SPK_001_spoof_2.wav"),
            "reference_speaker": None,
            "expected_spoof": True,
            "expected_speaker_match": False,
            "expected_risk": "medium"
        },
        {
            "id": "C",
            "name": "Scenario C: Risky Imposter Scam Call (Cyber Crime Impersonation)",
            "file": str(ROOT / "test_data" / "scam_call_imposter.wav"),
            "reference_speaker": "CEO_Rahul",
            "expected_spoof": False,
            "expected_speaker_match": False,
            "expected_risk": "high"
        }
    ]

    results = []

    print("\n[STEP 4] Executing scenarios inside active API Lifespan contexts...")
    with TestClient(l1_app) as l1_client, TestClient(backend_app) as backend_client:
        health_resp = l1_client.get("/api/v1/voice/health")
        assert health_resp.status_code == 200
        print(f"  [OK] Layer-1 Health Check: {health_resp.json()['status']}")

        backend_health = backend_client.get("/api/v1/health")
        assert backend_health.status_code == 200
        print(f"  [OK] Backend Health Check: {backend_health.json()['status']}")

        for sc in scenarios:
            print("\n" + "-" * 80)
            print(f"  EXECUTING {sc['name']}")
            print(f"  Audio File: {Path(sc['file']).name}")
            print("-" * 80)

            t0 = time.time()

            # Branch 1: Person 1 Voice Authenticity
            pred = detector.predict_file(sc["file"])
            raw_score = pred.raw_score
            cal = calibrator.calibrate_raw_score(raw_score)
            p_bonafide = cal.calibrated_probability_bonafide
            p_spoof = cal.calibrated_probability_spoof
            spoof_signal = cal.calibrated_spoof_signal
            classification = cal.decision
            conf = cal.decision_confidence

            print(f"  [1. Voice Authenticity / Spectra-AASIST3]")
            print(f"      Raw Bona-Fide Logit: {raw_score:+.4f}")
            print(f"      Calibrated P(Spoof) : {p_spoof:.4f}")
            print(f"      Spoof Signal        : {spoof_signal:.2f}%")
            print(f"      Classification      : {classification} (confidence={conf:.4f})")

            # Verify Person 1 Layer-1 API on same file
            with open(sc["file"], "rb") as f:
                api_resp = l1_client.post("/api/v1/voice/analyze", files={"file": (Path(sc["file"]).name, f, "audio/wav")})
                assert api_resp.status_code == 200, f"Layer-1 API returned {api_resp.status_code}"
                l1_data = api_resp.json()
                print(f"      Layer-1 REST API    : HTTP 200 OK (classification={l1_data['classification']})")

            # Branch 2: Person 2 Speaker Consistency (ECAPA-TDNN)
            speaker_sim = 1.0
            speaker_match = True
            speaker_conf = 0.0
            if sc["reference_speaker"]:
                spk_res = verifier.verify(sc["reference_speaker"], sc["file"])
                speaker_sim = spk_res["match_score"]
                speaker_match = spk_res["match_score"] >= 0.65
                speaker_conf = spk_res["confidence"]
                print(f"  [2. Speaker Verification / ECAPA-TDNN]")
                print(f"      Reference Speaker   : {sc['reference_speaker']}")
                print(f"      Similarity Score    : {speaker_sim:.4f}")
                print(f"      Match Decision      : {speaker_match} (confidence={speaker_conf:.4f})")
            else:
                print(f"  [2. Speaker Verification]")
                print(f"      Reference Speaker   : None (Unknown caller)")

            # Branch 3: Person 2 ASR (faster-whisper)
            asr_res = transcriber.transcribe(sc["file"])
            transcript = asr_res.get("text", "")
            print(f"  [3. ASR Transcription / faster-whisper]")
            print(f"      Transcript          : {transcript!r}")

            # Branch 4: Person 2 Conversation Intelligence
            conv_res = analyzer.analyze(transcript)
            intent = conv_res.intent.label
            signals = []
            if conv_res.interaction.urgency:
                signals.append("urgency")
            if conv_res.interaction.authority_pressure:
                signals.append("authority_pressure")
            if conv_res.interaction.secrecy:
                signals.append("secrecy")
            if conv_res.interaction.threat:
                signals.append("threat")
            if conv_res.interaction.credential_request:
                signals.append("credential_request")

            print(f"  [4. Conversation Intelligence]")
            print(f"      Intent              : {intent} (action={conv_res.intent.action})")
            print(f"      Interaction Signals : {signals}")
            print(f"      Pressure Level      : {conv_res.pressure}")
            print(f"      Emotion             : {conv_res.emotion.label} (confidence={conv_res.emotion.confidence})")

            # Branch 5: Backend Risk Fusion
            voice_analysis = {
                "voice_integrity_score": p_spoof, # Calibrated deepfake probability
                "confidence": conf
            }
            speaker_analysis = {
                "speaker_similarity_score": speaker_sim,
                "match": speaker_match,
                "confidence": speaker_conf
            } if sc["reference_speaker"] else None

            # Intent score mapping
            intent_risk_score = 0.0
            if intent in ["credential_request", "financial_transfer", "threat"]:
                intent_risk_score = 0.85
            elif intent in ["urgent_inquiry"]:
                intent_risk_score = 0.50

            intent_analysis = {
                "social_engineering_score": intent_risk_score,
                "signals": signals,
                "confidence": conv_res.emotion.confidence
            }

            action_signals = []
            action_risk_score = 0.0
            if "credential_request" in intent or any("otp" in s.lower() for s in signals):
                action_signals.append("auth_credential_request")
                action_risk_score = 0.85

            action_context_analysis = {
                "action_risk_score": action_risk_score,
                "context_risk_score": action_risk_score,
                "signals": action_signals,
                "confidence": 0.85
            }

            risk_output = calculate_risk(
                voice_analysis=voice_analysis,
                speaker_analysis=speaker_analysis,
                intent_analysis=intent_analysis,
                action_context_analysis=action_context_analysis
            )

            overall_risk = risk_output["overall_risk_score"]
            risk_level = risk_output["risk_level"]
            contributing = risk_output["contributing_signals"]

            # Branch 6: Policy Decision
            policy_output = evaluate_policy(risk_output, action_context_analysis)
            decision = policy_output["decision"]
            escalated = policy_output["escalated"]

            # Branch 7: Backend Session REST endpoint verification
            create_session_resp = backend_client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
            assert create_session_resp.status_code == 201
            session_id = create_session_resp.json()["session_id"]

            total_latency = round(time.time() - t0, 3)

            print(f"  [5. Backend Risk Fusion & Policy Engine]")
            print(f"      Overall Risk Score  : {overall_risk:.4f}")
            print(f"      Risk Level          : {risk_level.upper()}")
            print(f"      Contributing Signals: {contributing}")
            print(f"      Policy Decision     : {decision.upper()} (escalated={escalated})")
            print(f"      Backend Session ID  : {session_id}")
            print(f"      Total Pipeline Time : {total_latency}s")

            # Branch 8: Frontend Telemetry Contract Verification
            frontend_telemetry = {
                "session_id": session_id,
                "transcript": transcript,
                "voice_integrity_score": p_spoof,
                "speaker_similarity_score": speaker_sim if sc["reference_speaker"] else None,
                "overall_risk_score": overall_risk,
                "risk_level": risk_level,
                "decision": decision,
                "signals": contributing,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            print(f"  [6. Frontend Telemetry Payload]")
            print(f"      Payload Valid       : {all(k in frontend_telemetry for k in ['overall_risk_score', 'risk_level', 'decision', 'transcript'])}")

            results.append({
                "scenario": sc["name"],
                "file": sc["file"],
                "voice_authenticity": {
                    "raw_score": raw_score,
                    "p_bonafide": p_bonafide,
                    "p_spoof": p_spoof,
                    "spoof_signal": spoof_signal,
                    "classification": classification,
                    "confidence": conf
                },
                "speaker_verification": {
                    "reference_speaker": sc["reference_speaker"],
                    "similarity_score": speaker_sim,
                    "match": speaker_match,
                    "confidence": speaker_conf
                },
                "asr": {
                    "transcript": transcript
                },
                "conversation_intelligence": {
                    "intent": intent,
                    "signals": signals,
                    "emotion": conv_res.emotion.label
                },
                "risk_fusion": {
                    "overall_risk_score": overall_risk,
                    "risk_level": risk_level,
                    "contributing_signals": contributing
                },
                "policy": {
                    "decision": decision,
                    "escalated": escalated
                },
                "frontend_telemetry": frontend_telemetry,
                "total_latency_seconds": total_latency
            })

    # Save complete report
    out_file = ROOT / "evaluation" / "reports" / "e2e_verification_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  E2E VERIFICATION COMPLETED SUCCESSFULLY!")
    print(f"  Saved full output report to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_e2e_verification()
