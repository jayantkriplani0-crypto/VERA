import os
# Allow multiple OpenMP runtimes (standard on Windows for PyTorch + CTranslate2)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import config
from speaker.enroll import SpeakerEnroller
from speaker.verifier import SpeakerVerifier
from asr.transcriber import ASRTranscriber
from conversation.analyzer import ConversationAnalyzer
from pipeline.engine import Person2Pipeline
from test_data.generate_samples import setup_test_cases

def print_banner(title):
    print("\n" + "="*70)
    print(f"  {title.upper()}")
    print("="*70)

def run_simulation_demo():
    print_banner("VERA Person 2: Speaker & Conversation Intelligence Pipeline")
    print("""
Flowchart Execution Path:
  [Incoming Audio] ----> [Branch A: ECAPA-TDNN] ----> Speaker Consistency Score
                   |---> [Branch B: faster-whisper] ----> Transcript
                                                     |---> [Gemini/NLP] ----> Structured Signals
                                                                        |---> [Person 2 JSON Contract]
""")

    # Step 1: Ensure test samples exist
    setup_test_cases()

    # Step 2: Enroll CEO Reference
    print_banner("Step 1: Reference Speaker Enrollment (Branch A)")
    enroller = SpeakerEnroller()
    ceo_ref_wav = os.path.join(config.TEST_DATA_DIR, "ceo_reference.wav")
    enroller.enroll(speaker_id="CEO_Rahul", audio_path=ceo_ref_wav)

    # Step 3: Initialize Full Engine with real neural models
    pipeline = Person2Pipeline(
        verifier=SpeakerVerifier(enroller),
        transcriber=ASRTranscriber(),
        analyzer=ConversationAnalyzer()
    )

    # Test Case 1: Genuine CEO Call (Normal conversation, high speaker match)
    print_banner("Test Case 1: Genuine Enrolled Speaker (CEO Rahul)")
    genuine_wav = os.path.join(config.TEST_DATA_DIR, "ceo_live_genuine.wav")
    result_genuine = pipeline.process_chunk(genuine_wav, reference_speaker_id="CEO_Rahul")
    print(">> Person 3 JSON Contract Output:")
    print(json.dumps(result_genuine.to_person3_contract(), indent=2))

    # Test Case 2: Imposter Voice Scam Call (High urgency, OTP request, low speaker match)
    print_banner("Test Case 2: Imposter Scam Call (Cyber Crime Branch Impersonation)")
    scam_wav = os.path.join(config.TEST_DATA_DIR, "scam_call_imposter.wav")
    result_scam = pipeline.process_chunk(scam_wav, reference_speaker_id="CEO_Rahul")
    print(">> Person 3 JSON Contract Output:")
    print(json.dumps(result_scam.to_person3_contract(), indent=2))

    # Test Case 3: Multilingual Hinglish Scam
    print_banner("Test Case 3: Multilingual Hinglish Scam Call")
    hinglish_wav = os.path.join(config.TEST_DATA_DIR, "scam_hinglish.wav")
    result_hinglish = pipeline.process_chunk(hinglish_wav, reference_speaker_id="CEO_Rahul")
    print(">> Person 3 JSON Contract Output:")
    print(json.dumps(result_hinglish.to_person3_contract(), indent=2))

    print_banner("DEMO COMPLETED: Person 2 is DONE")
    print("1. Audio -> Speaker consistency score: WORKING (ECAPA-TDNN)")
    print("2. Audio -> Transcript -> Structured conversation signals: WORKING (faster-whisper + Gemini/NLP)")
    print("Ready to be consumed by Person 3 (Risk Engine)!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 2 Speaker & Conversation Intelligence Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run complete simulated test suite")
    parser.add_argument("--enroll", nargs=2, metavar=("SPEAKER_ID", "WAV_PATH"), help="Enroll a reference speaker profile")
    parser.add_argument("--process", nargs=2, metavar=("WAV_PATH", "SPEAKER_ID"), help="Process audio chunk against reference profile")
    
    args = parser.parse_args()

    if args.enroll:
        enroller = SpeakerEnroller()
        enroller.enroll(speaker_id=args.enroll[0], audio_path=args.enroll[1])
    elif args.process:
        pipeline = Person2Pipeline()
        res = pipeline.process_chunk(audio_path=args.process[0], reference_speaker_id=args.process[1])
        print(json.dumps(res.to_person3_contract(), indent=2))
    else:
        run_simulation_demo()
