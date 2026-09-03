import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import config
from speaker.enroll import SpeakerEnroller
from speaker.verifier import SpeakerVerifier
from asr.transcriber import ASRTranscriber
from conversation.analyzer import ConversationAnalyzer
from pipeline.engine import Person2Pipeline

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("="*75)
    print("  VERA PERSON 2: LIVE REAL-TIME DEMO RUNNER")
    print("="*75)

    api_key = os.environ.get("GEMINI_API_KEY") or config.GEMINI_API_KEY
    if not api_key:
        print("\n[!] No GEMINI_API_KEY detected in environment or config.py.")
        user_key = input("Enter your Gemini API Key (or press Enter to use NLP engine): ").strip()
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
            config.GEMINI_API_KEY = user_key
            api_key = user_key

    if api_key:
        print(f"[*] Live Gemini API: ACTIVE (Key: {api_key[:6]}...{api_key[-4:]})")
    else:
        print("[*] Live Gemini API: Using built-in local NLP engine")

    print("\n[1/3] Initializing SpeechBrain ECAPA-TDNN neural network...")
    enroller = SpeakerEnroller()
    verifier = SpeakerVerifier(enroller)

    print("[2/3] Initializing faster-whisper ASR model...")
    transcriber = ASRTranscriber()

    print("[3/3] Initializing Conversation AI Analyzer...")
    analyzer = ConversationAnalyzer(api_key=api_key)

    pipeline = Person2Pipeline(verifier=verifier, transcriber=transcriber, analyzer=analyzer)

    # Ensure reference profile exists
    ref_audio = os.path.join(config.TEST_DATA_DIR, "ceo_reference.wav")
    if not os.path.exists(ref_audio):
        from test_data.generate_samples import setup_test_cases
        setup_test_cases()
    enroller.enroll(speaker_id="CEO_Rahul", audio_path=ref_audio)

    while True:
        print("\n" + "-"*75)
        print("CHOOSE A LIVE TEST OPTION:")
        print("  1. Test custom scam / normal dialogue text with Live Gemini AI")
        print("  2. Test WAV audio file against enrolled speaker (ECAPA + Whisper + Gemini)")
        print("  3. Run full automated 3-case benchmark")
        print("  4. Exit")
        print("-"*75)

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            print("\nType or paste any incoming call transcript (English, Hindi, or Hinglish):")
            print("Examples:")
            print(" - 'Hi Rahul, let us discuss the project update tomorrow.'")
            print(" - 'Sir I am calling from Mumbai Police cyber cell, share your OTP now or you will be arrested.'")
            print(" - 'Aapka bank account freeze ho gaya hai, turant 5 digit verification code batayein.'")
            text = input("\nEnter dialogue: ").strip()
            if text:
                print("\n[AI] Analyzing dialogue with Gemini API in real time...")
                t0 = time.time()
                res = analyzer.analyze(text)
                dt = round((time.time() - t0) * 1000, 2)
                
                print("\n" + "="*50)
                print("  GEMINI CONVERSATION INTELLIGENCE SIGNALS")
                print("="*50)
                print(f"Transcript:          {res.transcript}")
                print(f"Intent:              {res.intent.label} (Action: {res.intent.action})")
                print(f"Urgency:             {'🚨 TRUE' if res.interaction.urgency else 'False'}")
                print(f"Secrecy:             {'🤫 TRUE' if res.interaction.secrecy else 'False'}")
                print(f"Authority Pressure:  {'👮 TRUE' if res.interaction.authority_pressure else 'False'}")
                print(f"Threat Detected:     {'⚠️ TRUE' if res.interaction.threat else 'False'}")
                print(f"Credential Request:  {'🔑 TRUE' if res.interaction.credential_request else 'False'}")
                print(f"Tone / Emotion:      {res.emotion.label} (Confidence: {res.emotion.confidence})")
                print(f"Pressure Level:      {res.pressure.upper()}")
                print(f"API Latency:         {dt} ms")
                print("="*50)

        elif choice == "2":
            wav_path = input("Enter path to audio WAV file (e.g. test_data/scam_call_imposter.wav): ").strip()
            if not wav_path:
                wav_path = os.path.join(config.TEST_DATA_DIR, "scam_call_imposter.wav")
            
            speaker_id = input("Enter reference speaker profile to match against [default: CEO_Rahul]: ").strip()
            if not speaker_id:
                speaker_id = "CEO_Rahul"

            print(f"\nProcessing '{wav_path}' through ECAPA + Whisper + Gemini...")
            result = pipeline.process_chunk(audio_path=wav_path, reference_speaker_id=speaker_id)
            print("\n>> PERSON 3 JSON CONTRACT OUTPUT:")
            print(json.dumps(result.to_person3_contract(), indent=2))

        elif choice == "3":
            from main import run_simulation_demo
            run_simulation_demo()

        elif choice == "4":
            print("Exiting. Have a great demo!")
            break
        else:
            print("Invalid selection, please choose 1-4.")

if __name__ == "__main__":
    main()
