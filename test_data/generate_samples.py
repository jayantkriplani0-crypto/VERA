import os
import win32com.client
import config

def text_to_wav(text: str, filename: str, voice_index: int = 0, rate: int = 0):
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    voices = speaker.GetVoices()
    if voice_index < voices.Count:
        speaker.Voice = voices.Item(voice_index)
    speaker.Rate = rate
    
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    # 3 = SSFMCreateForWrite
    stream.Open(filename, 3)
    speaker.AudioOutputStream = stream
    speaker.Speak(text)
    stream.Close()

def setup_test_cases():
    os.makedirs(config.TEST_DATA_DIR, exist_ok=True)

    # 1. Enrolled CEO Reference Voice (Voice 0)
    ceo_ref_path = os.path.join(config.TEST_DATA_DIR, "ceo_reference.wav")
    text_to_wav("Hello team, this is Rahul. Welcome to our quarterly business review.", ceo_ref_path, voice_index=0, rate=0)

    # 2. Matching Genuine CEO Call (Same Voice 0)
    ceo_live_path = os.path.join(config.TEST_DATA_DIR, "ceo_live_genuine.wav")
    text_to_wav("Hi team, please review the quarterly budget spreadsheet before the 4 PM meeting.", ceo_live_path, voice_index=0, rate=0)

    # 3. Non-Matching Imposter Scam Voice (Faster speech rate / Imposter Voice)
    scam_imposter_path = os.path.join(config.TEST_DATA_DIR, "scam_call_imposter.wav")
    text_to_wav("Attention sir, this is Senior Officer Sharma from Cyber Crime Branch. Your account is linked to money laundering. Share your OTP immediately or police will be dispatched. Do not inform anyone.", scam_imposter_path, voice_index=1 if win32com.client.Dispatch("SAPI.SpVoice").GetVoices().Count > 1 else 0, rate=3)

    # 4. Hinglish Scam Case
    hinglish_scam_path = os.path.join(config.TEST_DATA_DIR, "scam_hinglish.wav")
    text_to_wav("Sir your bank account will be blocked today. Send your OTP immediately and do not tell anyone.", hinglish_scam_path, voice_index=0, rate=2)

    print(f"[Setup] Generated real spoken speech WAV files in {config.TEST_DATA_DIR}")

if __name__ == "__main__":
    setup_test_cases()
