import os
import time
from datetime import datetime
import config
from speaker.verifier import SpeakerVerifier
from asr.transcriber import ASRTranscriber
from conversation.analyzer import ConversationAnalyzer
from conversation.schemas import Person2Output, SpeakerOutput

class Person2Pipeline:
    """
    Person 2 Core Orchestration Engine.
    Coordinates Branch A (Speaker Verification) and Branch B (ASR + Conversation AI),
    with DoS protection, and formats the final output payload for Person 3 (Risk Engine).
    """
    def __init__(self, verifier: SpeakerVerifier = None, transcriber: ASRTranscriber = None, analyzer: ConversationAnalyzer = None):
        self.verifier = verifier or SpeakerVerifier()
        self.transcriber = transcriber or ASRTranscriber()
        self.analyzer = analyzer or ConversationAnalyzer()

    def process_chunk(self, audio_path: str, reference_speaker_id: str = None) -> Person2Output:
        """
        Executes Person 2 pipeline on incoming live audio chunk with DoS size verification.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Fix 4: DoS Attack Defense - Reject files larger than MAX_AUDIO_FILE_SIZE_BYTES (10MB)
        file_size = os.path.getsize(audio_path)
        max_size = getattr(config, "MAX_AUDIO_FILE_SIZE_BYTES", 10 * 1024 * 1024)
        if file_size > max_size:
            raise ValueError(
                f"Security Alert (DoS Defense): Audio file size ({round(file_size / (1024*1024), 2)}MB) "
                f"exceeds maximum allowed limit of {max_size // (1024*1024)}MB. Processing rejected."
            )

        start_time = time.time()
        
        # 1. Branch A: Speaker Consistency
        if reference_speaker_id:
            try:
                speaker_res = self.verifier.verify(reference_speaker_id, audio_path)
                speaker_out = SpeakerOutput(
                    match_score=speaker_res["match_score"],
                    confidence=speaker_res["confidence"],
                    speaker_id=speaker_res["speaker_id"]
                )
            except Exception as e:
                print(f"[Pipeline] Speaker verification failed ({e}). Defaulting to neutral consistency.")
                speaker_out = SpeakerOutput(match_score=0.5, confidence=0.5, speaker_id=reference_speaker_id)
        else:
            speaker_out = SpeakerOutput(match_score=1.0, confidence=0.0, speaker_id=None)

        # 2. Branch B: ASR Transcription
        asr_res = self.transcriber.transcribe(audio_path)
        transcript = asr_res.get("text", "")

        # 3. Branch B: Conversation Intelligence
        conv_out = self.analyzer.analyze(transcript)

        # 4. Assemble Person 2 Standard Contract
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        output = Person2Output(
            speaker=speaker_out,
            conversation=conv_out,
            timestamp=datetime.utcnow().isoformat() + "Z",
            latency_ms=latency_ms
        )
        return output
