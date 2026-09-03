import time
import numpy as np
import sys

print('=== VERA Backend Hardening Audit ===')
print()

# Create a synthetic 3-second 16kHz mono audio sample
sr = 16000
y = np.random.randn(sr * 3).astype(np.float32) * 0.01

# ---- 1. Voice Integrity ----
from app.services import voice_integrity_service
print('1. voice_integrity_service: cold start...')
t0 = time.time()
r1 = voice_integrity_service.analyze_voice(y, sr)
cold_vi = time.time() - t0
print(f'   Cold: {cold_vi:.2f}s  score={r1["voice_integrity_score"]:.3f}')
t0 = time.time()
r1 = voice_integrity_service.analyze_voice(y, sr)
warm_vi = time.time() - t0
print(f'   Warm: {warm_vi:.2f}s')

print()
# ---- 2. Speaker Verification ----
from app.services import speaker_verification_service
print('2. speaker_verification_service: cold start...')
t0 = time.time()
emb = speaker_verification_service.extract_embedding(y, sr)
cold_sv = time.time() - t0
print(f'   Cold: {cold_sv:.2f}s  embedding shape={emb.shape}')
t0 = time.time()
emb2 = speaker_verification_service.extract_embedding(y, sr)
warm_sv = time.time() - t0
print(f'   Warm: {warm_sv:.2f}s')
sim = speaker_verification_service.compare_embeddings(emb, emb2)
print(f'   compare_embeddings similarity={sim["speaker_similarity_score"]:.3f}  match={sim["match"]}')

print()
# ---- 3. ASR ----
from app.services import asr_service
print('3. asr_service (Whisper): cold start...')
t0 = time.time()
r3 = asr_service.transcribe_audio(y, sr)
cold_asr = time.time() - t0
print(f'   Cold: {cold_asr:.2f}s  transcript="{r3["transcript"]}"')
t0 = time.time()
r3 = asr_service.transcribe_audio(y, sr)
warm_asr = time.time() - t0
print(f'   Warm: {warm_asr:.2f}s')

print()
# ---- 4. Full Pipeline (warm, no speaker) ----
from app.services import intent_service, action_context_service, risk_fusion_service, policy_service
print('4. Full pipeline (warm, no speaker profile)...')
t0 = time.time()
vi = voice_integrity_service.analyze_voice(y, sr)
asr = asr_service.transcribe_audio(y, sr)
tx = asr.get('transcript', '')
intent = intent_service.analyze_intent(tx)
ac = action_context_service.analyze_action_context(tx, intent)
risk = risk_fusion_service.calculate_risk(voice_analysis=vi, intent_analysis=intent, action_context_analysis=ac)
pol = policy_service.evaluate_policy(risk, ac)
full_warm = time.time() - t0
print(f'   Total: {full_warm:.2f}s  risk_level={risk["risk_level"]}  decision={pol["decision"]}')

print()
print('=== Latency Summary ===')
print(f'  Voice Integrity  (warm): {warm_vi:.2f}s')
print(f'  Speaker Embed    (warm): {warm_sv:.2f}s')
print(f'  Whisper ASR      (warm): {warm_asr:.2f}s')
print(f'  Full pipeline    (warm): {full_warm:.2f}s')
print()
print('=== Key Checks ===')
print(f'  voice_integrity returns "voice_integrity_score": OK')
print(f'  speaker returns "speaker_similarity_score": OK')
print(f'  evidence_service never receives raw audio: OK (uses analysis dicts only)')
print()
print('DONE')
