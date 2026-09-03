import { useState, useRef, useCallback, useEffect } from 'react';

export interface UseAudioRecorderResult {
  isRecording: boolean;
  recordingTime: number;
  audioBlob: Blob | null;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  clearRecording: () => void;
}

// ── WAV Encoding Helpers ────────────────────────────────────────────────────
// These are the same helpers used by useLiveDetection.ts (WebSocket path).
// By re-encoding to WAV here, the backend always receives a format that
// soundfile/librosa can reliably decode — no new backend dependencies needed.

function floatTo16BitPCM(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function encodeWAV(samples: Int16Array, sampleRate = 16000): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.byteLength);
  const view = new DataView(buffer);
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.byteLength, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);          // PCM chunk size
  view.setUint16(20, 1, true);           // PCM format
  view.setUint16(22, 1, true);           // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);           // block align
  view.setUint16(34, 16, true);          // bits per sample
  writeString(view, 36, 'data');
  view.setUint32(40, samples.byteLength, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    view.setInt16(offset, samples[i], true);
  }
  return buffer;
}

/**
 * Re-encode any browser Blob (WebM/Opus, etc.) into a 16 kHz mono PCM WAV
 * using the Web Audio API's AudioContext.decodeAudioData.
 *
 * Returns a WAV Blob ready for the VERA backend. Throws if decoding fails.
 */
async function reencodeToWav(blob: Blob): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer();

  // AudioContext.decodeAudioData handles WebM/Opus, WAV, MP3, OGG natively in
  // every modern browser — no extra packages needed.
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  let decoded: AudioBuffer;
  try {
    decoded = await audioCtx.decodeAudioData(arrayBuffer);
  } finally {
    await audioCtx.close();
  }

  // Mix down to mono
  const numFrames = decoded.length;
  const mixed = new Float32Array(numFrames);
  const numChannels = decoded.numberOfChannels;
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = decoded.getChannelData(ch);
    for (let i = 0; i < numFrames; i++) {
      mixed[i] += channelData[i];
    }
  }
  if (numChannels > 1) {
    for (let i = 0; i < numFrames; i++) {
      mixed[i] /= numChannels;
    }
  }

  const pcm16 = floatTo16BitPCM(mixed);
  const wavBuffer = encodeWAV(pcm16, 16000);
  return new Blob([wavBuffer], { type: 'audio/wav' });
}

// ── Hook ────────────────────────────────────────────────────────────────────

export const useAudioRecorder = (): UseAudioRecorderResult => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setAudioBlob(null);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Audio recording is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const rawMime = mediaRecorder.mimeType || 'audio/webm';
        const rawBlob = new Blob(chunksRef.current, { type: rawMime });
        chunksRef.current = [];

        // Re-encode to 16 kHz mono WAV so the backend (librosa/soundfile)
        // can always parse it, regardless of what MediaRecorder produced.
        try {
          const wavBlob = await reencodeToWav(rawBlob);
          setAudioBlob(wavBlob);
        } catch (encErr) {
          console.error('[useAudioRecorder] WAV re-encoding failed:', encErr);
          setError(
            encErr instanceof Error
              ? `Audio encoding failed: ${encErr.message}`
              : 'Audio encoding failed. Try recording again.'
          );
          setAudioBlob(null);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to access microphone.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, [isRecording]);

  const clearRecording = useCallback(() => {
    setAudioBlob(null);
    setRecordingTime(0);
    setError(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== 'inactive'
      ) {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }
    };
  }, []);

  return {
    isRecording,
    recordingTime,
    audioBlob,
    error,
    startRecording,
    stopRecording,
    clearRecording,
  };
};
