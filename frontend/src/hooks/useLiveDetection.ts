import { useState, useRef, useCallback, useEffect } from 'react';

export interface TelemetryData {
  session_id: string;
  transcript: string;
  voice_integrity_score?: number;
  speaker_similarity_score?: number | null;
  overall_risk_score?: number;
  risk_level?: string;
  decision?: string;
  signals?: string[];
  error?: string;
  timestamp?: string;
}

export type ConnectionState = 'Disconnected' | 'Connecting' | 'Live' | 'Processing' | 'Error';

export interface UseLiveDetectionResult {
  connectionState: ConnectionState;
  telemetry: TelemetryData | null;
  telemetryHistory: TelemetryData[];
  error: string | null;
  startLiveDetection: (sessionId: string) => Promise<void>;
  stopLiveDetection: () => void;
  getLiveSessionBlob: () => Blob | null;
}

const WS_BASE_URL = import.meta.env.VITE_API_BASE_URL
  ? (import.meta.env.VITE_API_BASE_URL as string).replace(/^http/, 'ws')
  : `ws://${window.location.host}`;

function floatTo16BitPCM(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output;
}

function encodeWAV(samples: Int16Array, sampleRate: number = 16000): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.byteLength);
  const view = new DataView(buffer);

  const writeString = (v: DataView, offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      v.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.byteLength, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // Bits per sample
  writeString(view, 36, 'data');
  view.setUint32(40, samples.byteLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    view.setInt16(offset, samples[i], true);
  }

  return buffer;
}

export const useLiveDetection = (): UseLiveDetectionResult => {
  const [connectionState, setConnectionState] = useState<ConnectionState>('Disconnected');
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryData[]>([]);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const pcmBufferRef = useRef<Int16Array[]>([]);
  const pcmLengthRef = useRef<number>(0);

  const fullSessionBufferRef = useRef<Int16Array[]>([]);
  const fullSessionLengthRef = useRef<number>(0);

  const cleanupResources = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    pcmBufferRef.current = [];
    pcmLengthRef.current = 0;
    // Don't clear fullSessionBufferRef here so we can generate evidence after stopping.
    // We'll clear it specifically inside startLiveDetection.
  }, []);

  const stopLiveDetection = useCallback(() => {
    cleanupResources();
    setConnectionState('Disconnected');
  }, [cleanupResources]);

  const startLiveDetection = useCallback(async (sessionId: string) => {
    cleanupResources();
    setConnectionState('Connecting');
    setError(null);
    setTelemetry(null);
    setTelemetryHistory([]);
    fullSessionBufferRef.current = [];
    fullSessionLengthRef.current = 0;

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone not supported on this browser.');
      }

      // Initialize WebSocket
      const wsUrl = `${WS_BASE_URL}/api/v1/ws/sessions/${sessionId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionState('Live');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.error) {
            console.error('WebSocket backend error:', data.error);
            // Non-fatal, keep connection open as per backend spec
            return;
          }
          data.timestamp = new Date().toISOString();
          setTelemetry(data);
          setTelemetryHistory(prev => [...prev, data]);
          setConnectionState('Live');
        } catch (err) {
          console.error('Malformed telemetry data', err);
        }
      };

      ws.onerror = (e) => {
        console.error('WebSocket Error', e);
        setError('WebSocket encountered an error.');
        setConnectionState('Error');
        cleanupResources();
      };

      ws.onclose = (e) => {
        if (e.code === 1008) {
          setError('Session not found or invalid.');
        } else if (e.code !== 1000) {
          setError('WebSocket disconnected unexpectedly.');
        }
        setConnectionState('Disconnected');
        cleanupResources();
      };

      // Initialize Web Audio API for PCM16 16kHz Mono capture
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      source.connect(processor);
      processor.connect(audioContext.destination);

      // We want to accumulate ~3 seconds of 16kHz (48,000 samples)
      // The backend triggers processing at 100,000 bytes (50,000 samples)
      const SAMPLES_THRESHOLD = 50000;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = floatTo16BitPCM(inputData);
        
        pcmBufferRef.current.push(pcm16);
        pcmLengthRef.current += pcm16.length;

        // ACCUMULATE FULL SESSION FOR EVIDENCE GENERATION
        fullSessionBufferRef.current.push(pcm16);
        fullSessionLengthRef.current += pcm16.length;

        if (pcmLengthRef.current >= SAMPLES_THRESHOLD) {
          setConnectionState('Processing');
          
          // Merge buffers
          const merged = new Int16Array(pcmLengthRef.current);
          let offset = 0;
          for (const buf of pcmBufferRef.current) {
            merged.set(buf, offset);
            offset += buf.length;
          }
          
          const wavBuffer = encodeWAV(merged, 16000);
          wsRef.current.send(wavBuffer);
          
          pcmBufferRef.current = [];
          pcmLengthRef.current = 0;
        }
      };

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start live detection.');
      setConnectionState('Error');
      cleanupResources();
    }
  }, [cleanupResources]);

  const getLiveSessionBlob = useCallback((): Blob | null => {
    if (fullSessionLengthRef.current === 0) return null;
    const merged = new Int16Array(fullSessionLengthRef.current);
    let offset = 0;
    for (const buf of fullSessionBufferRef.current) {
      merged.set(buf, offset);
      offset += buf.length;
    }
    const wavBuffer = encodeWAV(merged, 16000);
    return new Blob([wavBuffer], { type: 'audio/wav' });
  }, []);

  useEffect(() => {
    return () => {
      cleanupResources();
    };
  }, [cleanupResources]);

  return {
    connectionState,
    telemetry,
    telemetryHistory,
    error,
    startLiveDetection,
    stopLiveDetection,
    getLiveSessionBlob
  };
};
