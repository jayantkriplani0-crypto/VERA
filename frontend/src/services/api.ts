export interface HealthResponse {
  status: string;
  database: string;
}

export interface SessionCreateRequest {
  caller_id?: string;
}

export interface SessionResponse {
  id: number;
  session_id: string;
  caller_id: string | null;
  created_at: string;
  status: string;
  risk_level?: string;
  decision?: string;
}

export interface RiskAnalysis {
  overall_risk_score: number;
  risk_level: string;
  contributing_signals: string[];
  confidence: number;
}

export interface RiskResponse {
  session_id: string;
  status: string;
  data: {
    filename: string;
    transcript: string;
    risk_analysis: RiskAnalysis;
  };
}

export interface DecisionResponse {
  session_id: string;
  status: string;
  data: {
    filename: string;
    policy: {
      decision: string;
      escalated: boolean;
    };
  };
}

export interface EvidenceResponse {
  session_id: string;
  status: string;
  data: {
    evidence_record: any;
    hash: string | null;
    algorithm: string;
  };
}

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  status: number;
  
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function fetchWithHandle(endpoint: string, options?: RequestInit) {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      ...options,
    });

    if (!response.ok) {
      throw new ApiError(response.status, `API Error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new Error(error instanceof Error ? error.message : 'Unknown network error');
  }
}

// Utility to generate a dummy WAV blob for testing endpoints that require an audio file
export function generateDummyWavBlob(): Blob {
  const sampleRate = 16000;
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = numChannels * (bitsPerSample / 8);
  const byteRate = sampleRate * blockAlign;
  const dataSize = sampleRate * 1 * blockAlign; // 1 second
  const chunkSize = 36 + dataSize;
  const buffer = new ArrayBuffer(8 + chunkSize);
  const view = new DataView(buffer);

  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(view, 0, 'RIFF');
  view.setUint32(4, chunkSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  return new Blob([buffer], { type: 'audio/wav' });
}

export const api = {
  checkHealth: (): Promise<HealthResponse> => 
    fetchWithHandle('/api/v1/health', {
      headers: { 'Content-Type': 'application/json' }
    }),

  createSession: (data: SessionCreateRequest = {}): Promise<SessionResponse> => 
    fetchWithHandle('/api/v1/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  getSession: (sessionId: string): Promise<SessionResponse> => 
    fetchWithHandle(`/api/v1/sessions/${sessionId}`, {
      headers: { 'Content-Type': 'application/json' }
    }),

  getSessions: (): Promise<SessionResponse[]> => 
    fetchWithHandle(`/api/v1/sessions`, {
      headers: { 'Content-Type': 'application/json' }
    }),

  analyzeRisk: (sessionId: string, audioBlob: Blob): Promise<RiskResponse> => {
    const formData = new FormData();
    // useAudioRecorder now always produces audio/wav blobs; use .wav filename.
    const filename = audioBlob.type.includes('webm') ? 'recording.webm' : 'recording.wav';
    formData.append('file', audioBlob, filename);
    return fetchWithHandle(`/api/v1/sessions/${sessionId}/risk`, {
      method: 'POST',
      body: formData,
    });
  },

  getDecision: (sessionId: string, audioBlob: Blob): Promise<DecisionResponse> => {
    const formData = new FormData();
    const filename = audioBlob.type.includes('webm') ? 'recording.webm' : 'recording.wav';
    formData.append('file', audioBlob, filename);
    return fetchWithHandle(`/api/v1/sessions/${sessionId}/decision`, {
      method: 'POST',
      body: formData,
    });
  },

  getEvidence: (sessionId: string): Promise<EvidenceResponse> =>
    fetchWithHandle(`/api/v1/sessions/${sessionId}/evidence`, {
      headers: { 'Content-Type': 'application/json' }
    }),

  generateEvidence: (sessionId: string, audioBlob: Blob): Promise<EvidenceResponse> => {
    const formData = new FormData();
    const filename = audioBlob.type.includes('webm') ? 'recording.webm' : 'recording.wav';
    formData.append('file', audioBlob, filename);
    return fetchWithHandle(`/api/v1/sessions/${sessionId}/evidence`, {
      method: 'POST',
      body: formData,
    });
  }
};
