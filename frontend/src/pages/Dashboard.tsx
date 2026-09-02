import React, { useState } from 'react';
import { 
  ShieldCheck, 
  Mic, 
  Activity, 
  MessageSquareWarning, 
  AlertOctagon,
  Clock,
  Loader2,
  Radio,
  FileAudio,
  FileText,
  Fingerprint,
  ShieldAlert,
  AlertTriangle,
  Info,
  Users,
  CheckCircle2
} from 'lucide-react';
import { api, type SessionResponse, type EvidenceResponse } from '../services/api';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { useLiveDetection } from '../hooks/useLiveDetection';

// ── Batch analysis result shape stored in state ───────────────────────────
interface BatchRiskResult {
  overall_risk_score: number;
  risk_level: string;
  contributing_signals: string[];
  confidence?: number;
  voice_integrity_score?: number;
  voice_label?: string;
  voice_confidence?: number;
  speaker_similarity_score?: number | null;
  transcript?: string;
}

interface BatchDecisionResult {
  decision: string;
  escalated?: boolean;
}

// ── Processing stage labels ───────────────────────────────────────────────
type AnalysisStage =
  | 'idle'
  | 'decoding'
  | 'voice'
  | 'speech'
  | 'risk'
  | 'policy'
  | 'done'
  | 'error';

const STAGE_LABELS: Record<AnalysisStage, string> = {
  idle: '',
  decoding: 'Decoding audio…',
  voice: 'Analyzing voice integrity…',
  speech: 'Transcribing speech…',
  risk: 'Calculating risk score…',
  policy: 'Evaluating policy…',
  done: 'Analysis complete.',
  error: 'Analysis failed.',
};

// ── Component ─────────────────────────────────────────────────────────────
const Dashboard: React.FC = () => {
  const [activeSession, setActiveSession] = useState<SessionResponse | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detectionMode, setDetectionMode] = useState<'batch' | 'live'>('batch');

  // Batch result state — stored separately so partial results can be shown
  const [analysisStage, setAnalysisStage] = useState<AnalysisStage>('idle');
  const [batchRisk, setBatchRisk] = useState<BatchRiskResult | null>(null);
  const [batchDecision, setBatchDecision] = useState<BatchDecisionResult | null>(null);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const [evidenceData, setEvidenceData] = useState<EvidenceResponse['data'] | null>(null);

  const {
    isRecording,
    recordingTime,
    audioBlob,
    error: recorderError,
    startRecording,
    stopRecording,
    clearRecording,
  } = useAudioRecorder();

  const {
    connectionState,
    telemetry,
    telemetryHistory,
    error: liveError,
    startLiveDetection,
    stopLiveDetection,
    getLiveSessionBlob,
  } = useLiveDetection();

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // ── Session management ──────────────────────────────────────────────────
  const handleStartSession = async () => {
    setIsInitializing(true);
    setError(null);
    setBatchRisk(null);
    setBatchDecision(null);
    setDecisionError(null);
    setEvidenceData(null);
    setAnalysisStage('idle');
    clearRecording();
    if (connectionState !== 'Disconnected') stopLiveDetection();
    try {
      const session = await api.createSession();
      setActiveSession(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to connect to VERA backend.');
    } finally {
      setIsInitializing(false);
    }
  };

  // ── Batch Analysis — sequential, with partial result display ─────────────
  const handleRunAnalysis = async () => {
    if (!activeSession || !audioBlob) return;

    // Reset prior results
    setBatchRisk(null);
    setBatchDecision(null);
    setDecisionError(null);
    setEvidenceData(null);
    setError(null);

    setAnalysisStage('decoding');

    try {
      // ── Step 1: Risk (contains voice integrity + transcript + risk score) ──
      setAnalysisStage('voice');
      let riskRes;
      try {
        riskRes = await api.analyzeRisk(activeSession.session_id, audioBlob);
      } catch (riskErr) {
        setError(riskErr instanceof Error ? riskErr.message : 'Risk analysis failed.');
        setAnalysisStage('error');
        return;
      }

      // Extract every useful field from the risk response
      const ra = riskRes.data.risk_analysis as {
        overall_risk_score: number;
        risk_level: string;
        contributing_signals: string[];
        confidence?: number;
        // These are nested inside risk_analysis if the backend passes them through
        voice_integrity_score?: number;
        voice_label?: string;
        voice_confidence?: number;
        speaker_similarity_score?: number | null;
      };
      const extractedRisk: BatchRiskResult = {
        overall_risk_score: ra.overall_risk_score,
        risk_level: ra.risk_level,
        contributing_signals: ra.contributing_signals ?? [],
        confidence: ra.confidence,
        voice_integrity_score: ra.voice_integrity_score,
        voice_label: ra.voice_label,
        voice_confidence: ra.voice_confidence,
        speaker_similarity_score: ra.speaker_similarity_score ?? null,
        transcript: riskRes.data.transcript ?? '',
      };

      // Immediately show risk — user sees partial results right now
      setBatchRisk(extractedRisk);
      setAnalysisStage('policy');

      // ── Step 2: Decision (uses same audio) ──────────────────────────────
      try {
        const decisionRes = await api.getDecision(activeSession.session_id, audioBlob);
        setBatchDecision(decisionRes.data.policy);
      } catch (decErr) {
        // Risk succeeded — preserve it. Show decision-specific error.
        setDecisionError(
          decErr instanceof Error ? decErr.message : 'Policy decision unavailable.'
        );
      }

      // ── Step 3: Evidence (best-effort, silent failure OK) ───────────────
      try {
        const evidenceRes = await api.generateEvidence(activeSession.session_id, audioBlob);
        if (evidenceRes?.data) setEvidenceData(evidenceRes.data);
      } catch {
        // Evidence failure is non-fatal — batch flow still succeeds
      }

      setAnalysisStage('done');
    } catch (unexpectedErr) {
      setError(
        unexpectedErr instanceof Error ? unexpectedErr.message : 'Unexpected analysis error.'
      );
      setAnalysisStage('error');
    }
  };

  // ── Live evidence ────────────────────────────────────────────────────────
  const handleGenerateLiveEvidence = async () => {
    if (!activeSession) return;
    const blob = getLiveSessionBlob();
    if (!blob) {
      setError('No audio recorded yet in Live mode.');
      return;
    }
    setAnalysisStage('voice');
    try {
      const evidenceRes = await api.generateEvidence(activeSession.session_id, blob);
      setEvidenceData(evidenceRes.data);
      setAnalysisStage('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evidence generation failed');
      setAnalysisStage('error');
    }
  };

  // ── End session ──────────────────────────────────────────────────────────
  const handleEndSession = () => {
    if (isRecording) stopRecording();
    if (connectionState !== 'Disconnected') stopLiveDetection();
    setActiveSession(null);
    setBatchRisk(null);
    setBatchDecision(null);
    setDecisionError(null);
    setEvidenceData(null);
    setError(null);
    setAnalysisStage('idle');
    clearRecording();
  };

  // ── Derived display values ────────────────────────────────────────────────
  const displayRiskData =
    detectionMode === 'live' && telemetry
      ? {
          overall_risk_score: telemetry.overall_risk_score,
          risk_level: telemetry.risk_level ?? 'unavailable',
          contributing_signals: telemetry.signals ?? [],
        }
      : batchRisk
      ? {
          overall_risk_score: batchRisk.overall_risk_score,
          risk_level: batchRisk.risk_level,
          contributing_signals: batchRisk.contributing_signals,
        }
      : null;

  const displayDecisionData =
    detectionMode === 'live' && telemetry
      ? { decision: telemetry.decision ?? 'unavailable' }
      : batchDecision;

  const isProcessing =
    analysisStage !== 'idle' && analysisStage !== 'done' && analysisStage !== 'error';

  const activeError = error || recorderError || liveError;
  const isMicActive =
    isRecording ||
    connectionState === 'Connecting' ||
    connectionState === 'Live' ||
    connectionState === 'Processing';

  // ── Derived voice integrity display ──────────────────────────────────────
  const voiceIntegrityDisplay = (() => {
    if (detectionMode === 'live') {
      if (telemetry?.voice_integrity_score != null)
        return `${(telemetry.voice_integrity_score * 100).toFixed(1)}%`;
      return 'Unavailable';
    }
    if (batchRisk?.voice_integrity_score != null)
      return `${(batchRisk.voice_integrity_score * 100).toFixed(1)}%`;
    if (batchRisk) return 'Analyzed'; // score not exposed in this version of the endpoint
    return 'Unavailable';
  })();

  const voiceLabelDisplay = (() => {
    if (detectionMode === 'live') return null;
    if (batchRisk?.voice_label) return batchRisk.voice_label;
    return null;
  })();

  const speakerDisplay = (() => {
    if (detectionMode === 'live') {
      if (telemetry?.speaker_similarity_score != null)
        return `${(telemetry.speaker_similarity_score * 100).toFixed(1)}%`;
      return 'Unavailable';
    }
    if (batchRisk?.speaker_similarity_score != null)
      return `${(batchRisk.speaker_similarity_score * 100).toFixed(1)}%`;
    return 'Unavailable';
  })();

  const transcriptDisplay = (() => {
    if (detectionMode === 'live') {
      return telemetry?.transcript ?? null;
    }
    return batchRisk?.transcript ?? null;
  })();

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-6 max-w-[1400px] mx-auto pb-12">

      {/* 1. SESSION CONTROL HEADER */}
      <div className="bg-vera-panel border border-vera-border rounded-xl shadow-lg p-5 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div
            className={`w-12 h-12 rounded-full flex items-center justify-center ${
              isMicActive
                ? 'bg-vera-danger/20 text-vera-danger border border-vera-danger/30 animate-pulse'
                : isProcessing
                ? 'bg-vera-accent/20 text-vera-accent border border-vera-accent/30 animate-pulse'
                : activeSession
                ? 'bg-vera-success/20 text-vera-success border border-vera-success/30'
                : 'bg-vera-border text-vera-textMuted'
            }`}
          >
            {isProcessing ? <Loader2 size={24} className="animate-spin" /> : <Mic size={24} />}
          </div>
          <div>
            <h2 className="text-lg font-bold text-vera-text tracking-wide">
              {isProcessing
                ? 'ANALYZING…'
                : activeSession
                ? 'LIVE MONITORING'
                : 'READY TO MONITOR'}
            </h2>
            <div className="flex flex-col text-sm text-vera-textMuted">
              {activeSession ? (
                <>
                  <span>Session: {activeSession.session_id}</span>
                  <span
                    className={`mt-0.5 text-xs ${
                      isProcessing ? 'text-vera-accent' : 'text-vera-success'
                    }`}
                  >
                    {isProcessing ? STAGE_LABELS[analysisStage] : 'Status: Active'}
                  </span>
                </>
              ) : (
                <span>No active session</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3 w-full md:w-auto">
          {activeError && (
            <div className="text-vera-danger text-sm max-w-sm mr-4 flex items-center">
              <AlertTriangle size={14} className="mr-1.5 flex-shrink-0" />
              <span className="line-clamp-2">{activeError}</span>
            </div>
          )}

          {!activeSession ? (
            <button
              onClick={handleStartSession}
              disabled={isInitializing}
              className="px-6 py-2 bg-vera-accent hover:bg-blue-600 disabled:opacity-50 text-white rounded-lg font-medium transition-colors shadow-lg flex items-center"
            >
              {isInitializing && <Loader2 className="animate-spin mr-2" size={16} />}
              {isInitializing ? 'Initializing…' : 'Initialize Session'}
            </button>
          ) : (
            <div className="flex items-center space-x-3">
              {/* Mode toggle — only when idle */}
              {!isMicActive && !audioBlob && !isProcessing && (
                <div className="flex bg-vera-dark p-1 rounded-lg border border-vera-border">
                  <button
                    onClick={() => setDetectionMode('batch')}
                    className={`px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center ${
                      detectionMode === 'batch' ? 'bg-vera-accent text-white' : 'text-vera-textMuted hover:text-white'
                    }`}
                  >
                    <FileAudio size={14} className="mr-1.5" /> Batch
                  </button>
                  <button
                    onClick={() => setDetectionMode('live')}
                    className={`px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center ${
                      detectionMode === 'live' ? 'bg-vera-success text-white' : 'text-vera-textMuted hover:text-white'
                    }`}
                  >
                    <Radio size={14} className="mr-1.5" /> Stream
                  </button>
                </div>
              )}

              {/* BATCH CONTROLS */}
              {detectionMode === 'batch' && (
                <div className="flex items-center space-x-2">
                  {!audioBlob && !isRecording && !isProcessing && (
                    <button
                      onClick={startRecording}
                      className="px-4 py-2 bg-vera-danger/20 hover:bg-vera-danger/30 text-vera-danger border border-vera-danger/50 rounded-lg font-medium flex items-center transition-colors"
                    >
                      <Mic className="mr-2" size={16} /> Record
                    </button>
                  )}
                  {isRecording && (
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-vera-danger bg-vera-danger/10 px-3 py-2 rounded-lg border border-vera-danger/30 flex items-center">
                        <div className="w-2 h-2 rounded-full bg-vera-danger animate-pulse mr-2" />
                        {formatTime(recordingTime)}
                      </span>
                      <button
                        onClick={stopRecording}
                        className="px-4 py-2 bg-vera-border hover:bg-vera-danger hover:text-white border border-vera-border hover:border-vera-danger rounded-lg transition-colors"
                      >
                        Stop
                      </button>
                    </div>
                  )}
                  {audioBlob && !isProcessing && (
                    <>
                      <button
                        onClick={handleRunAnalysis}
                        className="px-4 py-2 bg-vera-accent hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
                      >
                        {batchRisk ? 'Re-analyze' : 'Run Analysis'}
                      </button>
                      <button
                        onClick={() => { clearRecording(); setBatchRisk(null); setBatchDecision(null); setDecisionError(null); setEvidenceData(null); setAnalysisStage('idle'); }}
                        className="px-4 py-2 bg-vera-border hover:bg-gray-700 text-gray-300 rounded-lg transition-colors"
                      >
                        Clear
                      </button>
                    </>
                  )}
                  {isProcessing && (
                    <div className="text-vera-accent flex items-center px-4 font-medium text-sm">
                      <Loader2 className="animate-spin mr-2" size={16} />
                      {STAGE_LABELS[analysisStage]}
                    </div>
                  )}
                </div>
              )}

              {/* LIVE CONTROLS */}
              {detectionMode === 'live' && (
                <div className="flex items-center space-x-2">
                  {connectionState === 'Disconnected' && (
                    <>
                      <button
                        onClick={() => startLiveDetection(activeSession.session_id)}
                        className="px-4 py-2 bg-vera-success hover:bg-emerald-600 text-white rounded-lg font-medium flex items-center transition-colors"
                      >
                        <Radio className="mr-2" size={16} /> Connect Live WebSocket
                      </button>
                      {telemetryHistory.length > 0 && !evidenceData && (
                        <button
                          onClick={handleGenerateLiveEvidence}
                          disabled={isProcessing}
                          className="px-4 py-2 bg-purple-500 hover:bg-purple-600 disabled:opacity-50 text-white rounded-lg font-medium flex items-center transition-colors"
                        >
                          {isProcessing ? (
                            <Loader2 className="animate-spin mr-2" size={16} />
                          ) : (
                            <FileText className="mr-2" size={16} />
                          )}
                          Generate Final Evidence
                        </button>
                      )}
                    </>
                  )}
                  {(connectionState === 'Connecting' ||
                    connectionState === 'Live' ||
                    connectionState === 'Processing') && (
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-vera-success bg-vera-success/10 px-3 py-2 rounded-lg border border-vera-success/30 flex items-center text-sm">
                        <div className="w-2 h-2 rounded-full bg-vera-success animate-pulse mr-2" />
                        {connectionState}
                      </span>
                      <button
                        onClick={stopLiveDetection}
                        className="px-4 py-2 bg-vera-border hover:bg-vera-success hover:text-white border border-vera-border hover:border-vera-success rounded-lg transition-colors"
                      >
                        Disconnect
                      </button>
                    </div>
                  )}
                </div>
              )}

              <div className="w-px h-8 bg-vera-border mx-2" />
              <button
                onClick={handleEndSession}
                className="text-sm text-vera-textMuted hover:text-vera-danger transition-colors underline-offset-4 hover:underline"
              >
                End Session
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 2. MAIN ANALYSIS GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* LEFT COLUMN: Voice Integrity, Speaker Consistency, Transcript */}
        <div className="lg:col-span-2 space-y-6">

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* VOICE INTEGRITY */}
            <div className="bg-vera-panel border border-vera-border rounded-xl p-6 shadow-md relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <ShieldCheck size={64} className="text-vera-accent" />
              </div>
              <h3 className="text-sm font-semibold text-vera-textMuted uppercase tracking-wider mb-4 flex items-center">
                <Activity size={16} className="mr-2 text-vera-accent" /> Voice Integrity
              </h3>
              <div className="flex items-baseline space-x-3 mb-1">
                <span className="text-4xl font-bold text-vera-text">
                  {isProcessing && analysisStage === 'voice' ? (
                    <Loader2 size={32} className="animate-spin text-vera-accent" />
                  ) : (
                    voiceIntegrityDisplay
                  )}
                </span>
                {voiceLabelDisplay && (
                  <span
                    className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${
                      voiceLabelDisplay === 'genuine'
                        ? 'text-vera-success bg-vera-success/10 border border-vera-success/20'
                        : 'text-vera-danger bg-vera-danger/10 border border-vera-danger/20'
                    }`}
                  >
                    {voiceLabelDisplay}
                  </span>
                )}
              </div>
              <p className="text-xs text-gray-500 mt-2">Deepfake / AI synthesis detection</p>
            </div>

            {/* SPEAKER CONSISTENCY */}
            <div className="bg-vera-panel border border-vera-border rounded-xl p-6 shadow-md relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <Fingerprint size={64} className="text-vera-accent" />
              </div>
              <h3 className="text-sm font-semibold text-vera-textMuted uppercase tracking-wider mb-4 flex items-center">
                <Users size={16} className="mr-2 text-vera-accent" /> Speaker Consistency
              </h3>
              <div className="flex items-end space-x-3 mb-1">
                <span className="text-4xl font-bold text-vera-text">{speakerDisplay}</span>
              </div>
              <p className="text-xs text-gray-500 mt-2">Trusted voice profile matching</p>
            </div>
          </div>

          {/* TRANSCRIPT */}
          <div className="bg-vera-panel border border-vera-border rounded-xl p-6 shadow-md flex flex-col min-h-[200px]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-vera-textMuted uppercase tracking-wider flex items-center">
                <MessageSquareWarning size={16} className="mr-2 text-vera-accent" />
                {detectionMode === 'live' ? 'Live Transcript' : 'Transcript'}
              </h3>
              {connectionState === 'Processing' && (
                <span className="flex items-center text-xs text-vera-accent bg-vera-accent/10 px-2 py-1 rounded border border-vera-accent/20">
                  <div className="w-1.5 h-1.5 rounded-full bg-vera-accent animate-pulse mr-1.5" /> LIVE
                </span>
              )}
              {analysisStage === 'done' && transcriptDisplay && (
                <span className="flex items-center text-xs text-vera-success bg-vera-success/10 px-2 py-1 rounded border border-vera-success/20">
                  <CheckCircle2 size={12} className="mr-1" /> Transcribed
                </span>
              )}
            </div>
            <div className="flex-1 bg-vera-dark border border-vera-border rounded-lg p-5 font-mono text-sm leading-relaxed text-gray-300 overflow-y-auto min-h-[100px]">
              {isProcessing && (analysisStage === 'speech' || analysisStage === 'voice') ? (
                <span className="flex items-center text-vera-textMuted">
                  <Loader2 size={14} className="animate-spin mr-2" /> Transcribing…
                </span>
              ) : transcriptDisplay ? (
                `> ${transcriptDisplay}`
              ) : (
                <span className="text-gray-600 italic">No transcript available yet.</span>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Risk Overview, Decision, Signals */}
        <div className="space-y-6">

          {/* RISK OVERVIEW */}
          <div className="bg-vera-panel border border-vera-border rounded-xl p-6 shadow-lg relative overflow-hidden flex flex-col items-center justify-center text-center">
            <h3 className="text-sm font-semibold text-vera-textMuted uppercase tracking-wider mb-6 self-start w-full text-left">
              Risk Overview
            </h3>

            <div className="relative mb-6">
              <svg className="w-40 h-40 transform -rotate-90">
                <circle cx="80" cy="80" r="70" stroke="#232E48" strokeWidth="8" fill="none" />
                <circle
                  cx="80"
                  cy="80"
                  r="70"
                  stroke={
                    displayRiskData?.risk_level === 'low'
                      ? '#10B981'
                      : displayRiskData?.risk_level === 'medium'
                      ? '#F59E0B'
                      : displayRiskData?.risk_level === 'high'
                      ? '#EF4444'
                      : displayRiskData?.risk_level === 'critical'
                      ? '#991B1B'
                      : '#374151'
                  }
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray="440"
                  strokeDashoffset={
                    displayRiskData?.overall_risk_score !== undefined
                      ? 440 - 440 * displayRiskData.overall_risk_score
                      : 440
                  }
                  className="transition-all duration-1000 ease-out"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                {isProcessing ? (
                  <Loader2 size={32} className="animate-spin text-vera-accent" />
                ) : (
                  <>
                    <span className="text-3xl font-bold text-vera-text">
                      {displayRiskData?.overall_risk_score !== undefined
                        ? `${(displayRiskData.overall_risk_score * 100).toFixed(0)}%`
                        : '—'}
                    </span>
                    <span className="text-xs text-vera-textMuted uppercase mt-1">Score</span>
                  </>
                )}
              </div>
            </div>

            <div className="w-full bg-vera-dark border border-vera-border rounded-lg p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Assessed Risk Level</p>
              <p
                className={`text-xl font-bold tracking-wide uppercase ${
                  displayRiskData?.risk_level === 'low'
                    ? 'text-vera-success'
                    : displayRiskData?.risk_level === 'medium'
                    ? 'text-vera-warning'
                    : displayRiskData?.risk_level === 'high'
                    ? 'text-vera-danger'
                    : displayRiskData?.risk_level === 'critical'
                    ? 'text-red-600'
                    : 'text-gray-500'
                }`}
              >
                {isProcessing ? '…' : displayRiskData?.risk_level || 'Unavailable'}
              </p>
            </div>
          </div>

          {/* POLICY DECISION */}
          <div
            className={`border rounded-xl p-5 shadow-md flex items-start space-x-4 ${
              displayDecisionData?.decision === 'allow'
                ? 'bg-vera-success/10 border-vera-success/30'
                : displayDecisionData?.decision === 'warn'
                ? 'bg-vera-warning/10 border-vera-warning/30'
                : displayDecisionData?.decision === 'verify'
                ? 'bg-vera-danger/10 border-vera-danger/30'
                : displayDecisionData?.decision === 'block'
                ? 'bg-red-900/20 border-red-700/50'
                : 'bg-vera-panel border-vera-border'
            }`}
          >
            <div
              className={`p-3 rounded-full flex-shrink-0 ${
                displayDecisionData?.decision === 'allow'
                  ? 'bg-vera-success/20 text-vera-success'
                  : displayDecisionData?.decision === 'warn'
                  ? 'bg-vera-warning/20 text-vera-warning'
                  : displayDecisionData?.decision === 'verify'
                  ? 'bg-vera-danger/20 text-vera-danger'
                  : displayDecisionData?.decision === 'block'
                  ? 'bg-red-800/30 text-red-500'
                  : 'bg-vera-border text-vera-textMuted'
              }`}
            >
              <AlertOctagon size={24} />
            </div>
            <div>
              <h3 className="text-xs font-semibold text-vera-textMuted uppercase tracking-wider mb-1">
                Policy Decision
              </h3>
              <p
                className={`text-xl font-bold tracking-wide uppercase ${
                  displayDecisionData?.decision === 'allow'
                    ? 'text-vera-success'
                    : displayDecisionData?.decision === 'warn'
                    ? 'text-vera-warning'
                    : displayDecisionData?.decision === 'verify'
                    ? 'text-vera-danger'
                    : displayDecisionData?.decision === 'block'
                    ? 'text-red-500'
                    : 'text-gray-500'
                }`}
              >
                {isProcessing && analysisStage === 'policy'
                  ? '…'
                  : decisionError
                  ? 'Unavailable'
                  : displayDecisionData?.decision || 'Unavailable'}
              </p>
              <p className="text-sm mt-1 text-gray-300">
                {decisionError ? (
                  <span className="text-vera-warning text-xs">{decisionError}</span>
                ) : displayDecisionData?.decision === 'allow' ? (
                  'Conversation appears safe.'
                ) : displayDecisionData?.decision === 'warn' ? (
                  'Additional caution recommended.'
                ) : displayDecisionData?.decision === 'verify' ? (
                  'Identity verification required.'
                ) : displayDecisionData?.decision === 'block' ? (
                  'High-risk interaction blocked.'
                ) : (
                  'Awaiting backend analysis.'
                )}
              </p>
            </div>
          </div>

          {/* FRAUD / INTENT SIGNALS */}
          <div className="bg-vera-panel border border-vera-border rounded-xl p-6 shadow-md">
            <h3 className="text-sm font-semibold text-vera-textMuted uppercase tracking-wider mb-4 flex items-center">
              <ShieldAlert size={16} className="mr-2 text-vera-accent" /> Fraud &amp; Intent Signals
            </h3>
            <div className="flex flex-wrap gap-2">
              {displayRiskData?.contributing_signals && displayRiskData.contributing_signals.length > 0 ? (
                displayRiskData.contributing_signals.map((sig, i) => (
                  <span
                    key={i}
                    className="px-3 py-1.5 bg-vera-dark border border-vera-border rounded-md text-xs font-medium text-gray-300 capitalize"
                  >
                    {sig.replace(/_/g, ' ')}
                  </span>
                ))
              ) : (
                <span className="text-sm text-gray-500 italic">
                  {isProcessing ? 'Analyzing…' : 'No signals detected / Unavailable'}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 3. TIMELINE + EVIDENCE ROW */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* RISK TIMELINE (live telemetry) */}
        <div className="bg-vera-panel border border-vera-border rounded-xl p-6 shadow-md">
          <h3 className="text-sm font-semibold text-vera-textMuted uppercase tracking-wider flex items-center mb-4">
            <Clock className="mr-2 text-vera-accent" size={16} /> Risk Timeline
          </h3>
          <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
            {telemetryHistory.length > 0 ? (
              [...telemetryHistory].reverse().map((evt, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-vera-dark rounded-lg border border-vera-border flex items-center text-sm"
                >
                  <span className="text-gray-500 font-mono text-xs w-20">
                    {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ''}
                  </span>
                  <span className="flex-1 truncate px-4 text-gray-300">
                    {evt.transcript || '<silence>'}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase w-16 text-center ${
                      evt.risk_level === 'low'
                        ? 'text-vera-success bg-vera-success/10 border border-vera-success/20'
                        : evt.risk_level === 'medium'
                        ? 'text-vera-warning bg-vera-warning/10 border border-vera-warning/20'
                        : evt.risk_level === 'high'
                        ? 'text-vera-danger bg-vera-danger/10 border border-vera-danger/20'
                        : evt.risk_level === 'critical'
                        ? 'text-red-600 bg-red-900/20 border border-red-700/30'
                        : 'text-gray-500 bg-gray-800'
                    }`}
                  >
                    {evt.risk_level || 'N/A'}
                  </span>
                </div>
              ))
            ) : (
              <div className="p-8 text-center text-gray-600 border border-dashed border-vera-border rounded-lg text-sm">
                Telemetry timeline will populate during live monitoring.
              </div>
            )}
          </div>
        </div>

        {/* EVIDENCE PANEL */}
        <div className="bg-vera-panel border border-vera-border rounded-xl p-6 shadow-md flex flex-col">
          <h3 className="text-sm font-semibold text-vera-textMuted uppercase tracking-wider flex items-center mb-4">
            <FileText className="mr-2 text-vera-accent" size={16} /> Cryptographic Evidence
          </h3>
          {evidenceData?.evidence_record ? (
            <div className="flex-1 bg-vera-dark border border-vera-border rounded-lg p-4 font-mono text-xs overflow-hidden flex flex-col">
              <div className="flex flex-col mb-4 pb-3 border-b border-vera-border space-y-1">
                <span className="text-gray-500 uppercase tracking-wider text-[10px]">SHA-256 HASH</span>
                <span className="text-vera-success break-all">{evidenceData.hash}</span>
              </div>
              <div className="flex-1 overflow-y-auto">
                <pre className="text-gray-400">{JSON.stringify(evidenceData.evidence_record, null, 2)}</pre>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center text-gray-600 border border-dashed border-vera-border rounded-lg text-sm">
              {isProcessing
                ? 'Generating evidence…'
                : 'Evidence generation pending final analysis completion.'}
            </div>
          )}
        </div>
      </div>

      {/* 4. DEMO SCENARIOS GUIDE */}
      <div className="bg-vera-dark border border-vera-border rounded-xl p-5 shadow-sm mt-8">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center">
          <Info size={14} className="mr-2" /> Simulated Demo Guidance
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-300">
          <div className="p-4 bg-vera-panel rounded-lg border border-vera-border">
            <strong className="text-vera-success block mb-1">1. Genuine + Normal</strong>
            <p className="text-xs text-gray-400 mb-2">Say: "Hi, I'd like to check my account balance."</p>
            <div className="text-[10px] uppercase font-bold text-vera-success bg-vera-success/10 inline-block px-2 py-0.5 rounded">
              Exp: Low / Allow
            </div>
          </div>
          <div className="p-4 bg-vera-panel rounded-lg border border-vera-border">
            <strong className="text-vera-danger block mb-1">2. Cloned + Dangerous</strong>
            <p className="text-xs text-gray-400 mb-2">Say: "This is urgent, transfer $5000 immediately."</p>
            <div className="text-[10px] uppercase font-bold text-red-500 bg-vera-danger/10 inline-block px-2 py-0.5 rounded">
              Exp: Critical / Block
            </div>
          </div>
          <div className="p-4 bg-vera-panel rounded-lg border border-vera-border">
            <strong className="text-vera-warning block mb-1">3. Genuine + Dangerous</strong>
            <p className="text-xs text-gray-400 mb-2">Say: "Can you reset my password for me?"</p>
            <div className="text-[10px] uppercase font-bold text-vera-warning bg-vera-warning/10 inline-block px-2 py-0.5 rounded">
              Exp: High / Verify
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default Dashboard;
