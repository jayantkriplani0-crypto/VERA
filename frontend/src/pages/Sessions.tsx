import React, { useEffect, useState } from 'react';
import { Activity, Clock, RefreshCw, Eye, ShieldAlert, CheckCircle, AlertTriangle, XCircle, HelpCircle, FileCheck } from 'lucide-react';
import { api, type SessionResponse } from '../services/api';
import { useNavigate } from 'react-router-dom';

const Sessions: React.FC = () => {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // For modal/details view
  const [selectedSession, setSelectedSession] = useState<SessionResponse | null>(null);

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getSessions();
      setSessions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch sessions');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const getStatusBadge = (status: string) => {
    const s = status.toLowerCase();
    if (s === 'active') {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider text-vera-accent bg-vera-accent/10 border border-vera-accent/20">
          <Activity size={10} className="mr-1 animate-pulse" /> ACTIVE
        </span>
      );
    }
    if (s === 'completed') {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider text-vera-success bg-vera-success/10 border border-vera-success/20">
          <CheckCircle size={10} className="mr-1" /> COMPLETED
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider text-vera-danger bg-vera-danger/10 border border-vera-danger/20">
        <XCircle size={10} className="mr-1" /> ERROR
      </span>
    );
  };

  const getRiskBadge = (risk: string | undefined | null) => {
    if (!risk) return <span className="text-vera-textMuted text-xs">Unavailable</span>;
    const r = risk.toLowerCase();
    if (r === 'low') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-vera-success bg-vera-success/10 border border-vera-success/20">LOW</span>;
    }
    if (r === 'medium') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-vera-warning bg-vera-warning/10 border border-vera-warning/20">MEDIUM</span>;
    }
    if (r === 'high') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-vera-danger bg-vera-danger/10 border border-vera-danger/20">HIGH</span>;
    }
    if (r === 'critical') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-red-500 bg-red-900/20 border border-red-700/30">CRITICAL</span>;
    }
    return <span className="text-vera-textMuted text-xs">Unavailable</span>;
  };

  const getDecisionBadge = (decision: string | undefined | null) => {
    if (!decision) return <span className="text-vera-textMuted text-xs">Unavailable</span>;
    const d = decision.toLowerCase();
    if (d === 'allow') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-vera-success bg-vera-success/10 border border-vera-success/20">ALLOW</span>;
    }
    if (d === 'warn') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-vera-warning bg-vera-warning/10 border border-vera-warning/20">WARN</span>;
    }
    if (d === 'verify') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-vera-danger bg-vera-danger/10 border border-vera-danger/20">VERIFY</span>;
    }
    if (d === 'block') {
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase text-red-500 bg-red-900/20 border border-red-700/30">BLOCK</span>;
    }
    return <span className="text-vera-textMuted text-xs">Unavailable</span>;
  };

  const shortenUUID = (uuid: string) => {
    if (uuid.length <= 12) return uuid;
    return `${uuid.substring(0, 8)}...${uuid.substring(uuid.length - 4)}`;
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12 w-full">
      
      {/* Header */}
      <div className="bg-vera-panel border border-vera-border rounded-xl shadow-lg p-5 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 rounded-full bg-vera-accent/20 text-vera-accent border border-vera-accent/30 flex items-center justify-center">
            <Activity size={24} />
          </div>
          <div>
            <h2 className="text-lg font-bold text-vera-text tracking-wide">Session History</h2>
            <p className="text-sm text-vera-textMuted">Monitor and review previous voice analysis sessions.</p>
          </div>
        </div>
        
        <div className="flex-shrink-0 w-full md:w-auto">
          <button 
            onClick={fetchSessions}
            disabled={loading}
            className="px-6 py-2 bg-vera-dark hover:bg-vera-border disabled:opacity-50 border border-vera-border text-vera-text rounded-lg font-medium transition-colors shadow flex items-center justify-center w-full md:w-auto"
          >
            <RefreshCw size={16} className={`mr-2 ${loading ? 'animate-spin text-vera-accent' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-vera-danger/10 px-4 py-3 rounded-lg border border-vera-danger/30 text-sm text-vera-danger flex items-center">
          <AlertTriangle size={16} className="mr-2" />
          <span><strong>Error:</strong> {error}</span>
        </div>
      )}

      {/* Main Table Container */}
      <div className="bg-vera-panel border border-vera-border rounded-xl shadow-lg overflow-hidden flex flex-col">
        {loading && sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-16 text-vera-textMuted">
            <Activity size={48} className="mb-4 text-vera-accent/50 animate-pulse" />
            <h3 className="text-lg font-medium text-vera-text mb-1 tracking-wide">Loading Sessions...</h3>
          </div>
        ) : sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-16 text-vera-textMuted">
            <ShieldAlert size={48} className="mb-4 opacity-50" />
            <h3 className="text-lg font-medium text-vera-text mb-2 tracking-wide">No analysis sessions yet.</h3>
            <button 
              onClick={() => navigate('/')}
              className="mt-4 px-6 py-2 bg-vera-accent hover:bg-blue-600 text-white rounded-lg font-medium transition-colors shadow-lg"
            >
              Start a new analysis
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left border-collapse min-w-[700px]">
              <thead>
                <tr className="bg-vera-dark border-b border-vera-border text-xs text-vera-textMuted uppercase tracking-wider">
                  <th className="p-4 font-semibold whitespace-nowrap">Session</th>
                  <th className="p-4 font-semibold whitespace-nowrap">Created</th>
                  <th className="p-4 font-semibold whitespace-nowrap">Status</th>
                  <th className="p-4 font-semibold whitespace-nowrap">Risk</th>
                  <th className="p-4 font-semibold whitespace-nowrap">Decision</th>
                  <th className="p-4 font-semibold text-right whitespace-nowrap">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-vera-border bg-vera-panel">
                {sessions.map(session => (
                  <tr key={session.session_id} className="hover:bg-vera-dark/50 transition-colors group">
                    <td className="p-4 font-mono text-sm text-gray-300">
                      <span className="hidden sm:inline" title={session.session_id}>{shortenUUID(session.session_id)}</span>
                      <span className="sm:hidden" title={session.session_id}>{session.session_id.substring(0, 8)}</span>
                    </td>
                    <td className="p-4 text-sm text-gray-400 whitespace-nowrap flex items-center h-full">
                      <Clock size={14} className="mr-2 opacity-50 inline-block align-text-bottom" />
                      <span className="inline-block align-bottom leading-none pt-0.5">{new Date(session.created_at + 'Z').toLocaleString()}</span>
                    </td>
                    <td className="p-4">
                      {getStatusBadge(session.status)}
                    </td>
                    <td className="p-4">
                      {getRiskBadge(session.risk_level)}
                    </td>
                    <td className="p-4">
                      {getDecisionBadge(session.decision)}
                    </td>
                    <td className="p-4 text-right">
                      <button 
                        onClick={() => setSelectedSession(session)}
                        className="inline-flex items-center justify-center p-2 rounded-lg bg-vera-dark border border-vera-border text-vera-textMuted hover:text-vera-accent hover:border-vera-accent/50 transition-colors"
                        title="View Details"
                      >
                        <Eye size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Compact Session Detail Modal */}
      {selectedSession && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={() => setSelectedSession(null)}>
          <div className="bg-vera-panel border border-vera-border rounded-xl shadow-2xl max-w-md w-full overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-vera-border bg-vera-dark">
              <h3 className="text-sm font-semibold text-white flex items-center uppercase tracking-wider">
                <FileCheck size={16} className="mr-2 text-vera-accent" /> Session Details
              </h3>
              <button 
                onClick={() => setSelectedSession(null)}
                className="text-vera-textMuted hover:text-white transition-colors"
              >
                <XCircle size={20} />
              </button>
            </div>
            <div className="p-6 space-y-4">
              
              <div>
                <span className="text-[10px] uppercase text-vera-textMuted font-bold tracking-wider block mb-1">Session ID</span>
                <span className="font-mono text-sm text-gray-300 break-all">{selectedSession.session_id}</span>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-[10px] uppercase text-vera-textMuted font-bold tracking-wider block mb-1">Created At</span>
                  <span className="text-sm text-gray-300 flex items-center">
                    <Clock size={12} className="mr-1.5 opacity-70" />
                    {new Date(selectedSession.created_at + 'Z').toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] uppercase text-vera-textMuted font-bold tracking-wider block mb-2">Status</span>
                  {getStatusBadge(selectedSession.status)}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-vera-border">
                <div>
                  <span className="text-[10px] uppercase text-vera-textMuted font-bold tracking-wider block mb-2">Risk Level</span>
                  {getRiskBadge(selectedSession.risk_level)}
                </div>
                <div>
                  <span className="text-[10px] uppercase text-vera-textMuted font-bold tracking-wider block mb-2">Decision</span>
                  {getDecisionBadge(selectedSession.decision)}
                </div>
              </div>

              <div className="pt-4 border-t border-vera-border">
                <span className="text-[10px] uppercase text-vera-textMuted font-bold tracking-wider block mb-2">Additional Context</span>
                <div className="bg-vera-dark border border-vera-border rounded p-3 flex items-start text-xs text-gray-400">
                  <HelpCircle size={14} className="mr-2 text-vera-textMuted flex-shrink-0 mt-0.5" />
                  <p>Detailed evidence, transcripts, and speaker metrics are currently unavailable in this compact view. Navigate to Evidence search or query the backend directly for full artifacts.</p>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default Sessions;
