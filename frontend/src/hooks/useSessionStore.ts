import { useState, useEffect } from 'react';

export interface SessionHistoryItem {
  session_id: string;
  created_at: string;
  status: string;
  risk_level?: string;
  decision?: string;
}

// Simple in-memory global state for demo purposes to avoid complex state management
// We only store the basic fields to satisfy Task 3.
let globalSessions: SessionHistoryItem[] = [];
let listeners: Array<() => void> = [];

const notifyListeners = () => {
  listeners.forEach(listener => listener());
};

export const addSessionToHistory = (session: SessionHistoryItem) => {
  globalSessions = [session, ...globalSessions];
  notifyListeners();
};

export const updateSessionInHistory = (session_id: string, updates: Partial<SessionHistoryItem>) => {
  globalSessions = globalSessions.map(s => 
    s.session_id === session_id ? { ...s, ...updates } : s
  );
  notifyListeners();
};

export const useSessionStore = () => {
  const [sessions, setSessions] = useState<SessionHistoryItem[]>(globalSessions);

  useEffect(() => {
    const listener = () => {
      setSessions([...globalSessions]);
    };
    listeners.push(listener);
    return () => {
      listeners = listeners.filter(l => l !== listener);
    };
  }, []);

  return {
    sessions,
  };
};
