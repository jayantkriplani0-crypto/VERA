import React from 'react';
import { Routes, Route } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import Sessions from './pages/Sessions';
import VoiceProfiles from './pages/VoiceProfiles';
import Evidence from './pages/Evidence';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="sessions" element={<Sessions />} />
        <Route path="voice-profiles" element={<VoiceProfiles />} />
        <Route path="evidence" element={<Evidence />} />
      </Route>
    </Routes>
  );
};

export default App;
