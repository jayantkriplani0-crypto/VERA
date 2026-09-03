import React from 'react';
import { Users, Plus } from 'lucide-react';

const VoiceProfiles: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Trusted Voice Profiles</h2>
          <p className="text-vera-textMuted">Manage enrolled voices for speaker verification.</p>
        </div>
        <button className="flex items-center px-4 py-2 bg-vera-accent hover:bg-blue-600 text-white rounded-lg transition-colors">
          <Plus size={18} className="mr-2" />
          Enroll New Voice
        </button>
      </div>

      <div className="bg-[#151C2C]/80 backdrop-blur-md border border-[#232E48] rounded-xl shadow-lg p-8 flex flex-col items-center justify-center text-vera-textMuted min-h-[400px]">
        <Users size={48} className="mb-4 opacity-50" />
        <h3 className="text-lg font-medium mb-1">No Profiles Enrolled</h3>
        <p className="text-sm">Click "Enroll New Voice" to add a trusted speaker.</p>
      </div>
    </div>
  );
};

export default VoiceProfiles;
