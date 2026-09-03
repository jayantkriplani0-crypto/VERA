import React from 'react';
import { FileCheck, Search } from 'lucide-react';

const Evidence: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Evidence Ledger</h2>
          <p className="text-vera-textMuted">Cryptographically verifiable session records.</p>
        </div>
        
        <div className="relative">
          <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-vera-textMuted" />
          <input 
            type="text" 
            placeholder="Search hash or session ID..." 
            className="pl-10 pr-4 py-2 bg-vera-dark border border-vera-border rounded-lg text-sm focus:outline-none focus:border-vera-accent text-vera-text w-64"
          />
        </div>
      </div>

      <div className="bg-[#151C2C]/80 backdrop-blur-md border border-[#232E48] rounded-xl shadow-lg p-8 flex flex-col items-center justify-center text-vera-textMuted min-h-[400px]">
        <FileCheck size={48} className="mb-4 opacity-50" />
        <h3 className="text-lg font-medium mb-1">No Evidence Records</h3>
        <p className="text-sm">Complete a session to generate a verifiable evidence record.</p>
      </div>
    </div>
  );
};

export default Evidence;
