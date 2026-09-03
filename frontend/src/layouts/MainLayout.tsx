import React, { useEffect, useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { 
  ShieldAlert, 
  LayoutDashboard, 
  Activity, 
  Users, 
  FileCheck,
  Server
} from 'lucide-react';
import { api } from '../services/api';

const MainLayout: React.FC = () => {
  const [isApiConnected, setIsApiConnected] = useState<boolean | null>(null);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        await api.checkHealth();
        setIsApiConnected(true);
      } catch (err) {
        setIsApiConnected(false);
      }
    };
    
    checkStatus();
    // Re-check periodically every 30 seconds
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen bg-vera-darker text-vera-text font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-[260px] bg-vera-dark border-r border-vera-border flex flex-col shadow-2xl z-20">
        <div className="h-16 flex items-center px-6 border-b border-vera-border bg-vera-darker/50">
          <ShieldAlert className="w-6 h-6 text-vera-accent mr-3" />
          <div>
            <span className="font-bold text-lg tracking-wide text-white block leading-tight">VERA</span>
            <span className="text-[10px] text-vera-textMuted uppercase tracking-widest block">SIH 2026</span>
          </div>
        </div>
        
        <nav className="flex-1 py-6 px-4 space-y-2">
          <NavItem to="/" icon={<LayoutDashboard size={18} />} label="Dashboard" exact />
          <NavItem to="/sessions" icon={<Activity size={18} />} label="Sessions" />
          <NavItem to="/voice-profiles" icon={<Users size={18} />} label="Voice Profiles" />
          <NavItem to="/evidence" icon={<FileCheck size={18} />} label="Evidence" />
        </nav>

        <div className="p-4 border-t border-vera-border bg-vera-darker/30">
          <div className="flex items-center text-xs text-vera-textMuted font-medium">
            <Server size={14} className="mr-2" />
            <span>Backend API:</span>
            <span className={`ml-auto px-2 py-0.5 rounded-full text-[10px] uppercase tracking-wider ${isApiConnected === null ? 'bg-vera-warning/10 text-vera-warning border border-vera-warning/20' : isApiConnected ? 'bg-vera-success/10 text-vera-success border border-vera-success/20' : 'bg-vera-danger/10 text-vera-danger border border-vera-danger/20'}`}>
              {isApiConnected === null ? 'Checking' : isApiConnected ? 'Online' : 'Offline'}
            </span>
          </div>
          {isApiConnected === false && (
            <div className="mt-2 text-[10px] text-vera-danger leading-tight">
              Backend unavailable — start the FastAPI server on port 8000.
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 bg-vera-darker">
        <header className="h-16 bg-vera-dark/80 backdrop-blur-md border-b border-vera-border flex items-center justify-between px-6 z-10 sticky top-0">
          <div className="flex items-center space-x-3">
            <h1 className="text-base font-semibold text-white tracking-wide">Voice Evidence & Risk Authentication</h1>
            <span className="w-1 h-1 rounded-full bg-vera-textMuted"></span>
            <span className="text-sm text-vera-textMuted">Real-Time Voice Security</span>
          </div>
          <div className="flex items-center space-x-4">
            <div className="w-8 h-8 rounded-full bg-vera-panel border border-vera-border flex items-center justify-center text-vera-textMuted font-bold text-xs shadow-inner">
              OP
            </div>
          </div>
        </header>
        
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;

interface NavItemProps {
  to: string;
  icon: React.ReactNode;
  label: string;
  exact?: boolean;
}

const NavItem: React.FC<NavItemProps> = ({ to, icon, label, exact }) => (
  <NavLink
    to={to}
    end={exact}
    className={({ isActive }) =>
      `flex items-center px-3 py-2.5 rounded-lg transition-colors duration-200 ${
        isActive 
          ? 'bg-vera-accent/10 text-vera-accent border border-vera-accent/20 font-medium' 
          : 'text-vera-textMuted hover:bg-vera-panel hover:text-vera-text'
      }`
    }
  >
    <span className="mr-3">{icon}</span>
    {label}
  </NavLink>
);
