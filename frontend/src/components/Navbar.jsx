import React from 'react';
import { Activity, Cpu, Database, RefreshCw } from 'lucide-react';

export default function Navbar({ health, loading, onRefresh }) {
  const isHealthy = health?.status === 'healthy';
  const device = health?.hardware?.device || 'Detecting...';
  const isCuda = health?.hardware?.cuda_available;
  const datasetFound = health?.dataset?.present;

  return (
    <header className="bg-slate-900/80 backdrop-blur border-b border-slate-800 sticky top-0 z-30 px-6 py-3.5 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg shadow-brand-500/20 text-lg">
          BA
        </div>
        <div>
          <h1 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
            Pediatric Bone Age Assessment
            <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 border border-brand-500/30">
              RSNA Challenge
            </span>
          </h1>
          <p className="text-xs text-slate-400">
            Advanced Image Preprocessing & CNN Classification Research Platform
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3 text-xs">
        {/* Hardware Status */}
        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-brand-400" />
          <span className="text-slate-400">Device:</span>
          <span className={`font-semibold uppercase ${isCuda ? 'text-emerald-400' : 'text-slate-200'}`}>
            {device} {isCuda ? '(GPU)' : '(CPU)'}
          </span>
        </div>

        {/* Dataset Status */}
        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <Database className="w-3.5 h-3.5 text-purple-400" />
          <span className="text-slate-400">Dataset:</span>
          <span className={`font-medium ${datasetFound ? 'text-emerald-400' : 'text-amber-400'}`}>
            {datasetFound ? 'Detected' : 'Pending Ingestion'}
          </span>
        </div>

        {/* Backend API Health Status */}
        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
          <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-slate-300 font-medium">
            {isHealthy ? 'Backend Active' : 'Disconnected'}
          </span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          disabled={loading}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors border border-slate-700/60"
          title="Refresh system diagnostics"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-brand-400' : ''}`} />
        </button>
      </div>
    </header>
  );
}
