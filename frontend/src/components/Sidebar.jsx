import React from 'react';
import {
  Home,
  Database,
  Sliders,
  Layers,
  Grid,
  BarChart3,
  Stethoscope,
  BookOpen
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'home', label: 'Home / Overview', icon: Home, badge: 'Overview' },
  { id: 'dataset', label: 'Dataset Explorer', icon: Database, badge: 'RSNA' },
  { id: 'preprocessing', label: 'Preprocessing Lab', icon: Sliders, badge: 'P0–P3' },
  { id: 'models', label: 'CNN Model Lab', icon: Layers, badge: '3 CNNs' },
  { id: 'matrix', label: 'Experiment Matrix', icon: Grid, badge: '12 Exps' },
  { id: 'analysis', label: 'Comparative Analysis', icon: BarChart3, badge: 'Results' },
  { id: 'predict', label: 'Prediction Demo', icon: Stethoscope, badge: 'Inference' },
  { id: 'docs', label: 'Research Docs', icon: BookOpen, badge: 'Methods' },
];

export default function Sidebar({ activePage, onSelectPage }) {
  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col flex-shrink-0 min-h-[calc(100vh-61px)]">
      <div className="p-4 flex-1">
        <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 px-3 mb-2">
          Academic Research Suite
        </div>
        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectPage(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-brand-600 text-white shadow-md shadow-brand-500/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <div className="flex items-center space-x-2.5">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded ${
                    isActive
                      ? 'bg-brand-700 text-brand-100'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {item.badge}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="p-4 border-t border-slate-800/80 text-[11px] text-slate-500">
        <div className="flex items-center justify-between mb-1">
          <span>Framework</span>
          <span className="font-mono text-slate-400">PyTorch 2.x</span>
        </div>
        <div className="flex items-center justify-between">
          <span>Dataset</span>
          <span className="font-mono text-slate-400">RSNA Pediatric</span>
        </div>
      </div>
    </aside>
  );
}
