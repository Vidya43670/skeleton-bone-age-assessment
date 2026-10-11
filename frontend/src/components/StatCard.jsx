import React from 'react';

export default function StatCard({ title, value, subtitle, icon: Icon, color = 'brand' }) {
  const colorMap = {
    brand: 'text-brand-400 bg-brand-500/10 border-brand-500/20',
    emerald: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    rose: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 relative overflow-hidden shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs font-medium text-slate-400 block mb-1">{title}</span>
          <span className="text-2xl font-bold text-white tracking-tight">{value}</span>
          {subtitle && (
            <span className="text-xs text-slate-500 block mt-1">{subtitle}</span>
          )}
        </div>
        {Icon && (
          <div className={`p-3 rounded-xl border ${colorMap[color] || colorMap.brand}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
    </div>
  );
}
