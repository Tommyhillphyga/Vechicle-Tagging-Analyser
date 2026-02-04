
import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: {
    value: number;
    isUp: boolean;
  };
}

export const StatCard: React.FC<StatCardProps> = ({ label, value, icon, trend }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-400 text-sm font-medium">{label}</span>
        <div className="text-indigo-400 bg-indigo-400/10 p-2 rounded-lg">
          {icon}
        </div>
      </div>
      <div className="flex items-baseline space-x-2">
        <h3 className="text-2xl font-bold tracking-tight">{value}</h3>
        {trend && (
          <span className={`text-xs font-semibold ${trend.isUp ? 'text-emerald-400' : 'text-rose-400'}`}>
            {trend.isUp ? '+' : '-'}{trend.value}%
          </span>
        )}
      </div>
    </div>
  );
};
