
import React from 'react';
import { MatchResult, Snapshot } from '../types';
import { ShieldCheck, ShieldAlert, Clock, Car, User } from 'lucide-react';

interface MatchCardProps {
  match: MatchResult;
  entrySnapshot: Snapshot;
  exitSnapshot: Snapshot;
}

export const MatchCard: React.FC<MatchCardProps> = ({ match, entrySnapshot, exitSnapshot }) => {
  const isMismatch = match.status === 'MISMATCH';

  return (
    <div className={`relative overflow-hidden bg-slate-900 border ${isMismatch ? 'border-rose-500/50' : 'border-slate-800'} rounded-2xl shadow-xl transition-all hover:border-indigo-500/30`}>
      {isMismatch && (
        <div className="absolute top-0 left-0 w-full h-1 bg-rose-500 animate-pulse" />
      )}
      
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-full ${isMismatch ? 'bg-rose-500/10 text-rose-500' : 'bg-emerald-500/10 text-emerald-500'}`}>
              {isMismatch ? <ShieldAlert size={20} /> : <ShieldCheck size={20} />}
            </div>
            <div>
              <h4 className="font-bold text-slate-100">{isMismatch ? 'Security Breach Detected' : 'Identity Verified'}</h4>
              <p className="text-xs text-slate-400 mono">ID: {match.id.toUpperCase()}</p>
            </div>
          </div>
          <div className={`px-3 py-1 rounded-full text-[10px] font-bold tracking-wider uppercase ${isMismatch ? 'bg-rose-500 text-white' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}`}>
            {match.status}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="space-y-2">
            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Entry Snapshot</p>
            <div className="relative aspect-video rounded-lg overflow-hidden border border-slate-800 bg-black">
              <img src={entrySnapshot.imageUrl} alt="Entry" className="w-full h-full object-cover" />
              <div className="absolute bottom-2 left-2 px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-md text-[10px] mono text-white">
                IN: {new Date(entrySnapshot.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Exit Snapshot</p>
            <div className="relative aspect-video rounded-lg overflow-hidden border border-slate-800 bg-black">
              <img src={exitSnapshot.imageUrl} alt="Exit" className="w-full h-full object-cover" />
              <div className="absolute bottom-2 left-2 px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-md text-[10px] mono text-white">
                OUT: {new Date(exitSnapshot.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
            <div className="flex items-center gap-2 mb-1 text-slate-400">
              <Car size={14} />
              <span className="text-[10px] font-bold uppercase">Vehicle</span>
            </div>
            <p className="text-lg font-bold text-indigo-400">{(match.vehicleSimilarity * 100).toFixed(0)}%</p>
          </div>
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
            <div className="flex items-center gap-2 mb-1 text-slate-400">
              <User size={14} />
              <span className="text-[10px] font-bold uppercase">Driver</span>
            </div>
            <p className={`text-lg font-bold ${match.driverSimilarity < 0.6 ? 'text-rose-400' : 'text-emerald-400'}`}>
              {(match.driverSimilarity * 100).toFixed(0)}%
            </p>
          </div>
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
            <div className="flex items-center gap-2 mb-1 text-slate-400">
              <ShieldCheck size={14} />
              <span className="text-[10px] font-bold uppercase">Overall</span>
            </div>
            <p className="text-lg font-bold text-slate-200">{(match.overallScore * 100).toFixed(0)}%</p>
          </div>
        </div>

        {match.reason && (
          <div className={`mt-4 p-3 rounded-lg text-xs leading-relaxed ${isMismatch ? 'bg-rose-500/10 text-rose-300 border border-rose-500/20' : 'bg-slate-950 text-slate-400'}`}>
            <span className="font-bold mr-2 uppercase text-[10px] text-slate-500 tracking-wider">Analysis:</span>
            {match.reason}
          </div>
        )}
      </div>
    </div>
  );
};
