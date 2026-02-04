
import React, { useState, useCallback, useRef } from 'react';
import { 
  Shield, 
  Upload, 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  ArrowRight,
  Database,
  BarChart3,
  Camera,
  Trash2,
  ScanSearch,
  Zap
} from 'lucide-react';
import { Snapshot, MatchResult, DetectionStatus, SystemStats } from './types';
import { StatCard } from './components/StatCard';
import { MatchCard } from './components/MatchCard';
import { analyzeTraffic } from './services/geminiService';

const App: React.FC = () => {
  const [status, setStatus] = useState<DetectionStatus>('idle');
  const [entrySnapshots, setEntrySnapshots] = useState<Snapshot[]>([]);
  const [exitSnapshots, setExitSnapshots] = useState<Snapshot[]>([]);
  const [results, setResults] = useState<MatchResult[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  const addLog = (msg: string) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 10));
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, type: 'entry' | 'exit') => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;

    // Explicitly cast to File[] to resolve potential 'unknown' type inference in Array.from and ensure Blob compatibility
    const files = Array.from(fileList) as File[];

    const newSnapshots: Snapshot[] = await Promise.all(
      files.map((file) => {
        return new Promise<Snapshot>((resolve) => {
          const reader = new FileReader();
          reader.onload = (event) => {
            resolve({
              id: Math.random().toString(36).substr(2, 9),
              timestamp: new Date().toISOString(),
              imageUrl: event.target?.result as string,
              detections: [],
              metadata: {}
            });
          };
          // reader.readAsDataURL expects a Blob; File inherits from Blob
          reader.readAsDataURL(file);
        });
      })
    );

    if (type === 'entry') setEntrySnapshots(prev => [...prev, ...newSnapshots]);
    else setExitSnapshots(prev => [...prev, ...newSnapshots]);
    
    addLog(`Uploaded ${newSnapshots.length} ${type} snapshots.`);
  };

  const runAnalysis = async () => {
    if (entrySnapshots.length === 0 || exitSnapshots.length === 0) {
      alert("Please upload both Entry and Exit images first.");
      return;
    }

    setStatus('processing');
    addLog("Initializing Computer Vision Pipeline...");
    
    try {
      const entryImages = entrySnapshots.map(s => s.imageUrl);
      const exitImages = exitSnapshots.map(s => s.imageUrl);
      
      addLog("Extracting vehicle and facial embeddings...");
      const analysisResults = await analyzeTraffic(entryImages, exitImages);
      
      setResults(analysisResults);
      setStatus('completed');
      addLog(`Analysis complete. Detected ${analysisResults.filter(r => r.status === 'MISMATCH').length} mismatches.`);
    } catch (error) {
      setStatus('error');
      addLog("Critical failure in AI Analysis pipeline.");
    }
  };

  const clearAll = () => {
    setEntrySnapshots([]);
    setExitSnapshots([]);
    setResults([]);
    setStatus('idle');
    setLogs([]);
  };

  const stats: SystemStats = {
    totalDetections: entrySnapshots.length + exitSnapshots.length,
    verifiedMatches: results.filter(r => r.status === 'VERIFIED').length,
    mismatchesDetected: results.filter(r => r.status === 'MISMATCH').length,
    averageProcessingTime: results.length > 0 ? 1.2 : 0
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Sidebar / Navigation Header */}
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Shield className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">SENTINEL</h1>
            <p className="text-[10px] text-indigo-400 font-bold tracking-widest uppercase">Vehicle Identity Verification</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            <div className={`w-2 h-2 rounded-full ${status === 'processing' ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'}`} />
            <span className="text-xs font-medium text-slate-300 capitalize">{status}</span>
          </div>
          <button 
            onClick={clearAll}
            className="p-2 text-slate-400 hover:text-rose-400 hover:bg-rose-400/10 rounded-lg transition-colors"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </header>

      <main className="flex-1 max-w-[1600px] mx-auto w-full p-6 lg:p-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Input & Controls */}
        <div className="lg:col-span-4 space-y-8">
          
          <section className="bg-slate-900/50 rounded-2xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 bg-slate-900 flex items-center gap-2">
              <Upload size={18} className="text-indigo-400" />
              <h2 className="font-bold text-sm tracking-wide uppercase text-slate-400">Data Acquisition</h2>
            </div>
            
            <div className="p-6 space-y-6">
              <div className="space-y-3">
                <label className="text-xs font-bold text-slate-500 uppercase flex items-center gap-2">
                  <ArrowRight size={14} className="text-emerald-500" /> Entry Camera Feed (Batch)
                </label>
                <div className="relative group">
                  <input 
                    type="file" 
                    multiple 
                    onChange={(e) => handleFileUpload(e, 'entry')}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div className="border-2 border-dashed border-slate-700 group-hover:border-indigo-500 group-hover:bg-indigo-500/5 rounded-xl p-8 transition-all flex flex-col items-center justify-center gap-3">
                    <Camera className="text-slate-600 group-hover:text-indigo-400 transition-colors" size={32} />
                    <p className="text-sm text-slate-400 text-center">
                      <span className="text-indigo-400 font-bold">Upload Entry Images</span><br/>
                      or drag and drop vehicle captures
                    </p>
                  </div>
                </div>
                {entrySnapshots.length > 0 && (
                  <p className="text-[10px] mono text-emerald-400 bg-emerald-400/5 p-2 rounded">
                    &gt; {entrySnapshots.length} files staged for analysis
                  </p>
                )}
              </div>

              <div className="space-y-3">
                <label className="text-xs font-bold text-slate-500 uppercase flex items-center gap-2">
                  <ArrowRight size={14} className="text-rose-500" /> Exit Camera Feed (Batch)
                </label>
                <div className="relative group">
                  <input 
                    type="file" 
                    multiple 
                    onChange={(e) => handleFileUpload(e, 'exit')}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div className="border-2 border-dashed border-slate-700 group-hover:border-indigo-500 group-hover:bg-indigo-500/5 rounded-xl p-8 transition-all flex flex-col items-center justify-center gap-3">
                    <Camera className="text-slate-600 group-hover:text-indigo-400 transition-colors" size={32} />
                    <p className="text-sm text-slate-400 text-center">
                      <span className="text-indigo-400 font-bold">Upload Exit Images</span><br/>
                      or drag and drop vehicle captures
                    </p>
                  </div>
                </div>
                {exitSnapshots.length > 0 && (
                  <p className="text-[10px] mono text-rose-400 bg-rose-400/5 p-2 rounded">
                    &gt; {exitSnapshots.length} files staged for analysis
                  </p>
                )}
              </div>

              <button 
                disabled={status === 'processing' || entrySnapshots.length === 0 || exitSnapshots.length === 0}
                onClick={runAnalysis}
                className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all transform active:scale-95 shadow-lg ${
                  status === 'processing' 
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-500/20'
                }`}
              >
                {status === 'processing' ? (
                  <>
                    <Zap className="animate-spin" size={20} />
                    Processing Deep Vision...
                  </>
                ) : (
                  <>
                    <ScanSearch size={20} />
                    Run Forensic Analysis
                  </>
                )}
              </button>
            </div>
          </section>

          {/* Real-time System Logs */}
          <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden h-64 flex flex-col">
            <div className="p-3 border-b border-slate-800 flex items-center gap-2 px-4">
              <Activity size={14} className="text-emerald-400" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Pipeline Output</span>
            </div>
            <div className="flex-1 p-4 mono text-[11px] overflow-y-auto space-y-2 bg-black/40">
              {logs.length > 0 ? logs.map((log, i) => (
                <div key={i} className={i === 0 ? 'text-indigo-400 font-bold' : 'text-slate-500'}>
                  {log}
                </div>
              )) : (
                <div className="text-slate-700 italic">Waiting for input signals...</div>
              )}
            </div>
          </section>

        </div>

        {/* Right Column: Statistics & Results */}
        <div className="lg:col-span-8 space-y-8">
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard 
              label="Total Captures" 
              value={stats.totalDetections} 
              icon={<Database size={20} />} 
            />
            <StatCard 
              label="Verified Identity" 
              value={stats.verifiedMatches} 
              icon={<CheckCircle2 size={20} />} 
              trend={{ value: 12, isUp: true }}
            />
            <StatCard 
              label="Theft Alerts" 
              value={stats.mismatchesDetected} 
              icon={<AlertTriangle size={20} />} 
              trend={{ value: 2, isUp: false }}
            />
            <StatCard 
              label="AI Inference Latency" 
              value={`${stats.averageProcessingTime}s`} 
              icon={<BarChart3 size={20} />} 
            />
          </div>

          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold flex items-center gap-3">
                Analysis Results 
                {results.length > 0 && (
                  <span className="text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded-full">{results.length} pairs processed</span>
                )}
              </h2>
            </div>

            {status === 'idle' && (
              <div className="flex flex-col items-center justify-center py-32 bg-slate-900/30 border border-dashed border-slate-800 rounded-3xl opacity-60">
                <Database size={48} className="text-slate-700 mb-4" />
                <p className="text-slate-500 font-medium">No active analysis session.</p>
                <p className="text-xs text-slate-600 mt-2">Upload entry and exit snapshots to begin forensics.</p>
              </div>
            )}

            {status === 'processing' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-pulse">
                {[1, 2].map(i => (
                  <div key={i} className="h-96 bg-slate-900/50 rounded-2xl border border-slate-800" />
                ))}
              </div>
            )}

            {status === 'completed' && results.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {results.map((match) => {
                  const entry = entrySnapshots.find(s => s.imageUrl.includes(match.entrySnapshotId)) || entrySnapshots[0];
                  const exit = exitSnapshots.find(s => s.imageUrl.includes(match.exitSnapshotId)) || exitSnapshots[0];
                  return (
                    <MatchCard 
                      key={match.id} 
                      match={match} 
                      entrySnapshot={entry} 
                      exitSnapshot={exit} 
                    />
                  );
                })}
              </div>
            )}
            
            {status === 'completed' && results.length === 0 && (
              <div className="flex flex-col items-center justify-center py-20 bg-rose-500/5 border border-rose-500/20 rounded-3xl">
                <AlertTriangle size={48} className="text-rose-500 mb-4" />
                <p className="text-rose-400 font-bold">Anomaly in AI Detection</p>
                <p className="text-xs text-rose-400/60 mt-2 max-w-sm text-center">The system was unable to correlate vehicles between entry and exit points. Verify image quality and alignment.</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer / Status Bar */}
      <footer className="bg-slate-950 border-t border-slate-900 py-3 px-6 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            <span className="text-[10px] mono text-slate-500 uppercase tracking-widest font-bold">Sentinel Engine v4.0.2</span>
          </div>
          <div className="hidden sm:flex items-center gap-2 text-[10px] text-slate-600 font-medium">
            <Activity size={12} />
            <span>HEARTBEAT: OK</span>
          </div>
        </div>
        <div className="text-[10px] text-slate-700 mono">
          &copy; 2024 AI SECURITY SYSTEMS • HIGH-PRECISION FORENSICS
        </div>
      </footer>
    </div>
  );
};

export default App;
