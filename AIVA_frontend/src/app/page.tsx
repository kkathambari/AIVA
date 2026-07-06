"use client";

import { useState } from "react";
import { ThemeToggle } from "@/components/theme-toggle";
import { UploadSection } from "@/components/upload-section";
import { VivaPanel } from "@/components/viva-panel";
import { TutorPanel } from "@/components/tutor-panel";
import { AnalyticsDashboard } from "@/components/analytics-dashboard";
import { BrainCircuit, MessageSquare, BarChart3, Network, History, Sparkles, Database, FileText, BookOpen } from "lucide-react";

export default function Home() {
  const [kgData, setKgData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("viva");
  const [analytics, setAnalytics] = useState<any>({
    coverage: { coverage_percentage: 0, concept_breakdown: {} },
    weaknesses: { weaknesses: [] },
    readiness_prediction: { probability: 0, readiness_level: "Not Started", recommendation: "Upload document and start viva." },
    difficulty_history: [1]
  });
  const [sessionHistory, setSessionHistory] = useState<any[]>([]);

  // Callback when a turn is finished in VivaPanel
  const handleTurnComplete = (data: any) => {
    if (data.analytics) {
      setAnalytics(data.analytics);
    }
    if (data.evaluation) {
      setSessionHistory((prev) => [
        ...prev,
        {
          turn: prev.length + 1,
          concept: data.evaluation.concept,
          score: data.evaluation.score,
          feedback: data.evaluation.feedback,
          question: data.question,
          agent: data.agent
        }
      ]);
    }
  };

  const handleUploadComplete = (data: any) => {
    setKgData(data);
    // If analytics are returned from upload, populate them
    if (data.metadata) {
      // Sync initial KG metadata metrics if desired
    }
  };

  const activeDifficulty = analytics.difficulty_history 
    ? analytics.difficulty_history[analytics.difficulty_history.length - 1] 
    : 1;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#080d19] text-slate-900 dark:text-slate-100 transition-colors duration-300">
      
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-white/70 dark:bg-[#080d19]/80 border-b border-slate-250 dark:border-slate-800/80">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5 text-indigo-600 dark:text-indigo-400">
            <BrainCircuit size={28} className="animate-pulse" />
            <h1 className="text-lg font-black tracking-tight">
              AIVA <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/40 text-slate-500 dark:text-indigo-300 ml-1">Research Project</span>
            </h1>
          </div>
          
          {/* Tab Navigation */}
          <nav className="flex bg-slate-200/60 dark:bg-slate-900/50 p-1 rounded-xl border border-slate-350 dark:border-slate-800/60">
            {[
              { id: "learn", label: "Learning", icon: BookOpen },
              { id: "viva", label: "Examination", icon: MessageSquare },
              { id: "analytics", label: "Analytics", icon: BarChart3 },
              { id: "kg", label: "Knowledge Graph", icon: Network },
              { id: "history", label: "History Log", icon: History }
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    isActive
                      ? "bg-white dark:bg-slate-800 text-indigo-600 dark:text-white shadow-md shadow-slate-300/10 dark:shadow-none"
                      : "text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
                  }`}
                >
                  <Icon size={14} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
          
          <ThemeToggle />
        </div>
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        
        {/* Tab 0: Learning */}
        {activeTab === "learn" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-6">
              <UploadSection onUploadComplete={handleUploadComplete} />
              
              {/* Document Overview Metadata */}
              {kgData && (
                <div className="p-6 rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl space-y-4">
                  <h3 className="font-bold text-sm flex items-center gap-2 border-b border-white/10 pb-2">
                    <FileText size={16} className="text-indigo-400" />
                    Report Overview
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-xl bg-slate-100/40 dark:bg-slate-800/40 border border-slate-200/50 dark:border-slate-800">
                      <span className="text-[10px] uppercase font-bold text-slate-400">RAG Chunks</span>
                      <div className="text-lg font-black text-indigo-600 dark:text-indigo-400 mt-0.5">{kgData.chunks_indexed}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-100/40 dark:bg-slate-800/40 border border-slate-200/50 dark:border-slate-800">
                      <span className="text-[10px] uppercase font-bold text-slate-400">Entities</span>
                      <div className="text-lg font-black text-indigo-600 dark:text-indigo-400 mt-0.5">{kgData.nodes?.length || 0}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="lg:col-span-2">
              <TutorPanel />
            </div>
          </div>
        )}

        {/* Tab 1: Examination */}
        {activeTab === "viva" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-6">
              <UploadSection onUploadComplete={handleUploadComplete} />
              
              {/* Document Overview Metadata */}
              {kgData && (
                <div className="p-6 rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl space-y-4">
                  <h3 className="font-bold text-sm flex items-center gap-2 border-b border-white/10 pb-2">
                    <FileText size={16} className="text-indigo-400" />
                    Report Overview
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-xl bg-slate-100/40 dark:bg-slate-800/40 border border-slate-200/50 dark:border-slate-800">
                      <span className="text-[10px] uppercase font-bold text-slate-400">RAG Chunks</span>
                      <div className="text-lg font-black text-indigo-600 dark:text-indigo-400 mt-0.5">{kgData.chunks_indexed}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-100/40 dark:bg-slate-800/40 border border-slate-200/50 dark:border-slate-800">
                      <span className="text-[10px] uppercase font-bold text-slate-400">Entities</span>
                      <div className="text-lg font-black text-indigo-600 dark:text-indigo-400 mt-0.5">{kgData.nodes?.length || 0}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="lg:col-span-2">
              <VivaPanel onTurnComplete={handleTurnComplete} activeDifficulty={activeDifficulty} />
            </div>
          </div>
        )}

        {/* Tab 2: Interactive Analytics */}
        {activeTab === "analytics" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div>
                <h2 className="text-xl font-black">Viva Intelligence Analytics</h2>
                <p className="text-xs text-slate-400 mt-0.5">Adaptive Reinforcement Learning decisions and synthetic ML predictions.</p>
              </div>
            </div>
            <AnalyticsDashboard analytics={analytics} history={sessionHistory} />
          </div>
        )}

        {/* Tab 3: Knowledge Graph */}
        {activeTab === "kg" && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center border-b border-white/10 pb-4">
              <div>
                <h2 className="text-xl font-black">Interactive Knowledge Graph</h2>
                <p className="text-xs text-slate-400 mt-0.5">Semantic relationships and entities extracted using Gemini Pro.</p>
              </div>
              
              {/* Graph Metrics Card */}
              {kgData && kgData.metadata && (
                <div className="flex gap-4 p-3 rounded-xl bg-slate-100/40 dark:bg-slate-900/40 border border-slate-200/50 dark:border-slate-800/80 font-mono text-xs text-slate-600 dark:text-slate-350">
                  <div>Nodes: <strong className="text-indigo-500">{kgData.metadata.num_nodes}</strong></div>
                  <div>Edges: <strong className="text-indigo-500">{kgData.metadata.num_edges}</strong></div>
                  <div>Density: <strong className="text-indigo-500">{kgData.metadata.graph_density}</strong></div>
                  <div>Time: <strong className="text-indigo-500">{kgData.metadata.generation_time_sec}s</strong></div>
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-white/20 dark:border-white/10 shadow-xl overflow-hidden min-h-[500px] bg-slate-900 relative">
              {kgData ? (
                <iframe
                  src="http://127.0.0.1:8000/kg_visualization"
                  className="w-full h-[600px] border-none"
                  title="PyVis Knowledge Graph"
                />
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 dark:text-slate-500 text-sm">
                  <BrainCircuit size={48} className="mb-2 opacity-50 text-indigo-500 animate-pulse" />
                  <span>Upload a document in the "Examination" tab to render the Knowledge Graph</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 4: Performance History Log */}
        {activeTab === "history" && (
          <div className="space-y-6">
            <div className="border-b border-white/10 pb-4">
              <h2 className="text-xl font-black">Examination Log & Transcript</h2>
              <p className="text-xs text-slate-400 mt-0.5">Chronological record of evaluated concepts, scores, and feedback.</p>
            </div>

            {sessionHistory.length === 0 ? (
              <div className="p-12 text-center rounded-2xl bg-white/10 dark:bg-slate-900/40 border border-white/20 dark:border-white/10 text-slate-400 dark:text-slate-500 italic">
                No exam history logged yet. Complete turns in the viva workspace to see evaluations.
              </div>
            ) : (
              <div className="space-y-6">
                {sessionHistory.map((item) => (
                  <div key={item.turn} className="p-6 rounded-2xl bg-white/15 dark:bg-slate-900/30 border border-white/20 dark:border-white/10 shadow-lg space-y-4">
                    <div className="flex justify-between items-center pb-2 border-b border-white/5">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-xs bg-indigo-500/10 text-indigo-500 px-2.5 py-0.5 rounded-full uppercase tracking-wider">
                          Turn {item.turn}
                        </span>
                        <span className="text-xs font-bold text-slate-400">
                          Examiner: <strong className="text-indigo-400">{item.agent}</strong>
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-2 text-xs font-mono">
                        <span className="font-semibold">Concept: <strong className="text-indigo-400">{item.concept}</strong></span>
                        <span className={`px-2.5 py-0.5 rounded font-black ${
                          item.score >= 0.7 
                            ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-500/20' 
                            : item.score >= 0.4 
                              ? 'bg-amber-950/60 text-amber-400 border border-amber-500/20' 
                              : 'bg-red-950/60 text-red-400 border border-red-500/20'
                        }`}>
                          Score: {Math.round(item.score * 100)}%
                        </span>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-xs">
                        <span className="text-slate-400 font-bold uppercase block mb-1">Question Asked:</span>
                        <p className="p-3 rounded-xl bg-slate-950/30 text-slate-200 leading-relaxed border border-white/5">{item.question}</p>
                      </div>
                      
                      <div className="text-xs">
                        <span className="text-slate-400 font-bold uppercase block mb-1">Evaluator Feedback:</span>
                        <p className="p-3 rounded-xl bg-slate-950/20 text-slate-350 italic leading-relaxed border border-white/5">{item.feedback}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}
