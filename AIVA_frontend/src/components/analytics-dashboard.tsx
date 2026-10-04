"use client";

import { AlertTriangle, Target, Activity, ShieldAlert, Award, ChevronRight, Repeat, Sparkles, BookOpen } from "lucide-react";

interface AnalyticsDashboardProps {
  analytics: any;
  history: any[];
}

export function AnalyticsDashboard({ analytics, history }: AnalyticsDashboardProps) {
  if (!analytics) return null;

  // Extract variables
  const coveragePercent = analytics.coverage?.coverage_percentage || 0;
  const coverageMap = analytics.coverage?.coverage_map || {};
  const conceptBreakdown = analytics.coverage?.concept_breakdown || {};
  
  const weaknesses = analytics.weaknesses?.weaknesses || [];
  
  const readiness = analytics.readiness_prediction || {
    probability: 0,
    readiness_level: "Not Started",
    recommendation: "Upload document and start viva."
  };
  
  const difficultyHistory = analytics.difficulty_history || [1];
  const fluency = analytics.fluency || null;

  // SVG dimensions for the difficulty timeline
  const svgWidth = 400;
  const svgHeight = 150;
  const padding = 20;
  
  // Mappings for difficulty name
  const difficultyMapping: { [key: number]: string } = {
    0: "Fundamentals",
    1: "Easy",
    2: "Medium",
    3: "Hard",
    4: "Research Level"
  };

  // Generate SVG path for difficulty timeline
  const generateSvgPath = () => {
    if (difficultyHistory.length <= 1) return "";
    
    const xStep = (svgWidth - padding * 2) / (difficultyHistory.length - 1);
    const yRange = svgHeight - padding * 2;
    
    return difficultyHistory.map((diff: number, i: number) => {
      const x = padding + i * xStep;
      // Invert Y so higher difficulty is at the top
      const y = padding + yRange - (diff / 4.0) * yRange;
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(" ");
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      
      {/* Column 1: Gauges & Timeline */}
      <div className="space-y-6">
        
        {/* ML Readiness Gauge */}
        <div className="p-6 rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl">
          <h3 className="text-base font-bold flex items-center gap-2 mb-4 text-slate-800 dark:text-slate-200">
            <Activity className="text-indigo-500" size={18} />
            ML Readiness Prediction
          </h3>
          <div className="flex items-center gap-6">
            {/* Speedometer SVG */}
            <div className="relative w-24 h-24 flex items-center justify-center shrink-0">
              <svg className="w-full h-full transform -rotate-90">
                {/* Background Track */}
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  fill="transparent"
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth="8"
                />
                {/* Progress Arc */}
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  fill="transparent"
                  stroke="url(#gradient)"
                  strokeWidth="8"
                  strokeDasharray={251.2}
                  strokeDashoffset={251.2 - (251.2 * readiness.probability) / 100}
                  className="transition-all duration-1000 ease-out"
                />
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#10b981" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute flex flex-col items-center justify-center">
                <span className="text-xl font-black text-slate-800 dark:text-slate-100">{Math.round(readiness.probability)}%</span>
                <span className="text-[9px] uppercase tracking-wider font-bold opacity-60">Pass Prob</span>
              </div>
            </div>
            
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <Award size={16} className="text-emerald-500" />
                <span className="font-bold text-sm text-slate-700 dark:text-slate-300">
                  {readiness.readiness_level}
                </span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-normal">
                {readiness.recommendation}
              </p>
              <div className="mt-2 text-[10px] uppercase font-bold tracking-wider text-slate-400">
                Confidence Interval: {(readiness.confidence_score * 100).toFixed(0)}% (Logistic Regression)
              </div>
              {readiness.fluency_penalty_pct > 0 && (
                <div className="mt-2 text-[11px] font-semibold text-rose-500 dark:text-rose-400 flex items-center gap-1.5 bg-rose-500/10 border border-rose-500/20 px-2.5 py-1 rounded-lg">
                  <Repeat size={13} className="shrink-0" />
                  <span>Fluency & Repetition Deduction: -{readiness.fluency_penalty_pct}% on overall pass probability</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Fluency Score Card */}
        {fluency && (
          <div className="p-6 rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-base font-bold flex items-center gap-2 text-slate-800 dark:text-slate-200">
                <Activity className="text-indigo-500" size={18} />
                Communication & Fluency
              </h3>
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                fluency.fluency_score >= 85 
                  ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20" 
                  : fluency.fluency_score >= 70 
                    ? "bg-amber-500/10 text-amber-500 border border-amber-500/20" 
                    : "bg-rose-500/10 text-rose-500 border border-rose-500/20"
              }`}>
                {fluency.fluency_score >= 85 ? "Articulate" : fluency.fluency_score >= 70 ? "Moderate Hesitation" : "Needs Revision"}
              </span>
            </div>
            
            <div className="flex items-start gap-4">
              <div className={`w-16 h-16 rounded-full flex flex-col items-center justify-center font-bold text-xl border-4 shrink-0 ${
                fluency.fluency_score >= 85
                  ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
                  : fluency.fluency_score >= 70
                    ? "bg-amber-500/10 text-amber-500 border-amber-500/20"
                    : "bg-rose-500/10 text-rose-500 border-rose-500/20"
              }`}>
                <span>{Math.round(fluency.fluency_score)}%</span>
                <span className="text-[8px] uppercase tracking-wider font-semibold opacity-75">Fluency</span>
              </div>
              
              <div className="flex-1 space-y-2">
                <p className="text-xs text-slate-600 dark:text-slate-300 font-medium leading-relaxed">
                  {fluency.feedback}
                </p>

                {/* Detected Fillers */}
                {Object.keys(fluency.filler_words_used || {}).length > 0 && (
                  <div className="text-xs text-slate-500 flex flex-wrap gap-1.5 items-center">
                    <span className="font-semibold text-slate-600 dark:text-slate-400 text-[11px]">Fillers:</span>
                    {Object.entries(fluency.filler_words_used).map(([word, count]: any) => (
                      <span key={word} className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 px-1.5 py-0.5 rounded text-[11px] font-mono">
                        "{word}" ({count})
                      </span>
                    ))}
                  </div>
                )}

                {/* Detected Repetitions */}
                {Object.keys(fluency.repetitive_words_used || {}).length > 0 && (
                  <div className="text-xs text-slate-500 flex flex-wrap gap-1.5 items-center">
                    <span className="font-semibold text-rose-600 dark:text-rose-400 text-[11px]">Repetitions:</span>
                    {Object.entries(fluency.repetitive_words_used).map(([word, count]: any) => (
                      <span key={word} className="bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20 px-1.5 py-0.5 rounded text-[11px] font-mono">
                        "{word}" ({count})
                      </span>
                    ))}
                  </div>
                )}

                {/* Repetition Penalty Tag */}
                {fluency.repetition_penalty > 0 && (
                  <div className="text-[11px] font-semibold text-rose-500 dark:text-rose-400 flex items-center gap-1">
                    <span>⚠️ -{fluency.repetition_penalty}% Fluency Penalty applied for repetitive vocabulary</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* RL Difficulty Timeline */}
        <div className="p-6 rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl">
          <h3 className="text-base font-bold flex items-center gap-2 mb-4 text-slate-800 dark:text-slate-200">
            <Target className="text-indigo-500" size={18} />
            RL Difficulty Timeline
          </h3>
          
          {difficultyHistory.length <= 1 ? (
            <div className="h-[150px] flex items-center justify-center text-xs text-slate-400 dark:text-slate-500 italic">
              Answer questions to plot the RL adaptive difficulty timeline.
            </div>
          ) : (
            <div className="flex flex-col items-center">
              <svg width="100%" height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="overflow-visible">
                {/* Horizontal Guide Lines */}
                {[0, 1, 2, 3, 4].map((gridY) => {
                  const y = padding + (svgHeight - padding * 2) - (gridY / 4.0) * (svgHeight - padding * 2);
                  return (
                    <g key={gridY}>
                      <line
                        x1={padding}
                        y1={y}
                        x2={svgWidth - padding}
                        y2={y}
                        stroke="rgba(255,255,255,0.05)"
                        strokeWidth="1"
                        strokeDasharray="4,4"
                      />
                      <text
                        x={padding - 5}
                        y={y + 3}
                        fill="rgba(255,255,255,0.3)"
                        fontSize="8"
                        textAnchor="end"
                        className="font-mono"
                      >
                        {gridY}
                      </text>
                    </g>
                  );
                })}
                
                {/* Graph Path */}
                <path
                  d={generateSvgPath()}
                  fill="none"
                  stroke="#6366f1"
                  strokeWidth="3"
                  className="transition-all duration-500"
                />
                
                {/* Dots on nodes */}
                {difficultyHistory.map((diff: number, i: number) => {
                  const xStep = (svgWidth - padding * 2) / (difficultyHistory.length - 1);
                  const x = padding + i * xStep;
                  const y = padding + (svgHeight - padding * 2) - (diff / 4.0) * (svgHeight - padding * 2);
                  return (
                    <circle
                      key={i}
                      cx={x}
                      cy={y}
                      r={4}
                      fill="#10b981"
                      className="hover:scale-150 transition-all cursor-pointer"
                    >
                      <title>{`Turn ${i + 1}: ${difficultyMapping[diff]}`}</title>
                    </circle>
                  );
                })}
              </svg>
              <div className="mt-3 flex justify-between w-full text-[10px] text-slate-400 font-mono">
                <span>Start</span>
                <span className="text-indigo-500 dark:text-indigo-400 font-bold uppercase">DQN Difficulty Control</span>
                <span>Turn {difficultyHistory.length}</span>
              </div>
            </div>
          )}
        </div>

      </div>

      {/* Column 2: Coverage & Weaknesses */}
      <div className="space-y-6">
        
        {/* Concept Coverage */}
        <div className="p-6 rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl h-[245px] overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-base font-bold flex items-center gap-2 text-slate-800 dark:text-slate-200">
              <Target className="text-indigo-500" size={18} />
              Concept Coverage
            </h3>
            <span className="text-sm font-black text-indigo-500 bg-indigo-500/10 px-2 py-0.5 rounded-full">
              {Math.round(coveragePercent)}%
            </span>
          </div>

          <div className="space-y-3">
            {Object.keys(conceptBreakdown).length === 0 ? (
              <p className="text-xs text-slate-400 dark:text-slate-500 italic text-center py-8">
                No concepts mapped yet. Start the viva to check coverage.
              </p>
            ) : (
              Object.entries(conceptBreakdown).map(([concept, simScore]: [string, any]) => (
                <div key={concept} className="space-y-1">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-700 dark:text-slate-300 truncate max-w-[200px] capitalize">{concept.replace("_", " ")}</span>
                    <span className="font-mono text-slate-500">{simScore.toFixed(0)}% match</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        simScore >= 12
                          ? "bg-emerald-500"
                          : simScore >= 5
                            ? "bg-amber-500 animate-pulse"
                            : "bg-indigo-600/30"
                      }`}
                      style={{ width: `${Math.min(simScore * 5, 100)}%` }}
                    />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* XGBoost Weakness Detection */}
        <div className="p-6 rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl min-h-[260px] max-h-[460px] overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-base font-bold flex items-center gap-2 text-slate-800 dark:text-slate-200">
              <ShieldAlert className="text-rose-500" size={18} />
              XGBoost Weakness Detection
            </h3>
            <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${
              weaknesses.length > 0 
                ? "bg-rose-500/10 text-rose-500 border border-rose-500/20" 
                : "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20"
            }`}>
              {weaknesses.length > 0 ? `${weaknesses.length} Area${weaknesses.length > 1 ? 's' : ''} Flagged` : "All Clear"}
            </span>
          </div>
          
          {weaknesses.length > 0 ? (
            <div className="space-y-3">
              {weaknesses.map((w: any, i: number) => {
                const isObj = typeof w === "object" && w !== null;
                const concept = isObj ? w.concept : w;
                const riskProb = isObj ? (w.risk_probability ?? 80) : 80;
                const avgScore = isObj ? w.average_score : null;
                const difficulty = isObj ? w.difficulty_level : null;
                const coverage = isObj ? w.coverage : null;
                const attempts = isObj ? w.attempts : 1;
                const severity = isObj ? (w.severity || "Critical Risk") : "High Risk";
                const reason = isObj ? w.reason : "Evaluation score below pass threshold";
                const recommendation = isObj ? w.recommendation : `Review core formulas, principles, and implementation details for ${concept}.`;

                const isCritical = String(severity).toLowerCase().includes("critical") || riskProb >= 75;
                const isHigh = !isCritical && (String(severity).toLowerCase().includes("high") || riskProb >= 60);

                return (
                  <div 
                    key={i} 
                    className={`p-3.5 rounded-xl border transition-all ${
                      isCritical 
                        ? "bg-rose-500/10 border-rose-500/30 text-rose-950 dark:text-rose-200" 
                        : isHigh 
                          ? "bg-amber-500/10 border-amber-500/30 text-amber-950 dark:text-amber-200" 
                          : "bg-indigo-500/10 border-indigo-500/30 text-indigo-950 dark:text-indigo-200"
                    }`}
                  >
                    {/* Header Row */}
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <AlertTriangle size={16} className={isCritical ? "text-rose-500 shrink-0" : isHigh ? "text-amber-500 shrink-0" : "text-indigo-500 shrink-0"} />
                        <h4 className="text-xs font-bold capitalize text-slate-800 dark:text-slate-100">
                          {String(concept).replace(/_/g, " ")}
                        </h4>
                      </div>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        isCritical 
                          ? "bg-rose-500/20 text-rose-600 dark:text-rose-300 border border-rose-500/30" 
                          : isHigh 
                            ? "bg-amber-500/20 text-amber-600 dark:text-amber-300 border border-amber-500/30" 
                            : "bg-indigo-500/20 text-indigo-600 dark:text-indigo-300 border border-indigo-500/30"
                      }`}>
                        {severity}
                      </span>
                    </div>

                    {/* Risk Probability Meter */}
                    <div className="mb-2.5">
                      <div className="flex justify-between text-[10px] font-semibold mb-1 text-slate-600 dark:text-slate-400">
                        <span>XGBoost Weakness Risk Probability</span>
                        <span className="font-mono font-bold text-rose-500 dark:text-rose-400">{riskProb}%</span>
                      </div>
                      <div className="w-full bg-slate-200 dark:bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${
                            isCritical ? "bg-rose-500" : isHigh ? "bg-amber-500" : "bg-indigo-500"
                          }`}
                          style={{ width: `${Math.min(riskProb, 100)}%` }}
                        />
                      </div>
                    </div>

                    {/* Telemetry Metrics Badges */}
                    {avgScore !== null && (
                      <div className="grid grid-cols-4 gap-1.5 mb-2.5 text-center">
                        <div className="bg-white/40 dark:bg-slate-900/50 p-1 rounded border border-white/20 dark:border-white/5">
                          <div className="text-[9px] text-slate-500 uppercase">Avg Score</div>
                          <div className="text-[11px] font-bold text-slate-800 dark:text-slate-200">{avgScore}%</div>
                        </div>
                        <div className="bg-white/40 dark:bg-slate-900/50 p-1 rounded border border-white/20 dark:border-white/5">
                          <div className="text-[9px] text-slate-500 uppercase">Diff Lvl</div>
                          <div className="text-[11px] font-bold text-slate-800 dark:text-slate-200">{difficulty}/4</div>
                        </div>
                        <div className="bg-white/40 dark:bg-slate-900/50 p-1 rounded border border-white/20 dark:border-white/5">
                          <div className="text-[9px] text-slate-500 uppercase">Coverage</div>
                          <div className="text-[11px] font-bold text-slate-800 dark:text-slate-200">{coverage}%</div>
                        </div>
                        <div className="bg-white/40 dark:bg-slate-900/50 p-1 rounded border border-white/20 dark:border-white/5">
                          <div className="text-[9px] text-slate-500 uppercase">Attempts</div>
                          <div className="text-[11px] font-bold text-slate-800 dark:text-slate-200">{attempts}</div>
                        </div>
                      </div>
                    )}

                    {/* Reason & Recommendation */}
                    <div className="space-y-1 text-[11px] leading-relaxed">
                      <p className="text-slate-600 dark:text-slate-300">
                        <span className="font-semibold text-slate-700 dark:text-slate-200">Diagnosis: </span>
                        {reason}
                      </p>
                      <p className="text-slate-500 dark:text-slate-400 bg-white/20 dark:bg-black/20 p-2 rounded-lg border border-white/10 dark:border-white/5 text-[10px]">
                        <span className="font-bold text-indigo-500 dark:text-indigo-400">💡 Target Fix: </span>
                        {recommendation}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="h-[180px] flex flex-col items-center justify-center text-slate-400 dark:text-slate-500 text-center">
              <Award className="mb-2 opacity-60 text-emerald-500 animate-pulse" size={32} />
              <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">No weaknesses detected yet.</p>
              <p className="text-[10px] opacity-75 mt-1 max-w-[280px]">
                Trained XGBoost classifier assesses performance score, difficulty, and depth on every viva answer.
              </p>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
