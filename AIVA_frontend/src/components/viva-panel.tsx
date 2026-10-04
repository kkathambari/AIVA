"use client";

import { useState, useRef } from "react";
import { Mic, Send, Bot, User, Loader2, Sparkles, TrendingUp, Play, RotateCcw } from "lucide-react";
import { sendVivaQuestion, transcribeAudio, generateTTS, resetExamination } from "@/lib/api";

interface VivaPanelProps {
  messages: { 
    role: string; 
    content: string; 
    agent?: string;
    evaluation?: any; 
  }[];
  setMessages: React.Dispatch<React.SetStateAction<any[]>>;
  onTurnComplete?: (data: any) => void;
  activeDifficulty?: number;
  loudspeakerEnabled: boolean;
  currentDocumentName?: string;
}

export function VivaPanel({ 
  messages, 
  setMessages, 
  onTurnComplete, 
  activeDifficulty = 1, 
  loudspeakerEnabled,
  currentDocumentName
}: VivaPanelProps) {
  const [input, setInput] = useState("");
  const [agentType, setAgentType] = useState("Auto (Panel)");
  const [testLength, setTestLength] = useState<number>(5);
  const [isTestComplete, setIsTestComplete] = useState<boolean>(false);
  const [loading, setLoading] = useState(false);
  const [startingViva, setStartingViva] = useState(false);
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Format difficulty level to text
  const difficultyMapping: { [key: number]: string } = {
    0: "Fundamentals (Very Easy)",
    1: "Easy",
    2: "Medium",
    3: "Hard",
    4: "Research Level (Very Hard)"
  };

  const handleStartViva = async (selectedQuestionsCount?: number) => {
    const questionsCount = selectedQuestionsCount !== undefined ? selectedQuestionsCount : testLength;
    if (selectedQuestionsCount !== undefined) {
      setTestLength(selectedQuestionsCount);
    }
    setStartingViva(true);
    setIsTestComplete(false);
    setLoading(true);

    try {
      // 1. Reset backend session examination state
      await resetExamination();

      // 2. Call backend with clean empty history to formulate the opening question
      const data = await sendVivaQuestion([], agentType, activeDifficulty, false);

      const aiResponse = data.question;
      const actualAgent = data.agent;
      const analytics = data.analytics;

      // 3. Set the first message from the examiner
      setMessages([{
        role: "ai",
        content: aiResponse,
        agent: actualAgent
      }]);

      if (onTurnComplete && analytics) {
        onTurnComplete({
          analytics: analytics,
          evaluation: null,
          question: aiResponse,
          agent: actualAgent
        });
      }

      if (loudspeakerEnabled) {
        try {
          const audioUrl = await generateTTS(aiResponse);
          const audio = new Audio(audioUrl);
          audio.play();
        } catch (ttsErr) {
          console.error("TTS generation failed:", ttsErr);
        }
      }
    } catch (error) {
      console.error("Failed to start viva:", error);
      alert("Could not start viva examination. Please verify the backend is running.");
    } finally {
      setStartingViva(false);
      setLoading(false);
    }
  };

  const handleSend = async (text: string) => {
    if (!text.trim()) return;

    // Build history for backend API (standard role/content)
    const apiHistory = messages.map(m => ({
      role: m.role,
      content: m.content
    }));

    const newMessages = [...messages, { role: "user", content: text }];
    const apiHistoryWithNew = [...apiHistory, { role: "user", content: text }];
    
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const aiQuestionsCount = messages.filter(m => m.role !== 'user').length;
      // Check if this response completes the selected questions
      const isFinalTurn = testLength > 0 && aiQuestionsCount >= testLength;

      const data = await sendVivaQuestion(apiHistoryWithNew, agentType, 5, isFinalTurn);
      
      const aiResponse = data.question;
      const actualAgent = data.agent;
      const evaluation = data.evaluation;
      const analytics = data.analytics;

      // Attach the evaluation of the user's answer to the user's message
      const updatedMessages = [...newMessages];
      if (updatedMessages.length > 0 && evaluation) {
        updatedMessages[updatedMessages.length - 1] = {
          ...updatedMessages[updatedMessages.length - 1],
          evaluation: evaluation
        };
      }

      // Add AI response
      setMessages([...updatedMessages, { 
        role: "ai", 
        content: aiResponse,
        agent: actualAgent
      }]);

      if (isFinalTurn) {
        setIsTestComplete(true);
      }

      // Trigger analytics & history callback
      if (onTurnComplete) {
        onTurnComplete({
          analytics: analytics,
          evaluation: evaluation,
          question: aiResponse,
          agent: actualAgent
        });
      }

      // Automatically generate TTS for AI response if loudspeaker is enabled
      if (loudspeakerEnabled) {
        try {
          const audioUrl = await generateTTS(aiResponse);
          const audio = new Audio(audioUrl);
          audio.play();
        } catch (ttsErr) {
          console.error("TTS generation failed:", ttsErr);
        }
      }

    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType });
        setLoading(true);
        try {
          const text = await transcribeAudio(audioBlob);
          setInput(text);
        } catch (error) {
          console.error(error);
          alert("Could not transcribe audio.");
        } finally {
          setLoading(false);
        }
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (error) {
      console.error("Microphone access denied", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  return (
    <div className="flex flex-col h-[640px] rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-white/20 dark:border-white/10 flex flex-col xl:flex-row gap-3 justify-between items-start xl:items-center bg-white/20 dark:bg-black/30">
        <div className="flex items-center gap-2">
          <Bot className="text-indigo-500 shrink-0" />
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-bold text-base leading-tight">Viva Panel</h2>
              {currentDocumentName && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 font-semibold truncate max-w-[180px]" title={currentDocumentName}>
                  {currentDocumentName}
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              <TrendingUp size={12} className="text-emerald-500" />
              <span>Current Difficulty: <strong className="text-indigo-600 dark:text-indigo-400">{difficultyMapping[activeDifficulty] || "Easy"}</strong></span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 w-full xl:w-auto justify-start xl:justify-end">
          {/* Persona Selection */}
          <div className="flex flex-wrap gap-1">
            {["Auto (Panel)", "Examiner", "Critic", "Industry Expert", "Professor"].map((agent) => (
              <button
                key={agent}
                onClick={() => setAgentType(agent)}
                className={`px-2.5 py-1 text-xs font-semibold rounded-full transition-all ${
                  agentType === agent
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20"
                    : "bg-white/40 dark:bg-white/5 text-slate-700 dark:text-slate-300 hover:bg-indigo-100 dark:hover:bg-indigo-900/30"
                }`}
              >
                {agent}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-slate-300 dark:bg-slate-700 hidden md:block" />

          {/* Test Length & Start / Restart Button */}
          <div className="flex items-center gap-2">
            <select
              value={testLength}
              onChange={(e) => setTestLength(Number(e.target.value))}
              disabled={startingViva}
              className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-xs px-2.5 py-1.5 outline-none font-bold text-indigo-600 dark:text-indigo-400 cursor-pointer shadow-sm hover:border-indigo-300 transition-colors"
              title="Number of questions in the exam"
            >
              <option value={5}>5 Questions (Quick Test)</option>
              <option value={10}>10 Questions (Standard)</option>
              <option value={15}>15 Questions</option>
              <option value={20}>20 Questions (Final Exam)</option>
              <option value={0}>Endless Mode</option>
            </select>

            <button
              onClick={() => handleStartViva()}
              disabled={startingViva}
              className="px-3.5 py-1.5 text-xs font-bold rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white shadow-md shadow-emerald-600/20 flex items-center gap-1.5 transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
              title={messages.length === 0 ? "Start Viva Examination" : "Restart Viva Examination"}
            >
              {startingViva ? (
                <>
                  <Loader2 size={13} className="animate-spin" />
                  <span>Starting...</span>
                </>
              ) : messages.length === 0 ? (
                <>
                  <Play size={13} className="fill-current" />
                  <span>Start Viva</span>
                </>
              ) : (
                <>
                  <RotateCcw size={13} />
                  <span>Restart Exam</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50 dark:bg-slate-950/20">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500 dark:text-slate-400">
            <div className="mb-4 p-4 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 animate-pulse">
              <Bot size={40} />
            </div>
            <h3 className="text-lg font-black text-slate-800 dark:text-slate-100 mb-1">
              Oral Viva Examination
            </h3>
            <p className="font-medium text-xs max-w-md text-slate-500 dark:text-slate-400 mb-6">
              {currentDocumentName ? (
                <>Ready to evaluate based on <strong className="text-indigo-600 dark:text-indigo-400">{currentDocumentName}</strong>. Choose your number of questions and click Start.</>
              ) : (
                <>Select your desired examination length and persona, then click Start Viva to begin.</>
              )}
            </p>

            {/* Quick Test Length Options */}
            <div className="flex flex-col items-center gap-4 w-full max-w-md">
              <div className="w-full">
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-2">
                  Select Number of Questions:
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {[
                    { count: 5, label: "5 Questions", desc: "Quick Test" },
                    { count: 10, label: "10 Questions", desc: "Standard" },
                    { count: 20, label: "20 Questions", desc: "Comprehensive" },
                    { count: 0, label: "Endless", desc: "Open Practice" },
                  ].map((option) => (
                    <button
                      key={option.count}
                      onClick={() => setTestLength(option.count)}
                      className={`p-2.5 rounded-xl border text-center transition-all ${
                        testLength === option.count
                          ? "bg-indigo-600 text-white border-indigo-600 shadow-md shadow-indigo-600/20 scale-105"
                          : "bg-white dark:bg-slate-800/80 border-slate-200 dark:border-slate-700/80 text-slate-700 dark:text-slate-300 hover:border-indigo-400"
                      }`}
                    >
                      <div className="font-bold text-xs">{option.label}</div>
                      <div className={`text-[10px] mt-0.5 ${testLength === option.count ? "text-indigo-200" : "text-slate-400"}`}>
                        {option.desc}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Big Start Button */}
              <button
                onClick={() => handleStartViva()}
                disabled={startingViva}
                className="mt-2 w-full py-3.5 bg-gradient-to-r from-indigo-600 to-emerald-600 hover:from-indigo-700 hover:to-emerald-700 text-white font-bold rounded-2xl shadow-xl shadow-indigo-600/20 transition-all hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2.5 text-sm disabled:opacity-50"
              >
                {startingViva ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    <span>Preparing Examination Questions...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    <span>Start Examination ({testLength > 0 ? `${testLength} Questions` : "Endless Mode"})</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex flex-col gap-1 max-w-[85%] ${
              msg.role === "user" ? "ml-auto items-end" : "mr-auto items-start"
            }`}
          >
            {/* Agent Type Label */}
            {msg.role !== "user" && (
              <span className="text-[10px] font-bold text-indigo-500 dark:text-indigo-400 ml-11 tracking-wider uppercase">
                {msg.agent || "Examiner"}
              </span>
            )}
            
            <div className={`flex gap-3 items-start ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
              <div className={`p-2 rounded-full h-8 w-8 flex items-center justify-center shrink-0 shadow-sm ${
                msg.role === "user" ? "bg-indigo-600 text-white" : "bg-emerald-500 text-white"
              }`}>
                {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div
                className={`p-3.5 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-indigo-600 text-white rounded-tr-none shadow-md shadow-indigo-600/10"
                    : "bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 rounded-tl-none border border-slate-100 dark:border-slate-800/80 shadow-sm"
                }`}
              >
                {msg.content}
              </div>
            </div>

            {/* Score evaluation card under student answers */}
            {msg.role === "user" && msg.evaluation && (
              <div className="mr-11 mt-2 text-xs p-3 rounded-xl bg-slate-100/80 dark:bg-slate-850/80 border border-slate-200/50 dark:border-slate-800 text-slate-700 dark:text-slate-300 w-full max-w-md shadow-sm">
                <div className="flex justify-between items-center mb-1.5 pb-1.5 border-b border-slate-200/40 dark:border-slate-700/40">
                  <span className="font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1">
                    <Sparkles size={12} className="text-amber-500 animate-spin" />
                    Concept: {msg.evaluation.concept}
                  </span>
                  <span className={`font-mono px-2 py-0.5 rounded font-bold ${
                    msg.evaluation.score >= 0.7 
                      ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300' 
                      : msg.evaluation.score >= 0.4 
                        ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300' 
                        : 'bg-red-100 dark:bg-red-950/60 text-red-700 dark:text-red-300'
                  }`}>
                    Score: {Math.round(msg.evaluation.score * 100)}%
                  </span>
                </div>
                <p className="opacity-90 italic text-[11px] leading-normal">{msg.evaluation.feedback}</p>
              </div>
            )}
          </div>
        ))}

        {isTestComplete && (
          <div className="p-5 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-center space-y-2 mt-4 max-w-xl mx-auto shadow-sm">
            <h4 className="font-bold text-sm text-emerald-700 dark:text-emerald-300 flex items-center justify-center gap-1.5">
              <Sparkles size={16} /> Examination Completed!
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-300">
              You answered all {testLength} questions. Review your evaluated turns above or visit the Analytics tab for your full performance report.
            </p>
            <button
              onClick={() => handleStartViva()}
              className="mt-2 px-5 py-2 rounded-full text-xs font-bold bg-emerald-600 hover:bg-emerald-700 text-white shadow-md transition-all inline-flex items-center gap-1.5"
            >
              <RotateCcw size={13} />
              <span>Start New Examination</span>
            </button>
          </div>
        )}

        {loading && (
          <div className="flex gap-3 max-w-[80%] mr-auto items-center">
            <div className="p-2 rounded-full h-8 w-8 flex items-center justify-center shrink-0 bg-emerald-500 text-white shadow-sm">
              <Loader2 className="animate-spin" size={16} />
            </div>
            <span className="text-xs text-slate-400 dark:text-slate-500 italic animate-pulse">Examiner is formulating next question...</span>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 bg-white/20 dark:bg-black/30 border-t border-white/20 dark:border-white/10 flex flex-col gap-2">
        {testLength > 0 && (
          <div className="flex justify-between items-center px-1">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              {isTestComplete ? "Test Complete" : `Progress: ${Math.min(messages.filter(m => m.role !== 'user').length, testLength)} / ${testLength} Questions`}
            </span>
            <div className="h-1 flex-1 mx-3 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-indigo-500 transition-all duration-500" 
                style={{ width: `${Math.min((messages.filter(m => m.role !== 'user').length / testLength) * 100, 100)}%` }}
              />
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <button
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={stopRecording}
            disabled={isTestComplete}
            className={`p-3 rounded-full transition-all shadow-sm ${
              recording
                ? "bg-red-500 text-white animate-pulse"
                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50"
            }`}
            title="Hold to record"
          >
            <Mic size={20} />
          </button>
          
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend(input)}
            disabled={isTestComplete}
            placeholder={isTestComplete ? "Test completed. Click Restart Exam above or check Analytics." : "Type your answer or hold the mic..."}
            className="flex-1 px-4 py-2 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm disabled:opacity-50"
          />
          
          <button
            onClick={() => handleSend(input)}
            disabled={!input.trim() || loading || isTestComplete}
            className="p-3 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 transition-colors disabled:opacity-50 shadow-md shadow-indigo-600/10"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
