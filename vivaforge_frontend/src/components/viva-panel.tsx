"use client";

import { useState, useRef } from "react";
import { Mic, Send, Bot, User, Loader2, Sparkles, TrendingUp } from "lucide-react";
import { sendVivaQuestion, transcribeAudio, generateTTS } from "@/lib/api";

interface VivaPanelProps {
  onTurnComplete?: (analytics: any) => void;
  activeDifficulty?: number;
}

export function VivaPanel({ onTurnComplete, activeDifficulty = 1 }: VivaPanelProps) {
  const [messages, setMessages] = useState<{ 
    role: string; 
    content: string; 
    agent?: string;
    evaluation?: any; 
  }[]>([]);
  const [input, setInput] = useState("");
  const [agentType, setAgentType] = useState("Auto (Panel)");
  const [loading, setLoading] = useState(false);
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

  const handleSend = async (text: string) => {
    if (!text.trim()) return;

    // Build history for backend API (we keep standard role/content)
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
      // difficulty 5 is passed as default, backend overrides it with RL
      const data = await sendVivaQuestion(apiHistoryWithNew, agentType, 5);
      
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

      // Trigger analytics callback
      if (onTurnComplete && analytics) {
        onTurnComplete(analytics);
      }

      // Automatically generate TTS for AI response
      const audioUrl = await generateTTS(aiResponse);
      const audio = new Audio(audioUrl);
      audio.play();

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
    <div className="flex flex-col h-[600px] rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-white/20 dark:border-white/10 flex flex-col md:flex-row gap-3 justify-between items-center bg-white/20 dark:bg-black/30">
        <div className="flex items-center gap-2">
          <Bot className="text-indigo-500" />
          <div>
            <h2 className="font-bold text-base leading-tight">Viva Panel</h2>
            <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              <TrendingUp size={12} className="text-emerald-500" />
              <span>Current Difficulty: <strong className="text-indigo-600 dark:text-indigo-400">{difficultyMapping[activeDifficulty] || "Easy"}</strong></span>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5 justify-end">
          {["Auto (Panel)", "Examiner", "Critic", "Industry Expert", "Professor"].map((agent) => (
            <button
              key={agent}
              onClick={() => setAgentType(agent)}
              className={`px-3 py-1 text-xs font-semibold rounded-full transition-all ${
                agentType === agent
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20"
                  : "bg-white/40 dark:bg-white/5 text-slate-700 dark:text-slate-300 hover:bg-indigo-100 dark:hover:bg-indigo-900/30"
              }`}
            >
              {agent}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50 dark:bg-slate-950/20">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 dark:text-slate-400">
            <Bot size={48} className="mb-4 opacity-50 text-indigo-500 animate-pulse" />
            <p className="font-medium text-sm">Select an examiner persona and send a message to begin the viva.</p>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Or choose "Auto (Panel)" for dynamic question routing.</p>
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

            {/* Render score evaluation card right under student answers */}
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
      <div className="p-4 bg-white/20 dark:bg-black/30 border-t border-white/20 dark:border-white/10">
        <div className="flex gap-2">
          <button
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={stopRecording}
            className={`p-3 rounded-full transition-all shadow-sm ${
              recording
                ? "bg-red-500 text-white animate-pulse"
                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
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
            placeholder="Type your answer or hold the mic..."
            className="flex-1 px-4 py-2 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
          />
          
          <button
            onClick={() => handleSend(input)}
            disabled={!input.trim() || loading}
            className="p-3 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 transition-colors disabled:opacity-50 shadow-md shadow-indigo-600/10"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
