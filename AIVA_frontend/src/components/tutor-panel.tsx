"use client";

import { useState, useRef } from "react";
import { Bot, Mic, Send, Trash2, User, Loader2 } from "lucide-react";
import { getTutorAnswer, transcribeAudio, generateTTS } from "@/lib/api";

export function TutorPanel() {
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleSend = async (text: string) => {
    if (!text.trim()) return;

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
      const data = await getTutorAnswer(text, apiHistoryWithNew);
      const tutorAnswer = data.answer;

      // Add Tutor response
      setMessages([...newMessages, { role: "tutor", content: tutorAnswer }]);

      // Automatically generate TTS for Tutor response
      const audioUrl = await generateTTS(tutorAnswer);
      const audio = new Audio(audioUrl);
      audio.play();

    } catch (error) {
      console.error(error);
      setMessages([...newMessages, { role: "tutor", content: "Error: Could not retrieve answer from the tutor. Is your backend running?" }]);
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

  const clearHistory = () => {
    setMessages([]);
  };

  return (
    <div className="flex flex-col h-[600px] rounded-2xl bg-white/10 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-white/20 dark:border-white/10 flex justify-between items-center bg-white/20 dark:bg-black/30">
        <div className="flex items-center gap-2">
          <Bot className="text-indigo-500" />
          <div>
            <h2 className="font-bold text-base leading-tight">Tutor Panel</h2>
            <span className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">Learn Mode: Ask questions about your project report.</span>
          </div>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearHistory}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/20 rounded-lg transition-all"
            title="Clear Chat History"
          >
            <Trash2 size={14} />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50 dark:bg-slate-950/20">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 dark:text-slate-400">
            <Bot size={48} className="mb-4 opacity-50 text-indigo-500 animate-pulse" />
            <p className="font-medium text-sm">Ask any question regarding your project report.</p>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">The Tutor will answer strictly based on the uploaded document.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex flex-col gap-1 max-w-[85%] ${
              msg.role === "user" ? "ml-auto items-end" : "mr-auto items-start"
            }`}
          >
            {msg.role !== "user" && (
              <span className="text-[10px] font-bold text-indigo-500 dark:text-indigo-400 ml-11 tracking-wider uppercase">
                Project Tutor
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
          </div>
        ))}
        {loading && (
          <div className="flex gap-3 max-w-[80%] mr-auto items-center">
            <div className="p-2 rounded-full h-8 w-8 flex items-center justify-center shrink-0 bg-emerald-500 text-white shadow-sm">
              <Loader2 className="animate-spin" size={16} />
            </div>
            <span className="text-xs text-slate-400 dark:text-slate-500 italic animate-pulse">Tutor is searching documentation...</span>
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
            placeholder="Ask a question about your project..."
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
