"use client";

import { useState } from "react";
import { Upload, FileUp, Loader2, CheckCircle } from "lucide-react";
import { uploadDocument } from "@/lib/api";

interface UploadSectionProps {
  onUploadComplete: (data: any) => void;
  currentDocumentName?: string;
}

export function UploadSection({ onUploadComplete, currentDocumentName }: UploadSectionProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [isReplacing, setIsReplacing] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const data = await uploadDocument(file);
      setIsReplacing(false);
      setFile(null);
      onUploadComplete(data);
    } catch (error) {
      console.error("Upload failed", error);
      alert("Failed to upload document. Is the backend running?");
    } finally {
      setUploading(false);
    }
  };

  const showActiveCard = currentDocumentName && !isReplacing && !file;

  return (
    <div className="p-6 rounded-2xl bg-white/40 dark:bg-black/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl">
      <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-indigo-300 dark:border-indigo-700 rounded-xl bg-indigo-50/50 dark:bg-indigo-950/20 transition-all hover:bg-indigo-50 dark:hover:bg-indigo-900/30">
        {showActiveCard ? (
          <div className="flex flex-col items-center text-center">
            <div className="mb-4 p-4 rounded-full bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400 shadow-md shadow-emerald-500/10">
              <CheckCircle size={40} />
            </div>
            <span className="text-[10px] uppercase font-bold tracking-wider px-2.5 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20 mb-2">
              Active Project Document
            </span>
            <h3 className="text-base font-black text-slate-800 dark:text-slate-100 break-all max-w-xs mb-1">
              {currentDocumentName}
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-5 max-w-xs">
              Synchronized across Learning, Examination, Knowledge Graph, and Analytics.
            </p>
            <button
              onClick={() => setIsReplacing(true)}
              className="px-4 py-2 rounded-full text-xs font-semibold bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800/60 hover:bg-indigo-50 dark:hover:bg-indigo-950/50 shadow-sm transition-all"
            >
              Upload Different Document
            </button>
          </div>
        ) : (
          <>
            <div className="mb-4 p-4 rounded-full bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400">
              <Upload size={32} />
            </div>
            <h3 className="text-xl font-bold mb-2 text-slate-800 dark:text-slate-200">
              {currentDocumentName ? "Update Project Document" : "Upload Project Document"}
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-6 text-center max-w-sm">
              Upload your PDF or DOCX file to re-index, rebuild the Knowledge Graph, and run viva examination.
            </p>

            <label className="cursor-pointer px-6 py-3 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium flex items-center gap-2 transition-all shadow-md shadow-indigo-600/20">
              <FileUp size={18} />
              <span>{file ? file.name : "Select File"}</span>
              <input
                type="file"
                className="hidden"
                accept=".pdf,.docx"
                onChange={handleFileChange}
              />
            </label>

            {file && (
              <div className="flex items-center gap-2 mt-4">
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="px-6 py-3 rounded-full bg-slate-800 dark:bg-slate-200 hover:bg-slate-900 dark:hover:bg-white text-white dark:text-slate-900 font-medium flex items-center gap-2 transition-all disabled:opacity-50"
                >
                  {uploading ? <Loader2 className="animate-spin" size={18} /> : "Process Document"}
                </button>
                {currentDocumentName && (
                  <button
                    onClick={() => {
                      setFile(null);
                      setIsReplacing(false);
                    }}
                    disabled={uploading}
                    className="px-4 py-3 rounded-full text-xs font-semibold text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
                  >
                    Cancel
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
