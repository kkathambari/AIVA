"use client";

import { useState } from "react";
import { Upload, FileUp, Loader2, CheckCircle } from "lucide-react";
import { uploadDocument } from "@/lib/api";

export function UploadSection({ onUploadComplete }: { onUploadComplete: (data: any) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setSuccess(false);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const data = await uploadDocument(file);
      setSuccess(true);
      onUploadComplete(data);
    } catch (error) {
      console.error("Upload failed", error);
      alert("Failed to upload document. Is the backend running?");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-6 rounded-2xl bg-white/40 dark:bg-black/40 backdrop-blur-md border border-white/20 dark:border-white/10 shadow-xl">
      <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-indigo-300 dark:border-indigo-700 rounded-xl bg-indigo-50/50 dark:bg-indigo-950/20 transition-all hover:bg-indigo-50 dark:hover:bg-indigo-900/30">
        {!success ? (
          <>
            <div className="mb-4 p-4 rounded-full bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400">
              <Upload size={32} />
            </div>
            <h3 className="text-xl font-bold mb-2 text-slate-800 dark:text-slate-200">
              Upload Project Document
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-6 text-center max-w-sm">
              Upload your PDF or DOCX file to generate the Knowledge Graph and begin the adaptive viva.
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
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="mt-4 px-6 py-3 rounded-full bg-slate-800 dark:bg-slate-200 hover:bg-slate-900 dark:hover:bg-white text-white dark:text-slate-900 font-medium flex items-center gap-2 transition-all disabled:opacity-50"
              >
                {uploading ? <Loader2 className="animate-spin" size={18} /> : "Process Document"}
              </button>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center">
            <div className="mb-4 p-4 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400">
              <CheckCircle size={48} />
            </div>
            <h3 className="text-xl font-bold mb-2 text-slate-800 dark:text-slate-200">
              Processing Complete
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 text-center max-w-sm">
              Your document has been parsed, chunked, and vectorized. The Knowledge Graph is ready.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
