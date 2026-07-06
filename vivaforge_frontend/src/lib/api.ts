import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
});

export const uploadDocument = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/upload_and_build_kg", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const fetchCoverageAnalytics = async () => {
  const response = await api.get("/analytics");
  return response.data;
};

export const sendVivaQuestion = async (
  conversationHistory: any[],
  agentType: string,
  difficulty: number
) => {
  const response = await api.post("/multi_agent_question", {
    conversation_history: conversationHistory,
    agent_type: agentType,
    difficulty: difficulty,
  });
  return response.data;
};

export const generateTTS = async (text: string) => {
  const response = await api.post("/tts", { text }, { responseType: "blob" });
  return URL.createObjectURL(response.data);
};

export const transcribeAudio = async (audioBlob: Blob) => {
  const formData = new FormData();
  
  // Detect container format to supply correct extension
  let ext = "wav";
  if (audioBlob.type.includes("webm")) {
    ext = "webm";
  } else if (audioBlob.type.includes("mp4") || audioBlob.type.includes("m4a")) {
    ext = "mp4";
  } else if (audioBlob.type.includes("ogg")) {
    ext = "ogg";
  } else if (audioBlob.type.includes("mp3") || audioBlob.type.includes("mpeg")) {
    ext = "mp3";
  }

  formData.append("file", audioBlob, `recording.${ext}`);
  const response = await api.post("/stt", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data.transcription;
};

export const getTutorAnswer = async (
  question: string,
  conversationHistory: any[]
) => {
  const response = await api.post("/tutor_answer", {
    question,
    conversation_history: conversationHistory,
  });
  return response.data;
};

export const fetchRAGResults = async (query: string) => {
  const response = await api.post("/rag_query", { query });
  return response.data;
};
