const API_PROXY_BASE_URL = "/api-proxy";
const ADMIN_PROXY_BASE_URL = "/admin-proxy";

export type Emotion = { valence: number; arousal: number; optimal_loss: number };
export type ModelInfo = { model_id: string; adapter_id: string | null };
export type Attachment = { type: string; url?: string; name?: string; content_type?: string };

export type ChatRequest = { text: string; attachments?: Attachment[]; debug?: boolean };
export type ChatResponse = {
  episode_id: string;
  response: string;
  emotion: Emotion;
  model: ModelInfo;
};

export type RetrievedMemory = {
  db1_results: Array<{ id: string; user_input: string; response: string; record_type: string }>;
  db2_results: Array<{ id: string; text: string; record_type: string }>;
};

export type DebugChatResponse = ChatResponse & {
  hidden_thought: string;
  loss: number;
  prompt: string;
  retrieved_memory: RetrievedMemory;
  generation_params: { max_new_tokens: number; temperature: number; top_p: number; do_sample: boolean };
};

export type EpisodeMemory = {
  id: string;
  user_input: string;
  response: string;
  loss: number;
  emotion_valence: number;
  emotion_arousal: number;
  record_type: string;
  archived: boolean;
  created_at: string;
};

export type SemanticMemory = {
  id: string;
  text: string;
  source_episode_ids: string[];
  record_type: string;
  created_at: string;
};

export type MemorySearchResponse = { db1_results: EpisodeMemory[]; db2_results: SemanticMemory[] };

export type Adapter = {
  adapter_id: string;
  base_model: string;
  path: string;
  status: string;
  dataset_path: string;
  dataset_hash: string;
  eval_score: number | null;
  eval_result_path: string | null;
  created_at: string;
  updated_at: string;
  notes: string;
};

export type AdapterListResponse = { adapters: Adapter[] };
export type AdapterEvaluateResponse = { adapter_id: string; score: number; decision: string; result_path: string; status: string };
export type SleepRunResponse = {
  selected_episode_ids: string[];
  semantic_memory_ids: string[];
  dream_dataset_path: string | null;
  adapter_id: string | null;
  adapter_status: string | null;
  dry_run: boolean | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${API_PROXY_BASE_URL}${path.replace(/^\/api/, "")}`, init);
}

async function adminRequest<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${ADMIN_PROXY_BASE_URL}${path}`, init);
}

async function requestUrl<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  chat: (body: ChatRequest) => request<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify(body) }),
  debugChat: (body: ChatRequest) => adminRequest<DebugChatResponse>("/chat/debug", { method: "POST", body: JSON.stringify(body) }),
  emotion: () => adminRequest<Emotion>("/state/emotion"),
  memorySearch: (query: string) => adminRequest<MemorySearchResponse>(`/memory/search?query=${encodeURIComponent(query)}`),
  sleepRun: () => adminRequest<SleepRunResponse>("/sleep/run", { method: "POST" }),
  adapters: () => adminRequest<AdapterListResponse>("/adapters"),
  evaluateAdapter: (adapterId: string, deterministic_score?: number) => adminRequest<AdapterEvaluateResponse>(`/adapters/${adapterId}/evaluate`, { method: "POST", body: JSON.stringify({ deterministic_score }) }),
  trialAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/trial`, { method: "POST" }),
  approveAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/approve`, { method: "POST" }),
  activateAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/activate`, { method: "POST" }),
  rejectAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/reject`, { method: "POST" }),
};
