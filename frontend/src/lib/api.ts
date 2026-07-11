const API_PROXY_BASE_URL = "/api-proxy";
const ADMIN_PROXY_BASE_URL = "/admin-proxy";

export type Emotion = { valence: number; arousal: number; optimal_loss: number };
export type ModelInfo = { model_id: string; adapter_id: string | null; fallback_used: boolean };
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
  attachments: Attachment[];
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
  tags: string[];
  operator_metadata: Record<string, unknown>;
};

export type SemanticMemory = {
  id: string;
  text: string;
  source_episode_ids: string[];
  record_type: string;
  archived: boolean;
  created_at: string;
  tags: string[];
  operator_metadata: Record<string, unknown>;
};

export type MemorySearchResponse = { db1_results: EpisodeMemory[]; db2_results: SemanticMemory[] };
export type MemoryMetadataUpdate = { tags?: string[]; operator_metadata?: Record<string, unknown> };

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
export type EvaluationResultSummary = {
  filename: string;
  adapter_id: string;
  score: number | null;
  decision: string | null;
  case_count: number | null;
  updated_at: string;
};
export type EvaluationResultListResponse = { results: EvaluationResultSummary[] };
export type EvaluationResultDetail = { filename: string; payload: Record<string, unknown> };
export type SleepRunResponse = {
  selected_episode_ids: string[];
  semantic_memory_ids: string[];
  dream_dataset_path: string | null;
  adapter_id: string | null;
  adapter_status: string | null;
  dry_run: boolean | null;
};
export type BuildInfo = { version: string; commit: string | null };
export type RuntimeInfo = {
  environment: string;
  provider: string;
  primary_model_id: string;
  fallback_configured: boolean;
  transformers_4bit: boolean;
  qlora_dry_run: boolean;
  admin_token_configured: boolean;
};
export type SystemInfoResponse = { project: string; status: string; build: BuildInfo; runtime: RuntimeInfo };
export type RuntimeEvent = {
  id: number;
  timestamp: string;
  category: string;
  event_type: string;
  message: string;
  metadata: Record<string, unknown>;
};
export type RuntimeEventListResponse = { events: RuntimeEvent[] };

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null = null,
    readonly statusText: string | null = null,
    readonly detail: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${API_PROXY_BASE_URL}${path.replace(/^\/api/, "")}`, init);
}

async function adminRequest<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${ADMIN_PROXY_BASE_URL}${path}`, init);
}

async function requestUrl<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (error) {
    throw new ApiError("Backend unavailable. Check that the API and frontend proxy are running.", null, null, error instanceof Error ? error.message : String(error));
  }
  if (!response.ok) {
    const detail = await readErrorDetail(response);
    throw new ApiError(formatHttpError(response.status, response.statusText, detail), response.status, response.statusText, detail);
  }
  return response.json() as Promise<T>;
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

async function readErrorDetail(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) return "";
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    return typeof parsed.detail === "string" ? parsed.detail : text;
  } catch {
    return text;
  }
}

function formatHttpError(status: number, statusText: string, detail: string): string {
  if (status === 401 || status === 403) {
    return detail ? `Admin access denied: ${detail}` : "Admin access denied. Check the admin token or private access boundary.";
  }
  if (status === 503) {
    return detail ? `Admin backend is not configured: ${detail}` : "Admin backend is not configured. Check KAGYA_ADMIN_TOKEN.";
  }
  if (status >= 500) {
    return detail ? `Backend failed: ${detail}` : `Backend failed with ${status} ${statusText}.`;
  }
  return detail ? `${status} ${statusText}: ${detail}` : `${status} ${statusText}`;
}

export const api = {
  chat: (body: ChatRequest) => request<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify(body) }),
  debugChat: (body: ChatRequest) => adminRequest<DebugChatResponse>("/chat/debug", { method: "POST", body: JSON.stringify(body) }),
  emotion: () => adminRequest<Emotion>("/state/emotion"),
  memorySearch: (query: string) => adminRequest<MemorySearchResponse>(`/memory/search?query=${encodeURIComponent(query)}`),
  archiveEpisodeMemory: (episodeId: string) => adminRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/archive`, { method: "POST" }),
  updateEpisodeMemoryMetadata: (episodeId: string, body: MemoryMetadataUpdate) => adminRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/metadata`, { method: "POST", body: JSON.stringify(body) }),
  archiveSemanticMemory: (memoryId: string) => adminRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/archive`, { method: "POST" }),
  updateSemanticMemoryMetadata: (memoryId: string, body: MemoryMetadataUpdate) => adminRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/metadata`, { method: "POST", body: JSON.stringify(body) }),
  sleepRun: () => adminRequest<SleepRunResponse>("/sleep/run", { method: "POST" }),
  adapters: () => adminRequest<AdapterListResponse>("/adapters"),
  evaluateAdapter: (adapterId: string, deterministic_score?: number) => adminRequest<AdapterEvaluateResponse>(`/adapters/${adapterId}/evaluate`, { method: "POST", body: JSON.stringify({ deterministic_score }) }),
  trialAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/trial`, { method: "POST" }),
  approveAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/approve`, { method: "POST" }),
  activateAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/activate`, { method: "POST" }),
  rejectAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/reject`, { method: "POST" }),
  evaluations: () => adminRequest<EvaluationResultListResponse>("/evaluations"),
  evaluationResult: (filename: string) => adminRequest<EvaluationResultDetail>(`/evaluations/${encodeURIComponent(filename)}`),
  systemInfo: () => requestUrl<SystemInfoResponse>("/api-proxy/system/info"),
  runtimeEvents: () => adminRequest<RuntimeEventListResponse>("/system/events"),
};
