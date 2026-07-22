const API_PROXY_BASE_URL = "/api-proxy";
const ADMIN_PROXY_BASE_URL = "/admin-proxy";
const ADMIN_AUTH_ENABLED = process.env.NEXT_PUBLIC_KAGYA_ADMIN_AUTH_ENABLED === "true";
let adminSessionPromise: Promise<void> | null = null;
let adminCsrfToken: string | null = null;

export type Emotion = { valence: number; arousal: number; optimal_loss: number };
export type ModelInfo = { model_id: string; adapter_id: string | null; adapter_hash: string | null; activation_sequence: number | null; fallback_used: boolean };
export type Attachment = { type: string; url?: string; name?: string; content_type?: string };

export type ChatRequest = { text: string; attachments?: Attachment[]; debug?: boolean; context_id?: string; client_session_id?: string; interlocutor_key?: string };
export type ChatResponse = {
  context_id: string;
  episode_id: string;
  experience_id: string;
  response: string;
  emotion: Emotion;
  model: ModelInfo;
};
export type FeedbackSignal = "good" | "bad" | "factual_error" | "style_problem" | "unsafe_behavior" | "remember" | "do_not_remember" | "correction" | "expected_answer" | "exclude_from_training";
export type FeedbackResponse = {
  feedback_id: string;
  current_revision: number;
  revisions: Array<{
    revision: number;
    status: "active" | "withdrawn";
    signals: FeedbackSignal[];
    propagation: { training_disposition: "include" | "exclude"; correction_memory_id: string | null };
  }>;
};
export type FeedbackRequest = {
  idempotency_key: string;
  target: { target_type: "response"; target_id: string; episode_id: string; experience_id: string; context_id: string };
  signals: FeedbackSignal[];
  correction?: string;
  expected_answer?: string;
};

export type RetrievedMemory = {
  db1_results: Array<{ id: string; user_input: string; response: string; record_type: string; context_id: string | null; semantic_relevance: number; context_compatibility: number; context_relation: string; cross_context: boolean }>;
  db2_results: Array<{ id: string; text: string; record_type: string; context_id: string | null; semantic_relevance: number; context_compatibility: number; context_relation: string; cross_context: boolean }>;
};

export type DebugChatResponse = ChatResponse & {
  hidden_thought: string;
  loss: number | null;
  prompt: string;
  attachments: Attachment[];
  retrieved_memory: RetrievedMemory;
  generation_params: { max_new_tokens: number; temperature: number; top_p: number; do_sample: boolean; repetition_penalty: number; no_repeat_ngram_size: number };
  working_memory: {
    items: Array<{ item_id: string; kind: string; selected: boolean; score: number; reasons: string[]; activation: number; salience: number; retention_reason: string; reference: string | null; context_id: string | null; context_compatibility: number; context_relation: string; cross_context: boolean }>;
    token_count: number;
    item_capacity: number;
    token_capacity: number;
  };
  loss_measurement: { raw_loss: number | null; mean_token_loss: number | null; target_token_count: number | null; model_key: string; valid: boolean; invalid_reason: string | null; calibrated_novelty: number | null };
  appraisal: { novelty: number | null; goal_progress: number; threat: number; controllability: number; certainty: number; social_relevance: number; effort_cost: number; novelty_valid: boolean; reasons: string[] };
  emotion_update: { valence_contributions: Record<string, number>; arousal_contributions: Record<string, number>; reasons: string[] };
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
  validation_status: string;
  lifecycle_status: string;
  generation_healthy: boolean;
  generation_health_reasons: string[];
  content_hash: string;
  source_event_id: string | null;
  source: string;
  processing_sequence: number | null;
  snapshot_sequence: number | null;
  provider: string;
  model_id: string;
  model_revision: string;
  adapter_id: string | null;
  consolidation_status: string;
  context_id: string | null;
  source_channel: string;
  source_session_id: string | null;
  semantic_relevance: number;
  context_compatibility: number;
  context_relation: string;
  cross_context: boolean;
  experience_id: string | null;
  subjective_salience: number;
  autobiographical_importance: number;
  supersedes_id: string | null;
  corrected_by_id: string | null;
  training_included: boolean;
  training_exclusion_refs: string[];
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
  context_id: string | null;
  source_channel: string;
  source_session_id: string | null;
  semantic_relevance: number;
  context_compatibility: number;
  context_relation: string;
  cross_context: boolean;
  schema_version: number;
  version: number;
  content_hash: string;
  confidence: number;
  effective_confidence: number;
  validity: string;
  valid_from: string | null;
  valid_until: string | null;
  expires_at: string | null;
  decay_rate: number;
  last_confirmed_at: string;
  lifecycle_status: string;
  supersedes_id: string | null;
  superseded_by_id: string | null;
  corrected_by_id: string | null;
  contradiction_ids: string[];
  source_feedback_ids: string[];
  merge_candidate_ids: string[];
  audit_log: Array<{ event_id: string; operation: string; detail: Record<string, unknown>; idempotency_key: string | null; created_at: string }>;
};

export type MemorySearchResponse = { db1_results: EpisodeMemory[]; db2_results: SemanticMemory[] };
export type MemoryMetadataUpdate = { tags?: string[]; operator_metadata?: Record<string, unknown> };
export type MemoryReviewUpdate = { validation_status: string; lifecycle_status: string };

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
  base_model_revision: string | null;
  adapter_hash: string | null;
  parent_adapter_id: string | null;
  parent_adapter_hash: string | null;
  activation_sequence: number | null;
  dataset_repetition_count: number;
  dataset_overlap_count: number;
  dataset_overlap_ratio: number;
  holdout_score: number | null;
  holdout_baseline_score: number | null;
  holdout_regression: boolean;
  drift_scores: Record<string, number> | null;
  activation_gate_passed: boolean;
  rollout_state: string;
  canary_failures: number;
  rollback_target_id: string | null;
};

export type AdapterListResponse = { adapters: Adapter[] };
export type AdapterProvenance = { adapter: Adapter; lineage: Adapter[]; activation_history: AdapterActivationResponse[] };
export type AdapterEvaluateResponse = { adapter_id: string; score: number; decision: string; result_path: string; status: string };
export type AdapterActivationResponse = { action: string; adapter_id: string | null; adapter_hash: string | null; previous_adapter_id: string | null; previous_adapter_hash: string | null; activation_sequence: number };
export type AdapterRuntimeState = { base_model: string; adapter_id: string | null; adapter_hash: string | null; activation_sequence: number | null };
export type EvaluationResultSummary = {
  filename: string;
  adapter_id: string;
  score: number | null;
  previous_score: number | null;
  score_delta: number | null;
  regression: boolean;
  decision: string | null;
  status_before: string | null;
  status_after: string | null;
  case_count: number | null;
  updated_at: string;
};
export type EvaluationResultListResponse = { results: EvaluationResultSummary[] };
export type AdapterEvaluationHistoryResponse = { adapter_id: string; results: EvaluationResultSummary[] };
export type EvaluationResultDetail = { filename: string; payload: Record<string, unknown> };
export type TrainingJob = {
  job_id: string;
  attempt_id: string;
  idempotency_key: string;
  status: string;
  bundle_path: string | null;
  bundle_hash: string | null;
  base_model_id: string;
  base_model_revision: string;
  parent_adapter_id: string | null;
  source_event_sequence_start: number;
  source_event_sequence_end: number;
  backend: string;
  remote_job_id: string | null;
  candidate_adapter_id: string | null;
  selected_episode_ids: string[];
  semantic_memory_ids: string[];
  created_at: string;
  updated_at: string;
  error: string | null;
  retry_count: number;
  phase_started_at: string | null;
  phase_durations_seconds: Record<string, number>;
  transferred_bytes: number;
  remote_last_contact: string | null;
  worker_node_id: string | null;
  worker_hostname: string | null;
  failure_category: string | null;
  retryable: boolean | null;
  import_status: string;
  correlation_id: string | null;
  processor_revision: string | null;
  training_metrics: Record<string, unknown>;
  total_duration_seconds: number;
  stale: boolean;
  dataset_revision: string | null;
  dataset_manifest_hash: string | null;
};
export type TrainingJobListResponse = { jobs: TrainingJob[] };
export type DatasetRevisionSummary = {
  revision: string;
  parent_revision: string | null;
  created_at: string;
  source_job_id: string | null;
  record_count: number;
  disposition_counts: Record<string, number>;
  split_counts: Record<string, number>;
  quality_findings: string[];
  record_ids: string[];
  manifest_hash: string;
};
export type GovernedDatasetRecord = {
  record_id: string;
  schema_version: number;
  input: string;
  thought: string;
  output: string;
  provenance: { source_kind: string; source_id: string; source_event_ids: string[]; source_memory_ids: string[]; source_decision_ids: string[]; source_feedback_ids: string[] };
  inclusion_reason: string;
  consent: string;
  privacy: string;
  disposition: "included" | "excluded" | "quarantined";
  split: "train" | "validation" | "test" | null;
  content_hash: string;
  quarantine_reasons: string[];
  exclusion_reasons: string[];
  quality_checks: string[];
  context_id: string | null;
  interlocutor_id: string | null;
};
export type DatasetRevisionDetail = { manifest: DatasetRevisionSummary; records: GovernedDatasetRecord[] };
export type DatasetRevisionDiff = { from_revision: string; to_revision: string; added_record_ids: string[]; removed_record_ids: string[]; changed_record_ids: string[] };
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
export type JournalRecord = {
  record_id: string;
  timestamp: string;
  lifecycle: string;
  event_id: string;
  event_type: string;
  source: string;
  processing_sequence: number | null;
  snapshot_sequence: number | null;
  causation_id: string | null;
  correlation_id: string | null;
  state_hash_before: string | null;
  state_hash_after: string | null;
  snapshot_hash: string | null;
  failure_category: string | null;
  actor_id: string | null;
  actor_role: string | null;
  target: string | null;
  reauthenticated: boolean | null;
  previous_record_hash: string | null;
  record_hash: string;
};
export type JournalRecordListResponse = { records: JournalRecord[] };
export type Experience = {
  experience_id: string;
  source_event_id: string | null;
  source_event_sequence: number | null;
  external_observation_refs: string[];
  subject_action_refs: string[];
  identity_origin: Record<string, unknown>;
  context_id: string;
  interlocutor_ids: string[];
  situation_codes: string[];
  interpretation_codes: string[];
  self_relevance: number;
  appraisal: Record<string, unknown>;
  subjective_salience: number;
  familiarity: number;
  agency_attribution: string;
  prediction_error: number | null;
  value_revision_refs: Record<string, number>;
  active_goal_refs: string[];
  self_model_revision: number;
  unresolved_tension: number;
  autobiographical_importance: number;
  result_refs: Record<string, string[]>;
  created_at: string;
  updated_at: string;
  revision: number;
  revisions: Array<Record<string, unknown>>;
  schema_version: number;
};
export type ExperienceListResponse = { experiences: Experience[] };
export type Belief = {
  belief_id: string;
  proposition: { normalized: string; subject: string | null; predicate: string | null; object: string | null };
  confidence: number;
  epistemic_status: string;
  lifecycle: string;
  identity_origin: Record<string, unknown>;
  evidence: Array<Record<string, unknown>>;
  context_scope: string[];
  valid_from: string | null;
  valid_until: string | null;
  contradiction_ids: string[];
  supersedes_id: string | null;
  superseded_by_id: string | null;
  revision: number;
  revisions: Array<Record<string, unknown>>;
  schema_version: number;
};
export type BeliefListResponse = { beliefs: Belief[] };
export type MotivationState = { schema_version: number; records: Array<Record<string, unknown>>; episodes: Array<Record<string, unknown>> };
export type OutboxMessage = {
  schema_version: 1;
  message_id: string;
  revision: number;
  kind: "question" | "approval_request" | "commitment_deadline" | "goal_state" | "action_result" | "anomaly" | "renegotiation" | "long_task_complete";
  title: string;
  body: string;
  context_id: string | null;
  interlocutor_id: string | null;
  references: { event_id: string | null; goal_id: string | null; plan_id: string | null; decision_id: string | null; action_id: string | null; commitment_id: string | null };
  urgency: "low" | "normal" | "high" | "critical";
  not_before: string;
  expires_at: string | null;
  channel: "local";
  privacy_class: "public" | "interlocutor" | "operator";
  delivery_status: "pending" | "delivered" | "failed" | "expired" | "cancelled";
  acknowledgment_status: "unacknowledged" | "read" | "replied" | "approved" | "rejected";
  deduplication_key: string;
  created_at: string;
  updated_at: string;
  delivered_at: string | null;
  acknowledged_at: string | null;
  attempts: Array<{ attempt: number; attempted_at: string; status: "delivered" | "failed"; failure_code: string | null }>;
  responses: Array<{ response_id: string; kind: string; actor_id: string; received_at: string; text: string | null; event_id: string | null; event_sequence: number | null }>;
  last_failure_code: string | null;
};
export type OutboxMessageListResponse = { messages: OutboxMessage[] };

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
  if (ADMIN_AUTH_ENABLED) await initializeAdminSession();
  const headers = new Headers(init?.headers);
  if (adminCsrfToken && (init?.method ?? "GET") !== "GET") {
    headers.set("X-KAGYA-CSRF-Token", adminCsrfToken);
  }
  return requestUrl<T>(`${ADMIN_PROXY_BASE_URL}${path}`, { ...init, headers });
}

export function initializeAdminSession(): Promise<void> {
  adminSessionPromise ??= fetch(`${ADMIN_PROXY_BASE_URL}/auth/session`, {
    headers: { "Content-Type": "application/json" },
  }).then(async (response) => {
    if (!response.ok) {
      const detail = await readErrorDetail(response);
      throw new ApiError(formatHttpError(response.status, response.statusText, detail), response.status, response.statusText, detail);
    }
    const body = await response.json() as { csrfToken: string };
    adminCsrfToken = body.csrfToken;
  }).catch((error) => {
    adminSessionPromise = null;
    throw error;
  });
  return adminSessionPromise;
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
  feedback: (body: FeedbackRequest) => request<FeedbackResponse>("/api/feedback", { method: "POST", body: JSON.stringify(body) }),
  debugChat: (body: ChatRequest) => adminRequest<DebugChatResponse>("/chat/debug", { method: "POST", body: JSON.stringify(body) }),
  emotion: () => adminRequest<Emotion>("/state/emotion"),
  memorySearch: (query: string) => adminRequest<MemorySearchResponse>(`/memory/search?query=${encodeURIComponent(query)}`),
  archiveEpisodeMemory: (episodeId: string) => adminRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/archive`, { method: "POST" }),
  updateEpisodeMemoryMetadata: (episodeId: string, body: MemoryMetadataUpdate) => adminRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/metadata`, { method: "POST", body: JSON.stringify(body) }),
  reviewEpisodeMemory: (episodeId: string, body: MemoryReviewUpdate) => adminRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/review`, { method: "POST", body: JSON.stringify(body) }),
  archiveSemanticMemory: (memoryId: string) => adminRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/archive`, { method: "POST" }),
  updateSemanticMemoryMetadata: (memoryId: string, body: MemoryMetadataUpdate) => adminRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/metadata`, { method: "POST", body: JSON.stringify(body) }),
  updateSemanticLifecycle: (memoryId: string, action: "archive" | "restore" | "forget", idempotencyKey: string) => adminRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/lifecycle`, { method: "POST", body: JSON.stringify({ action, idempotency_key: idempotencyKey }) }),
  semanticGraph: (memoryId: string) => adminRequest<{ records: SemanticMemory[] }>(`/memory/semantic/${encodeURIComponent(memoryId)}/graph`),
  relateSemanticMemory: (memoryId: string, targetId: string, relationship: "merge" | "contradiction" | "supersession" | "correction", idempotencyKey: string) => adminRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/relationships`, { method: "POST", body: JSON.stringify({ target_id: targetId, relationship, idempotency_key: idempotencyKey }) }),
  updateSemanticPolicy: (memoryId: string, body: { confidence: number; validity: "valid" | "disputed" | "invalid"; valid_from?: string | null; valid_until?: string | null; expires_at?: string | null; decay_rate: number; idempotency_key: string }) => adminRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/policy`, { method: "POST", body: JSON.stringify(body) }),
  createSleepJob: (idempotency_key?: string) => adminRequest<TrainingJob>("/sleep/jobs", { method: "POST", body: JSON.stringify({ idempotency_key }) }),
  sleepJobs: () => adminRequest<TrainingJobListResponse>("/sleep/jobs"),
  datasetRevisions: () => adminRequest<{ datasets: DatasetRevisionSummary[] }>("/training/datasets"),
  datasetRevision: (revision: string) => adminRequest<DatasetRevisionDetail>(`/training/datasets/${encodeURIComponent(revision)}`),
  datasetRevisionDiff: (fromRevision: string, toRevision: string) => adminRequest<DatasetRevisionDiff>(`/training/datasets/diff?from=${encodeURIComponent(fromRevision)}&to=${encodeURIComponent(toRevision)}`),
  cancelSleepJob: (jobId: string) => adminRequest<TrainingJob>(`/sleep/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }),
  retrySleepJob: (jobId: string) => adminRequest<TrainingJob>(`/sleep/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" }),
  reconcileSleepJob: (jobId: string) => adminRequest<TrainingJob>(`/sleep/jobs/${encodeURIComponent(jobId)}/reconcile`, { method: "POST" }),
  reconcileSleepJobs: () => adminRequest<{ jobs: TrainingJob[]; orphan_result_job_ids: string[]; orphan_remote_job_ids: string[] }>("/sleep/reconcile", { method: "POST" }),
  cleanupSleepArtifacts: () => adminRequest<{ removed: string[]; remote_removed: string[]; retention_days: number }>("/sleep/cleanup", { method: "POST" }),
  adapters: () => adminRequest<AdapterListResponse>("/adapters"),
  adapterProvenance: (adapterId: string) => adminRequest<AdapterProvenance>(`/adapters/${encodeURIComponent(adapterId)}/provenance`),
  adapterRuntime: () => adminRequest<AdapterRuntimeState>("/adapters/runtime"),
  evaluateAdapter: (adapterId: string, deterministic_score?: number) => adminRequest<AdapterEvaluateResponse>(`/adapters/${adapterId}/evaluate`, { method: "POST", body: JSON.stringify({ deterministic_score }) }),
  trialAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/trial`, { method: "POST" }),
  approveAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/approve`, { method: "POST" }),
  activateAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/activate`, { method: "POST" }),
  rollbackAdapter: () => adminRequest<AdapterActivationResponse>("/adapters/rollback", { method: "POST" }),
  rejectAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/reject`, { method: "POST" }),
  evaluations: () => adminRequest<EvaluationResultListResponse>("/evaluations"),
  adapterEvaluationHistory: (adapterId: string) => adminRequest<AdapterEvaluationHistoryResponse>(`/evaluations/adapters/${encodeURIComponent(adapterId)}/history`),
  evaluationResult: (filename: string) => adminRequest<EvaluationResultDetail>(`/evaluations/${encodeURIComponent(filename)}`),
  systemInfo: () => requestUrl<SystemInfoResponse>("/api-proxy/system/info"),
  runtimeEvents: () => adminRequest<RuntimeEventListResponse>("/system/events"),
  eventJournal: () => adminRequest<JournalRecordListResponse>("/system/journal"),
  experiences: () => adminRequest<ExperienceListResponse>("/experiences"),
  experience: (experienceId: string) => adminRequest<Experience>(`/experiences/${encodeURIComponent(experienceId)}`),
  beliefs: (activeOnly = false) => adminRequest<BeliefListResponse>(`/beliefs?active_only=${activeOnly}`),
  motivation: () => adminRequest<MotivationState>("/motivation"),
  outboxMessages: () => adminRequest<OutboxMessageListResponse>("/outbox/messages"),
  deliverOutbox: () => adminRequest<OutboxMessageListResponse>("/outbox/deliveries", { method: "POST" }),
  respondToOutbox: (messageId: string, kind: "read" | "reply" | "approval" | "reject", text?: string) => adminRequest<OutboxMessage>(`/outbox/messages/${encodeURIComponent(messageId)}/responses`, { method: "POST", body: JSON.stringify({ kind, text }) }),
};
