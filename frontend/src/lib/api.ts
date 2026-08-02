const API_PROXY_BASE_URL = "/api-proxy";
const ADMIN_PROXY_BASE_URL = "/admin-proxy";

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
export type OperationState = "queued" | "running" | "finalizing" | "completed" | "failed" | "canceled";
export type OperationStatus = {
  schema_version: 1;
  operation_id: string;
  event_id: string;
  status: OperationState;
  status_sequence: number;
  queue_position: number | null;
  submitted_at: string;
  started_at: string | null;
  finalizing_at: string | null;
  completed_at: string | null;
  updated_at: string;
  error_code: "internal_error" | "interrupted" | "timeout" | "provider_error" | "commit_indeterminate" | "committed_result_unavailable" | null;
  cancel_code: "client_request" | "timeout" | "shutdown" | null;
  cancel_requested: boolean;
  result_available: boolean;
};
export type ChatJobAccepted = { operation: OperationStatus; status_url: string; result_url: string; events_url: string; duplicate: boolean };
export type ChatJobResult = { operation: OperationStatus; result: ChatResponse };
export type ChatCancelDisposition = "canceled" | "cancel_requested" | "already_completed" | "failed";
export type ChatCancelResponse = { disposition: ChatCancelDisposition; operation: OperationStatus };
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
  behavior_class: "respond" | "refuse" | "request_information" | "defer" | "no_op" | "unable";
  response_parse_valid: boolean;
  response_status: "valid" | "invalid_json" | "invalid_schema" | "invalid_empty_response" | "invalid_private_content";
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

export type DecisionExplanation = {
  schema_version: 1;
  explanation_id: string;
  revision: number;
  decision_id: string;
  decision_revision: number;
  decision_status: string;
  disposition: string;
  selected: { candidate_id: string; action_type: string; eligible: boolean; score: number | null; uncertainty: number; risk: number; disposition_code: string; reason_codes: string[] };
  major_alternatives: Array<{ candidate_id: string; action_type: string; eligible: boolean; score: number | null; uncertainty: number; risk: number; disposition_code: string; reason_codes: string[] }>;
  contributions: Array<{ source_type: string; source_id: string; source_revision: number; contribution: number | null; evidence_refs: string[]; origin_ref: string | null; availability: "available" }>;
  evidence_refs: string[];
  uncertainty: Array<{ code: string; severity: number; refs: string[] }>;
  information_gap_codes: string[];
  omitted_reference_count: number;
  risk: { risk_class: string; policy_status: string; approval_status: string; policy_ref: string | null; approval_ref: string | null; action_intent_ref: string | null; validation_ref: string | null; receipt_ref: string | null; observation_ref: string | null; verification_ref: string | null; policy_reason_codes: string[] };
  tradeoff_refs: string[];
  conflict_codes: string[];
  boundary: { assessment_id: string; assessment_revision: number; classification: string; recommendation: string; disposition: string; reason_codes: string[] } | null;
  reason_codes: string[];
  outcome: { status: string; utility: number | null; prediction_error: number | null; observed_event_ref: string | null; post_assessment_ref: string | null };
  change: { previous_explanation_revision: number | null; changed_fields: string[]; reason_codes: string[] };
  renderer: { state: string; deterministic_template: string; offered_clause_ids: string[]; ordered_clause_ids: string[]; visible_explanation: string; failure_code: string | null };
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
  quality_gate_passed: boolean | null;
  holdout_gate_passed: boolean | null;
  drift_gate_passed: boolean | null;
  activation_gate_passed: boolean;
  behavioral_evaluation_id: string | null;
  behavioral_evaluation_path: string | null;
  behavioral_result_hash: string | null;
  behavioral_gate_passed: boolean | null;
  behavioral_candidate_adapter_hash: string | null;
  behavioral_base_model_revision: string | null;
  subject_revision: string | null;
  fixture_set_hash: string | null;
  behavioral_artifact_state: string;
  deterministic_coverage_status: "complete" | "incomplete" | "not_evaluated";
  deterministic_behavioral_artifact_status: BehavioralArtifactStatus;
  real_model_behavioral_evaluation_id: string | null;
  real_model_behavioral_gate_passed: boolean | null;
  real_model_behavioral_artifact_state: string;
  real_model_coverage_status: "complete" | "incomplete" | "not_evaluated";
  real_model_behavioral_artifact_status: BehavioralArtifactStatus;
  behavioral_artifact_hash_match: "passed" | "failed" | "not_run";
  activation_eligibility_reason: string;
  real_model_behavioral_required: boolean;
  behavioral_activation_policy: "real_model_required" | "deterministic_runtime_only" | "disabled";
  legacy_activation_warning: boolean;
  rollout_state: string;
  canary_failures: number;
  rollback_target_id: string | null;
  identity_integrity_status: "passed" | "failed" | "not_evaluated" | "stale";
  real_model_identity_integrity_status: "passed" | "failed" | "not_evaluated" | "stale";
  candidate_boundary_probe_choice?: string | null;
  candidate_boundary_probe_margin?: number | null;
  candidate_boundary_probe_count?: number;
  rollback_reason: string | null;
};

export type BehavioralArtifactStatus = "not_run" | "prepared" | "valid" | "hash_mismatch" | "corrupt" | "orphan";

export type AdapterListResponse = { adapters: Adapter[] };
export type AdapterProvenance = { adapter: Adapter; lineage: Adapter[]; activation_history: AdapterActivationResponse[] };
export type AdapterEvaluateResponse = { adapter_id: string; score: number; decision: string; result_path: string; status: string };
export type AdapterActivationResponse = { action: string; adapter_id: string | null; adapter_hash: string | null; previous_adapter_id: string | null; previous_adapter_hash: string | null; activation_sequence: number };
export type AdapterRuntimeState = { base_model: string; adapter_id: string | null; adapter_hash: string | null; activation_sequence: number | null };
export type AdapterBehavioralStatus = {
  adapter_id: string;
  policy: "real_model_required" | "deterministic_runtime_only" | "disabled";
  ordinary_gates: Record<string, "passed" | "failed" | "not_run">;
  deterministic_status: "not_run" | "failed" | "stale" | "corrupt" | "hash_mismatch" | "coverage_incomplete" | "passed";
  deterministic_coverage: "complete" | "incomplete" | "not_evaluated";
  deterministic_artifact: BehavioralArtifactStatus;
  real_status: "not_run" | "failed" | "stale" | "corrupt" | "hash_mismatch" | "coverage_incomplete" | "passed";
  real_coverage: "complete" | "incomplete" | "not_evaluated";
  real_required: boolean;
  real_artifact: BehavioralArtifactStatus;
  activation_eligible: boolean;
  activation_reason: string;
  identity_integrity_status: "passed" | "failed" | "not_evaluated" | "stale";
  real_model_identity_integrity_status: "passed" | "failed" | "not_evaluated" | "stale";
  candidate_boundary_probe_choice: string | null;
  candidate_boundary_probe_margin: number | null;
  candidate_boundary_probe_count: number;
  rollback_reason: string | null;
};
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
export type BehavioralEvaluationSummary = {
  evaluation_id: string;
  baseline_id: string;
  candidate_id: string;
  baseline_score: number;
  candidate_score: number;
  baseline_dimensions: Record<string, number>;
  candidate_dimensions: Record<string, number>;
  dimension_deltas: Record<string, number>;
  activation_gate_passed: boolean;
  regression_dimensions: string[];
  threshold_failure_dimensions: string[];
  hard_gate_failures: string[];
  tool_execution_dimensions_complete: boolean;
  created_at: string;
  runtime_kind: "synthetic_evaluator_contract" | "deterministic_runtime" | "real_model_runtime";
  source_commit_sha: string | null;
  adapter_hash: string | null;
  base_model_revision: string | null;
  fixture_set_hash: string | null;
  deterministic_runtime_gate_passed: boolean;
  real_model_runtime_gate_passed: boolean;
  activation_eligibility: string;
  evaluation_state: "pending" | "running" | "prepared" | "finalized" | "reconciled" | "failed";
  failure_code: string | null;
  source_integrity: "verified" | "unknown" | "dirty";
  model_integrity: "verified" | "unknown" | "mismatch";
  artifact_integrity: string;
};
export type BehavioralEvaluationHistoryResponse = { results: BehavioralEvaluationSummary[] };
export type BehavioralEvaluationDetail = { evaluation_id: string; payload: Record<string, unknown> };
export type BehavioralFailureArtifact = { evaluation_id: string; scenario_id: string; payload: Record<string, unknown> };
export type BehavioralRerunResponse = { source_evaluation_id: string; evaluation_id: string; fixture_hashes_match: boolean; activation_gate_passed: boolean };
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
  message_id: string;
  kind: "question" | "approval_request" | "commitment_deadline" | "goal_state" | "action_result" | "anomaly" | "renegotiation" | "long_task_complete";
  title: string;
  urgency: "low" | "normal" | "high" | "critical";
  delivery_status: "pending" | "delivered" | "failed" | "expired" | "cancelled";
  acknowledgment_status: "unacknowledged" | "read" | "replied" | "approved" | "rejected";
  created_at: string;
  channel: "local";
  privacy_class: "public" | "interlocutor" | "operator";
  last_failure_code: string | null;
  body_preview: string | null;
  references: { event_id: string | null; goal_id: string | null; plan_id: string | null; decision_id: string | null; action_id: string | null; commitment_id: string | null };
};
export type OutboxMessageListResponse = { messages: OutboxMessage[] };

export type RiskClass = "read_only" | "reversible_write" | "external_write" | "destructive" | "high_impact";
export type ActionTool = { name: string; risk_class: RiskClass; approval_required: boolean; reversible: boolean; effect_code: string; validation_schema_revision: string; enabled: boolean; executable: boolean; execution_authority: "action_execution" };
export type RegistryTool = { name: string; description: string | null; tool_type: string; status: string; generated: boolean; human_approved: boolean; execution_authority: "registry_only" };
export type ArgumentSummary =
  | { kind: "metadata_read"; namespace: string; key: string }
  | { kind: "document_search"; scope_kind: string; max_results: number; query_length: number }
  | { kind: "calendar_read"; starts_at: string; ends_at: string; max_results: number }
  | { kind: "notification"; channel: string; title: string; body_preview: string };
export type OperatorAction = {
  intent_id: string; revision: number; status: "awaiting_approval" | "approved" | "dry_run" | "executing" | "retry_pending" | "succeeded" | "failed" | "cancelled" | "rejected" | "compensated"; approval: { approval_id: string; status: "pending" | "approved" | "rejected"; requested_at: string } | null;
  tool: ActionTool; argument_summary: ArgumentSummary; policy: { allowed: boolean; approval_required: boolean; reason_codes: string[] };
  preview: { effect_code: string; effect: string; digest: string; compensation_available: boolean };
  budget: { max_attempts: number; max_cost_units: number; max_monetary_cost: number; deadline_at: string | null; attempts: number; cost_units_used: number; retry_at: string | null };
  provenance: { decision_id: string; plan_id: string | null; plan_revision: number | null; step_id: string | null; triggering_event_id: string | null };
  receipt: { receipt_id: string; status: string } | null; verification: { verification_id: string; success: boolean; reason: string } | null;
  idempotency_state: "reserved" | "released" | "completed" | "unknown"; available_commands: Array<"approve" | "reject" | "cancel" | "retry_now" | "compensate">;
  confirmation: { required: true; phrase: string } | null;
};
export type ActionOperatorSummary = { pending_approval_count: number; operator_action_count: number; risk_ceiling: RiskClass; actions: OperatorAction[]; action_tools: ActionTool[]; registry_tools: RegistryTool[] };
export type OperatorSummary = ActionOperatorSummary;
export type ActionMutationCommon = { expected_intent_revision: number; expected_preview_digest: string; confirmation_phrase?: string };
export type ApproveActionRequest = ActionMutationCommon & { expected_approval_id: string; reason?: string };
export type ActionMutationResponse = { command: "approve" | "reject" | "cancel" | "retry_now" | "compensate"; event_id: string; processing_sequence: number; action: OperatorAction; disposition: "awaiting_scheduler" | "rejected" | "cancelled" | "executed" | "compensated" };
export type CockpitOutboxMessage = {
  message_id: string;
  title: string;
  urgency: "low" | "normal" | "high" | "critical";
  delivery_status: "pending" | "delivered" | "failed" | "expired" | "cancelled";
  acknowledgment_status: "unacknowledged" | "read" | "replied" | "approved" | "rejected";
  references: {
    event_id: string | null;
    goal_id: string | null;
    plan_id: string | null;
    decision_id: string | null;
    action_id: string | null;
    commitment_id: string | null;
  };
};
export type CockpitOutboxResponse = {
  pending_count: number;
  critical_count: number;
  messages: CockpitOutboxMessage[];
};
export type CockpitActionProvenance = {
  decision_id: string;
  candidate_id: string;
  triggering_event_id: string | null;
  plan_id: string | null;
  plan_revision: number | null;
  step_id: string | null;
};
export type CockpitActionApproval = {
  approval_id: string | null;
  status: "pending" | "approved" | "rejected" | null;
  requested_at: string | null;
  resolved_at: string | null;
  resolved_by_operator: boolean;
};
export type CockpitActionReceipt = {
  receipt_id: string;
  status: "succeeded" | "failed" | "timed_out" | "cancelled" | "compensated";
  attempt: number;
  duration_ms: number;
  event_id: string | null;
  event_sequence: number | null;
  error_code: string | null;
  compensation_of: string | null;
};
export type CockpitActionRelatedReceipt = {
  receipt_id: string;
  status: CockpitActionReceipt["status"];
};
export type CockpitActionObservation = {
  observation_id: string;
  valid: boolean;
  validation_errors: string[];
  result_digest: string;
};
export type CockpitActionVerification = {
  verification_id: string;
  success: boolean;
  reason: string;
};
export type CockpitActionTrace = {
  intent_id: string;
  revision: number;
  tool_name: string;
  risk_class: "read_only" | "reversible_write" | "external_write" | "destructive" | "high_impact";
  status: "awaiting_approval" | "approved" | "dry_run" | "executing" | "retry_pending" | "succeeded" | "failed" | "cancelled" | "rejected" | "compensated";
  dry_run: boolean;
  created_at: string;
  updated_at: string;
  failure_code: string | null;
  provenance: CockpitActionProvenance;
  approval: CockpitActionApproval;
  receipt: CockpitActionReceipt | null;
  related_receipts: CockpitActionRelatedReceipt[];
  observation: CockpitActionObservation | null;
  verification: CockpitActionVerification | null;
};
export type CockpitPreIntentFailure = {
  failure_id: string;
  failure_type: "validation" | "policy_rejection";
  decision_id: string | null;
  candidate_id: string | null;
  tool_name: string | null;
  risk_class: CockpitActionTrace["risk_class"] | null;
  error_codes: string[];
  event_id: string;
  event_sequence: number;
  occurred_at: string;
};
export type CockpitActionTraceResponse = {
  pending_approval_count: number;
  retry_pending_count: number;
  failed_count: number;
  traces: CockpitActionTrace[];
  pre_intent_failures: CockpitPreIntentFailure[];
};

export type CockpitTrainingNode = {
  node_id: string;
  role: "inference" | "worker";
  backend: string;
  status: "online" | "unavailable";
  last_contact_at: string | null;
  expected_model_id: string | null;
  expected_model_revision: string | null;
  expected_processor_revision: string | null;
  observed_model_id: string | null;
  observed_model_revision: string | null;
  model_matches_expected: boolean | null;
  gpu_name: string | null;
  cuda_version: string | null;
  driver_version: string | null;
};
export type CockpitTrainingFailureCode = "worker_unavailable" | "ssh_failed" | "transfer_failed" | "timeout" | "cancelled" | "bundle_invalid" | "result_invalid" | "training_failed" | "cuda_oom" | "non_finite_metrics" | "import_failed" | "evaluation_failed" | "unknown_failure";
export type CockpitTrainingJob = {
  job_id: string;
  attempt_id: string;
  status: "preparing" | "ready" | "dispatched" | "running" | "succeeded" | "importing" | "completed" | "failed" | "cancelled" | "unavailable";
  backend: string | null;
  created_at: string | null;
  updated_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  source_event_start: number | null;
  source_event_end: number | null;
  selected_episode_count: number | null;
  remote_job_id: string | null;
  worker_node_id: string | null;
  retry_count: number | null;
  transferred_bytes: number | null;
  failure_code: CockpitTrainingFailureCode | null;
  candidate_adapter_id: string | null;
  import_status: "not_started" | "importing" | "completed" | "failed" | "unavailable";
  bundle_digest: string | null;
  result_digest: string | null;
};
export type CockpitAdapterLineage = {
  adapter_id: string;
  status: string;
  adapter_hash: string | null;
  base_model_id: string | null;
  base_model_revision: string | null;
  parent_adapter_id: string | null;
  training_job_id: string | null;
  training_node_id: string | null;
  submitted_by_node_id: string | null;
  imported_by_node_id: string | null;
  evaluation_id: string | null;
  evaluation_status: "passed" | "failed" | "stale" | "corrupt" | "unavailable";
  approved: boolean;
  active: boolean;
  rollback_candidate: boolean;
  activation_event_id: string | null;
  activation_event_sequence: number | null;
  rollback_event_id: string | null;
  rollback_event_sequence: number | null;
};
export type CockpitTrainingSummary = {
  node_count: number;
  online_node_count: number;
  running_job_count: number;
  failed_job_count: number;
  importing_job_count: number;
  active_adapter_count: number;
  candidate_adapter_count: number;
  nodes: CockpitTrainingNode[];
  jobs: CockpitTrainingJob[];
  adapters: CockpitAdapterLineage[];
};

export type ContextFrame = {
  context_id: string;
  context_type: string;
  source_channel: string;
  source_session_id: string | null;
  participant_ids: string[];
  active_topic: string | null;
  active_task: string | null;
  status: "active" | "suspended" | "ended";
};
export type ContextListResponse = { contexts: ContextFrame[] };

export type Goal = {
  goal_id: string;
  goal_type: "intrinsic" | "external_request" | "commitment";
  description: string;
  priority: number;
  urgency: number;
  confidence: number;
  origin: string;
  status: "candidate" | "active" | "suspended" | "completed" | "abandoned" | "failed";
  dependency_ids: string[];
  conflict_ids: string[];
  deadline: string | null;
  needs_information: boolean;
  created_at: string;
  updated_at: string;
};
export type GoalDecision = {
  decision_id: string;
  action: "activate" | "suspend" | "resume" | "defer" | "request_information" | "no_action";
  goal_id: string | null;
  score: number | null;
  reasons: string[];
  conflicting_goal_ids: string[];
  created_at: string;
};
export type GoalListResponse = { goals: Goal[]; decisions: GoalDecision[] };

export type Commitment = {
  commitment_id: string;
  description: string;
  related_goal_id: string | null;
  status: "proposed" | "active" | "renegotiating" | "fulfilled" | "released" | "breached";
  beneficiary: string;
  scope: string;
  deadline: string | null;
  cost: number;
  burden: number;
  fulfillability: "unknown" | "fulfillable" | "at_risk" | "impossible";
  fulfillability_reason: string | null;
  decision_refs: string[];
  created_at: string;
  updated_at: string;
};
export type CommitmentListResponse = { commitments: Commitment[] };

export type PlanStep = {
  step_id: string;
  action_type: "respond" | "internal" | "no_op" | "defer" | "observe" | "request_information" | "delegate" | "refuse" | "unable" | "replan";
  action_code: string;
  dependency_ids: string[];
  status: "pending" | "ready" | "in_progress" | "waiting_retry" | "completed" | "failed" | "cancelled";
  attempt_count: number;
  started_at: string | null;
  retry_at: string | null;
  completed_at: string | null;
};
export type Plan = {
  plan_id: string;
  goal_id: string;
  revision: number;
  status: "draft" | "active" | "paused" | "completed" | "failed" | "abandoned";
  steps: PlanStep[];
  created_at: string;
  updated_at: string;
};
export type PlanListResponse = { plans: Plan[] };

export type DecisionCandidate = {
  candidate_id: string;
  candidate_type: PlanStep["action_type"];
  proposed_action: string;
  plan_id: string | null;
  plan_revision: number | null;
  step_id: string | null;
  goal_refs: string[];
  commitment_refs: string[];
};
export type Decision = {
  decision_id: string;
  context_id: string | null;
  active_goal_ids: string[];
  selected_candidate_id: string;
  selected_candidate: DecisionCandidate;
  selection_confidence: number;
  status: "awaiting_outcome" | "resolved";
  outcome_status: "pending" | "succeeded" | "failed" | "compensated";
  created_at: string;
  updated_at: string;
};
export type DecisionListResponse = { decisions: Decision[] };

export type WorkingMemorySummary = {
  item_count: number;
  token_count: number;
  item_capacity: number;
  token_capacity: number;
};

export type OperatorRestoreTarget = {
  target_sequence: number;
  target_snapshot_hash: string;
  checkpoint_kind: "bootstrap" | "journal_completed" | "journal_recovered" | "checkpoint";
  timestamp: string;
  event_type: string | null;
  eligible: boolean;
  reason_codes: string[];
};
export type OperatorRestoreOperation = {
  operation_id: string; target_sequence: number; target_snapshot_hash: string; preview_digest: string;
  requested_at: string; started_at: string | null; completed_at: string | null; event_id: string;
  processing_sequence: number | null; state: "previewed" | "finalizing" | "completed" | "failed" | "commit_indeterminate";
  error_code: string | null; external_side_effects_replayed: false;
};
export type OperatorRestoreSummary = {
  schema_version: 1; current_sequence: number; current_snapshot_hash: string; current_logical_digest: string;
  semantic_revision: number; retained_min_sequence: number; retained_max_sequence: number;
  targets: OperatorRestoreTarget[]; latest_operation: OperatorRestoreOperation | null;
  external_side_effects_replayed: false;
};
export type OperatorRestorePreviewDomain = {
  domain: "emotion_state" | "working_memory" | "motivation" | "identity" | "context_state" | "appraisal" | "experience" | "belief" | "attention" | "decision" | "decision_explanation" | "agency_attribution" | "counterfactual" | "feedback" | "metacognition" | "action_execution" | "proactive_outbox" | "subject_scheduler" | "extensions"; before_count: number; after_count: number; added_count: number; removed_count: number;
  changed_count: number; changed_revision_count: number; newer_state_loss_count: number;
  refs: Array<{ kind: "goal" | "commitment" | "decision" | "plan" | "action" | "outbox" | "journal" | "experience" | "memory" | "belief"; id: string }>; truncated: boolean; reason_code: string | null;
};
export type OperatorRestorePreview = {
  schema_version: 1; operation_id: string; preview_digest: string; created_at: string; expires_at: string;
  current_logical_digest: string; semantic_revision: number; display_sequence: number; target_sequence: number;
  target_snapshot_hash: string; newer_authoritative_event_count: number; domains: OperatorRestorePreviewDomain[];
  external_effects: { consistency_status: "consistent" | "inconsistent"; artifacts: Array<{ artifact_type: "memory" | "dataset" | "adapter" | "outbox" | "unknown"; count: number; refs: string[]; truncated: boolean }>;
    retained_not_replayed_count: number; pending_count: number; orphaned_count: number; retryable_count: number;
    effect_digest: string; external_side_effects_replayed: false };
  restoreable: boolean; reason_codes: string[]; external_side_effects_replayed: false; confirmation_phrase: string;
};
export type OperatorRestoreCommitRequest = {
  target_sequence: number; expected_target_hash: string; expected_semantic_revision: number;
  expected_current_logical_digest: string; expected_preview_digest: string;
  expected_external_effect_digest: string; confirmation_phrase: string;
};
export type OperatorRestoreCommitResponse = {
  command: "restore"; disposition: "completed" | "commit_indeterminate"; operation_id: string; event_id: string;
  processing_sequence: number; restored_target_sequence: number; restored_target_hash: string; post_restore_sequence: number;
  post_restore_hash: string; operation_status: "previewed" | "finalizing" | "completed" | "failed" | "commit_indeterminate";
  error_code: string | null; external_side_effects_replayed: false;
};

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

export class ChatJobCanceledError extends Error {
  constructor(readonly code: string) {
    super("Chat was canceled.");
    this.name = "ChatJobCanceledError";
  }
}

export class ChatJobFailedError extends Error {
  constructor(readonly code: string) {
    super(`Chat failed: ${code.replaceAll("_", " ")}.`);
    this.name = "ChatJobFailedError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${API_PROXY_BASE_URL}${path.replace(/^\/api/, "")}`, init);
}

async function privateApiRequest<T>(path: string, init?: RequestInit): Promise<T> {
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
    const message = formatPublicHttpError(response.status, response.statusText, detail);
    throw new ApiError(message, response.status, response.statusText, detail);
  }
  return response.json() as Promise<T>;
}

export async function streamChatJob(
  body: ChatRequest,
  callbacks: { status: (status: OperationStatus) => void; token: (text: string) => void },
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const clientId = getChatClientId();
  const idempotencyKey = crypto.randomUUID();
  let submission: ChatJobAccepted | ChatResponse | null = null;
  for (let attempt = 0; attempt < 3 && submission === null; attempt += 1) {
    try {
      submission = await request<ChatJobAccepted | ChatResponse>("/api/chat/jobs", {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey, "X-KAGYA-Client-ID": clientId },
        body: JSON.stringify(body),
        signal,
      });
    } catch (error) {
      if (signal?.aborted || attempt === 2) throw error;
    }
  }
  if (submission === null) throw new ApiError("Chat submission failed.");
  if (!("operation" in submission)) return submission;
  const accepted = submission;
  callbacks.status(accepted.operation);
  let lastEventId = 0;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      const response = await fetch(`${API_PROXY_BASE_URL}${accepted.events_url.replace(/^\/api/, "")}`, {
        headers: lastEventId ? { "Last-Event-ID": String(lastEventId) } : undefined,
        signal,
      });
      if (!response.ok || !response.body) continue;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { done, value } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const frames = pending.split("\n\n");
        pending = frames.pop() ?? "";
        for (const frame of frames) {
          if (!frame || frame.startsWith(":")) continue;
          const fields = Object.fromEntries(frame.split("\n").map((line) => {
            const separator = line.indexOf(":");
            return [line.slice(0, separator), line.slice(separator + 1).trimStart()];
          }));
          if (fields.id) lastEventId = Number(fields.id);
          if (!fields.data) continue;
          const data = JSON.parse(fields.data) as Record<string, unknown>;
          if (fields.event === "status") callbacks.status(data as OperationStatus);
           if (fields.event === "token" && typeof data.text === "string") callbacks.token(data.text);
           if (fields.event === "final") return data as ChatResponse;
           if (fields.event === "canceled") {
             throw new ChatJobCanceledError(typeof data.code === "string" ? data.code : "client_request");
           }
           if (fields.event === "error") {
             throw new ChatJobFailedError(typeof data.code === "string" ? data.code : "internal_error");
           }
        }
        if (done) break;
      }
    } catch (error) {
      if (signal?.aborted || error instanceof ChatJobCanceledError || error instanceof ChatJobFailedError) throw error;
    }
  }
  const final = await request<ChatJobResult>(accepted.result_url);
  return final.result;
}

function getChatClientId(): string {
  const key = "kagya-chat-client-id";
  const existing = globalThis.localStorage?.getItem(key);
  if (existing) return existing;
  const created = crypto.randomUUID();
  globalThis.localStorage?.setItem(key, created);
  return created;
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

function formatPublicHttpError(status: number, statusText: string, detail: string): string {
  if (status === 503) {
    return detail ? `Chat service is temporarily unavailable: ${detail}` : "Chat service is temporarily unavailable.";
  }
  if (status >= 500) {
    return detail ? `Backend failed: ${detail}` : `Backend failed with ${status} ${statusText}.`;
  }
  return detail ? `${status} ${statusText}: ${detail}` : `${status} ${statusText}`;
}

function parseDecisionExplanationResponse(value: unknown): { explanations: DecisionExplanation[] } {
  if (!isRecord(value) || !Array.isArray(value.explanations)) throw new ApiError("Backend returned an invalid decision explanation response.");
  for (const item of value.explanations) {
    if (!isRecord(item) || typeof item.explanation_id !== "string" || typeof item.decision_id !== "string" || typeof item.revision !== "number" || typeof item.decision_revision !== "number") throw new ApiError("Backend returned an invalid decision explanation response.");
    if (!isRecord(item.renderer) || !Array.isArray(item.renderer.offered_clause_ids) || !Array.isArray(item.renderer.ordered_clause_ids) || typeof item.renderer.visible_explanation !== "string") throw new ApiError("Backend returned an invalid decision explanation renderer.");
    if (!isRecord(item.risk) || !isRecord(item.outcome) || !isRecord(item.change) || !isRecord(item.selected)) throw new ApiError("Backend returned an invalid decision explanation projection.");
    for (const key of ["major_alternatives", "contributions", "evidence_refs", "uncertainty", "information_gap_codes", "tradeoff_refs", "conflict_codes", "reason_codes"] as const) {
      if (!Array.isArray(item[key])) throw new ApiError("Backend returned an invalid decision explanation projection.");
    }
    if (typeof item.omitted_reference_count !== "number") throw new ApiError("Backend returned an invalid decision explanation projection.");
  }
  return value as { explanations: DecisionExplanation[] };
}

function parseContexts(value: unknown): ContextListResponse {
  const items = responseArray(value, "contexts", "context");
  return { contexts: items.map((item) => ({
    context_id: id(item.context_id, "context"),
    context_type: text(item.context_type, "context"),
    source_channel: text(item.source_channel, "context"),
    source_session_id: optionalId(item.source_session_id, "context"),
    participant_ids: idArray(item.participant_ids, "context"),
    active_topic: optionalText(item.active_topic, "context"),
    active_task: optionalText(item.active_task, "context"),
    status: enumValue(item.status, ["active", "suspended", "ended"] as const, "context"),
  })) };
}

function parseGoals(value: unknown): GoalListResponse {
  if (!isRecord(value)) invalid("goal");
  const goals = recordArray(value.goals, "goal").map((item) => ({
    goal_id: id(item.goal_id, "goal"),
    goal_type: enumValue(item.goal_type, ["intrinsic", "external_request", "commitment"] as const, "goal"),
    description: text(item.description, "goal"),
    priority: finiteNumber(item.priority, "goal"),
    urgency: finiteNumber(item.urgency, "goal"),
    confidence: finiteNumber(item.confidence, "goal"),
    origin: identityOrigin(item.identity_origin, "goal"),
    status: enumValue(item.status, ["candidate", "active", "suspended", "completed", "abandoned", "failed"] as const, "goal"),
    dependency_ids: idArray(item.dependency_ids, "goal"),
    conflict_ids: idArray(item.conflict_ids, "goal"),
    deadline: optionalText(item.deadline, "goal"),
    needs_information: booleanValue(item.needs_information, "goal"),
    created_at: text(item.created_at, "goal"),
    updated_at: text(item.updated_at, "goal"),
  }));
  const decisions = recordArray(value.decisions, "goal decision").map((item) => ({
    decision_id: id(item.decision_id, "goal decision"),
    action: enumValue(item.action, ["activate", "suspend", "resume", "defer", "request_information", "no_action"] as const, "goal decision"),
    goal_id: optionalId(item.goal_id, "goal decision"),
    score: optionalNumber(item.score, "goal decision"),
    reasons: stringArray(item.reasons, "goal decision"),
    conflicting_goal_ids: idArray(item.conflicting_goal_ids, "goal decision"),
    created_at: text(item.created_at, "goal decision"),
  }));
  return { goals, decisions };
}

function parseCommitments(value: unknown): CommitmentListResponse {
  const items = responseArray(value, "commitments", "commitment");
  return { commitments: items.map((item) => ({
    commitment_id: id(item.commitment_id, "commitment"),
    description: text(item.description, "commitment"),
    related_goal_id: optionalId(item.related_goal_id, "commitment"),
    status: enumValue(item.status, ["proposed", "active", "renegotiating", "fulfilled", "released", "breached"] as const, "commitment"),
    beneficiary: text(item.beneficiary, "commitment"),
    scope: text(item.scope, "commitment"),
    deadline: optionalText(item.deadline, "commitment"),
    cost: finiteNumber(item.cost, "commitment"),
    burden: finiteNumber(item.burden, "commitment"),
    fulfillability: enumValue(item.fulfillability, ["unknown", "fulfillable", "at_risk", "impossible"] as const, "commitment"),
    fulfillability_reason: optionalText(item.fulfillability_reason, "commitment"),
    decision_refs: idArray(item.decision_refs, "commitment"),
    created_at: text(item.created_at, "commitment"),
    updated_at: text(item.updated_at, "commitment"),
  })) };
}

function parsePlans(value: unknown): PlanListResponse {
  const items = responseArray(value, "plans", "plan");
  return { plans: items.map((item) => {
    const planId = id(item.plan_id, "plan");
    const revision = positiveInteger(item.revision, "plan");
    const revisions = recordArray(item.revisions, "plan revision");
    const current = revisions.at(-1);
    if (!current || positiveInteger(current.revision, "plan revision") !== revision) invalid("plan revision");
    const definitions = recordArray(current.steps, "plan step");
    const states = recordArray(item.step_states, "plan step state");
    const stateById = new Map(states.map((state) => [id(state.step_id, "plan step state"), state]));
    const steps = definitions.map((step): PlanStep => {
      const stepId = id(step.step_id, "plan step");
      const state = stateById.get(stepId);
      if (!state) invalid("plan step state");
      return {
        step_id: stepId,
        action_type: enumValue(step.action_type, ACTION_TYPES, "plan step"),
        action_code: text(step.action_code, "plan step"),
        dependency_ids: idArray(step.dependency_ids, "plan step"),
        status: enumValue(state.status, ["pending", "ready", "in_progress", "waiting_retry", "completed", "failed", "cancelled"] as const, "plan step state"),
        attempt_count: nonnegativeInteger(state.attempt_count, "plan step state"),
        started_at: optionalText(state.started_at, "plan step state"),
        retry_at: optionalText(state.retry_at, "plan step state"),
        completed_at: optionalText(state.completed_at, "plan step state"),
      };
    });
    if (steps.length !== states.length) invalid("plan step state");
    return {
      plan_id: planId,
      goal_id: id(item.goal_id, "plan"),
      revision,
      status: enumValue(item.status, ["draft", "active", "paused", "completed", "failed", "abandoned"] as const, "plan"),
      steps,
      created_at: text(item.created_at, "plan"),
      updated_at: text(item.updated_at, "plan"),
    };
  }) };
}

const ACTION_TYPES = ["respond", "internal", "no_op", "defer", "observe", "request_information", "delegate", "refuse", "unable", "replan"] as const;

function parseDecisions(value: unknown): DecisionListResponse {
  const items = responseArray(value, "decisions", "decision");
  return { decisions: items.map((item) => {
    const candidates = recordArray(item.considered_candidates, "decision candidate").map((evaluation): DecisionCandidate => {
      if (!isRecord(evaluation.candidate)) invalid("decision candidate");
      const candidate = evaluation.candidate;
      const planId = optionalId(candidate.plan_id, "decision candidate");
      const planRevision = optionalPositiveInteger(candidate.plan_revision, "decision candidate");
      const stepId = optionalId(candidate.step_id, "decision candidate");
      if ([planId, planRevision, stepId].some((part) => part === null) && [planId, planRevision, stepId].some((part) => part !== null)) invalid("decision candidate reference");
      return {
        candidate_id: id(candidate.candidate_id, "decision candidate"),
        candidate_type: enumValue(candidate.candidate_type, ACTION_TYPES, "decision candidate"),
        proposed_action: text(candidate.proposed_action, "decision candidate"),
        plan_id: planId,
        plan_revision: planRevision,
        step_id: stepId,
        goal_refs: idArray(candidate.goal_refs, "decision candidate"),
        commitment_refs: idArray(candidate.commitment_refs, "decision candidate"),
      };
    });
    const selectedId = id(item.selected_candidate_id, "decision");
    const selected = candidates.find((candidate) => candidate.candidate_id === selectedId);
    if (!selected) invalid("decision selected candidate");
    const outcomeStatus = decisionOutcomeStatus(item.actual_outcome);
    return {
      decision_id: id(item.decision_id, "decision"),
      context_id: optionalId(item.context_id, "decision"),
      active_goal_ids: idArray(item.active_goal_ids, "decision"),
      selected_candidate_id: selectedId,
      selected_candidate: selected,
      selection_confidence: finiteNumber(item.selection_confidence, "decision"),
      status: enumValue(item.status, ["awaiting_outcome", "resolved"] as const, "decision"),
      outcome_status: outcomeStatus,
      created_at: text(item.created_at, "decision"),
      updated_at: text(item.updated_at, "decision"),
    };
  }) };
}

function parseWorkingMemory(value: unknown): WorkingMemorySummary {
  if (!isRecord(value)) invalid("working-memory summary");
  return {
    item_count: nonnegativeInteger(value.item_count, "working-memory summary"),
    token_count: nonnegativeInteger(value.token_count, "working-memory summary"),
    item_capacity: nonnegativeInteger(value.item_capacity, "working-memory summary"),
    token_capacity: nonnegativeInteger(value.token_capacity, "working-memory summary"),
  };
}

function boundedRestoreCodes(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || value.length > 64) invalid(label);
  return value.map((item) => boundedCode(item, label));
}
function restoreDigest(value: unknown, label: string): string { return digest(value, label); }
function restoreOperationId(value: unknown, label: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value)) invalid(label);
  return value;
}
function restoreOperationEvent(value: unknown, operationId: string, label: string): string {
  const expected = `operator-restore-${operationId}`;
  if (value !== expected) invalid(label);
  return expected;
}
function restoreArtifactReference(value: unknown, label: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) invalid(label);
  return value;
}
function restoreId(value: unknown, label: string, kind?: string): string { return publicReference(value, label, kind); }
function restoreCount(value: unknown, label: string): number { const result = nonnegativeInteger(value, label); if (result > 1_000_000) invalid(label); return result; }
function restoreSequence(value: unknown, label: string): number { return restoreCount(value, label); }
function restoreLimit(value: unknown): number { const result = nonnegativeInteger(value, "restore limit"); if (result > 1000) invalid("restore limit"); return result; }
function parseRestoreTarget(value: unknown): OperatorRestoreTarget {
  exactRecord(value, ["target_sequence", "target_snapshot_hash", "checkpoint_kind", "timestamp", "event_type", "eligible", "reason_codes"], "restore target");
  return { target_sequence: restoreSequence(value.target_sequence, "restore target sequence"), target_snapshot_hash: restoreDigest(value.target_snapshot_hash, "restore target hash"), checkpoint_kind: enumValue(value.checkpoint_kind, ["bootstrap", "journal_completed", "journal_recovered", "checkpoint"] as const, "restore checkpoint"), timestamp: isoTimestamp(value.timestamp, "restore timestamp"), event_type: value.event_type === null ? null : boundedCode(value.event_type, "restore event type"), eligible: booleanValue(value.eligible, "restore eligibility"), reason_codes: boundedRestoreCodes(value.reason_codes, "restore reason codes") };
}
function parseRestoreOperation(value: unknown): OperatorRestoreOperation {
  exactRecord(value, ["operation_id", "target_sequence", "target_snapshot_hash", "preview_digest", "requested_at", "started_at", "completed_at", "event_id", "processing_sequence", "state", "error_code", "external_side_effects_replayed"], "restore operation");
  if (value.external_side_effects_replayed !== false) invalid("restore operation replay flag");
  const operationId = restoreOperationId(value.operation_id, "restore operation ID");
  return { operation_id: operationId, target_sequence: restoreSequence(value.target_sequence, "restore operation target"), target_snapshot_hash: restoreDigest(value.target_snapshot_hash, "restore operation target hash"), preview_digest: restoreDigest(value.preview_digest, "restore operation digest"), requested_at: isoTimestamp(value.requested_at, "restore requested timestamp"), started_at: optionalIsoTimestamp(value.started_at, "restore started timestamp"), completed_at: optionalIsoTimestamp(value.completed_at, "restore completed timestamp"), event_id: restoreOperationEvent(value.event_id, operationId, "restore operation event"), processing_sequence: value.processing_sequence === null ? null : positiveInteger(value.processing_sequence, "restore processing sequence"), state: enumValue(value.state, ["previewed", "finalizing", "completed", "failed", "commit_indeterminate"] as const, "restore operation state"), error_code: value.error_code === null ? null : boundedCode(value.error_code, "restore operation error"), external_side_effects_replayed: false };
}
function parseOperatorRestoreSummary(value: unknown): OperatorRestoreSummary {
  exactRecord(value, ["schema_version", "current_sequence", "current_snapshot_hash", "current_logical_digest", "semantic_revision", "retained_min_sequence", "retained_max_sequence", "targets", "latest_operation", "external_side_effects_replayed"], "restore summary");
  if (value.schema_version !== 1 || value.external_side_effects_replayed !== false) invalid("restore summary");
  if (!Array.isArray(value.targets) || value.targets.length > 1000) invalid("restore targets");
  return { schema_version: 1, current_sequence: restoreSequence(value.current_sequence, "restore current sequence"), current_snapshot_hash: restoreDigest(value.current_snapshot_hash, "restore current hash"), current_logical_digest: restoreDigest(value.current_logical_digest, "restore logical digest"), semantic_revision: restoreSequence(value.semantic_revision, "restore semantic revision"), retained_min_sequence: restoreSequence(value.retained_min_sequence, "restore retained minimum"), retained_max_sequence: restoreSequence(value.retained_max_sequence, "restore retained maximum"), targets: value.targets.map(parseRestoreTarget), latest_operation: value.latest_operation === null ? null : parseRestoreOperation(value.latest_operation), external_side_effects_replayed: false };
}
function parseOperatorRestorePreview(value: unknown): OperatorRestorePreview {
  exactRecord(value, ["schema_version", "operation_id", "preview_digest", "created_at", "expires_at", "current_logical_digest", "semantic_revision", "display_sequence", "target_sequence", "target_snapshot_hash", "newer_authoritative_event_count", "domains", "external_effects", "restoreable", "reason_codes", "external_side_effects_replayed", "confirmation_phrase"], "restore preview");
  if (value.schema_version !== 1 || value.external_side_effects_replayed !== false) invalid("restore preview");
  if (!Array.isArray(value.domains) || value.domains.length > 256) invalid("restore domains");
  const domains = value.domains.map((raw) => { exactRecord(raw, ["domain", "before_count", "after_count", "added_count", "removed_count", "changed_count", "changed_revision_count", "newer_state_loss_count", "refs", "truncated", "reason_code"], "restore domain"); if (!Array.isArray(raw.refs) || raw.refs.length > 256) invalid("restore references"); return { domain: enumValue(raw.domain, ["emotion_state", "working_memory", "motivation", "identity", "context_state", "appraisal", "experience", "belief", "attention", "decision", "decision_explanation", "agency_attribution", "counterfactual", "feedback", "metacognition", "action_execution", "proactive_outbox", "subject_scheduler", "extensions"] as const, "restore domain name"), before_count: restoreCount(raw.before_count, "restore before count"), after_count: restoreCount(raw.after_count, "restore after count"), added_count: restoreCount(raw.added_count, "restore added count"), removed_count: restoreCount(raw.removed_count, "restore removed count"), changed_count: restoreCount(raw.changed_count, "restore changed count"), changed_revision_count: restoreCount(raw.changed_revision_count, "restore changed revision count"), newer_state_loss_count: restoreCount(raw.newer_state_loss_count, "restore loss count"), refs: raw.refs.map((ref) => { exactRecord(ref, ["kind", "id"], "restore reference"); const kind = enumValue(ref.kind, ["goal", "commitment", "decision", "plan", "action", "outbox", "journal", "experience", "memory", "belief"] as const, "restore reference kind"); return { kind, id: restoreId(ref.id, "restore reference", kind) }; }), truncated: booleanValue(raw.truncated, "restore domain truncation"), reason_code: raw.reason_code === null ? null : boundedCode(raw.reason_code, "restore domain reason") }; });
  const effects = value.external_effects; exactRecord(effects, ["consistency_status", "artifacts", "retained_not_replayed_count", "pending_count", "orphaned_count", "retryable_count", "effect_digest", "external_side_effects_replayed"], "restore external effects");
  if (effects.external_side_effects_replayed !== false || !Array.isArray(effects.artifacts) || effects.artifacts.length > 256) invalid("restore external effects");
  const artifacts = effects.artifacts.map((raw) => { exactRecord(raw, ["artifact_type", "count", "refs", "truncated"], "restore artifact"); if (!Array.isArray(raw.refs) || raw.refs.length > 256) invalid("restore artifact references"); return { artifact_type: enumValue(raw.artifact_type, ["memory", "dataset", "adapter", "outbox", "unknown"] as const, "restore artifact type"), count: restoreCount(raw.count, "restore artifact count"), refs: raw.refs.map((ref) => restoreArtifactReference(ref, "restore artifact reference")), truncated: booleanValue(raw.truncated, "restore artifact truncation") }; });
  const operationId = restoreOperationId(value.operation_id, "restore preview operation ID");
  return { schema_version: 1, operation_id: operationId, preview_digest: restoreDigest(value.preview_digest, "restore preview digest"), created_at: isoTimestamp(value.created_at, "restore preview created timestamp"), expires_at: isoTimestamp(value.expires_at, "restore preview expiry"), current_logical_digest: restoreDigest(value.current_logical_digest, "restore preview logical digest"), semantic_revision: restoreSequence(value.semantic_revision, "restore preview semantic revision"), display_sequence: restoreSequence(value.display_sequence, "restore display sequence"), target_sequence: restoreSequence(value.target_sequence, "restore target sequence"), target_snapshot_hash: restoreDigest(value.target_snapshot_hash, "restore target hash"), newer_authoritative_event_count: restoreCount(value.newer_authoritative_event_count, "restore newer event count"), domains, external_effects: { consistency_status: enumValue(effects.consistency_status, ["consistent", "inconsistent"] as const, "restore consistency"), artifacts, retained_not_replayed_count: restoreCount(effects.retained_not_replayed_count, "restore retained effects"), pending_count: restoreCount(effects.pending_count, "restore pending effects"), orphaned_count: restoreCount(effects.orphaned_count, "restore orphaned effects"), retryable_count: restoreCount(effects.retryable_count, "restore retryable effects"), effect_digest: restoreDigest(effects.effect_digest, "restore effect digest"), external_side_effects_replayed: false }, restoreable: booleanValue(value.restoreable, "restoreable"), reason_codes: boundedRestoreCodes(value.reason_codes, "restore reason codes"), external_side_effects_replayed: false, confirmation_phrase: publicPreviewText(value.confirmation_phrase, "restore confirmation phrase", 200) };
}
function parseOperatorRestoreCommit(value: unknown): OperatorRestoreCommitResponse {
  exactRecord(value, ["command", "disposition", "operation_id", "event_id", "processing_sequence", "restored_target_sequence", "restored_target_hash", "post_restore_sequence", "post_restore_hash", "operation_status", "error_code", "external_side_effects_replayed"], "restore commit");
  if (value.external_side_effects_replayed !== false) invalid("restore commit replay flag");
  const operationId = restoreOperationId(value.operation_id, "restore commit operation ID");
  return { command: enumValue(value.command, ["restore"] as const, "restore command"), disposition: enumValue(value.disposition, ["completed", "commit_indeterminate"] as const, "restore disposition"), operation_id: operationId, event_id: restoreOperationEvent(value.event_id, operationId, "restore commit event"), processing_sequence: positiveInteger(value.processing_sequence, "restore commit sequence"), restored_target_sequence: restoreSequence(value.restored_target_sequence, "restored target sequence"), restored_target_hash: restoreDigest(value.restored_target_hash, "restored target hash"), post_restore_sequence: restoreSequence(value.post_restore_sequence, "post restore sequence"), post_restore_hash: restoreDigest(value.post_restore_hash, "post restore hash"), operation_status: enumValue(value.operation_status, ["previewed", "finalizing", "completed", "failed", "commit_indeterminate"] as const, "restore operation status"), error_code: value.error_code === null ? null : boundedCode(value.error_code, "restore commit error"), external_side_effects_replayed: false };
}

function prepareOperatorRestoreCommit(value: OperatorRestoreCommitRequest): OperatorRestoreCommitRequest {
  exactRecord(value, ["target_sequence", "expected_target_hash", "expected_semantic_revision", "expected_current_logical_digest", "expected_preview_digest", "expected_external_effect_digest", "confirmation_phrase"], "restore commit request");
  return { target_sequence: restoreSequence(value.target_sequence, "restore target sequence"), expected_target_hash: restoreDigest(value.expected_target_hash, "restore expected target hash"), expected_semantic_revision: restoreSequence(value.expected_semantic_revision, "restore expected semantic revision"), expected_current_logical_digest: restoreDigest(value.expected_current_logical_digest, "restore expected logical digest"), expected_preview_digest: restoreDigest(value.expected_preview_digest, "restore expected preview digest"), expected_external_effect_digest: restoreDigest(value.expected_external_effect_digest, "restore expected external effect digest"), confirmation_phrase: publicPreviewText(value.confirmation_phrase, "restore confirmation phrase", 200) };
}

async function restoreRequest<T>(path: string, init?: RequestInit): Promise<T> {
  try { return await privateApiRequest<T>(path, init); }
  catch (error) {
    if (error instanceof ApiError && error.status !== null) {
      const code = restoreErrorCode(error.detail) ?? (error.status === 409 ? "conflict" : error.status === 404 ? "not_found" : error.status === 422 ? "invalid_request" : error.status >= 500 ? "backend_failure" : "request_failed");
      throw new ApiError(`Restore request failed (${error.status}).`, error.status, null, code);
    }
    throw error;
  }
}

const RESTORE_ERROR_CODES = new Set([
  "restore_target_not_retained", "restore_target_unverified", "restore_wal_integrity_invalid",
  "restore_journal_integrity_invalid", "restore_checkpoint_mismatch", "restore_preview_stale",
  "restore_preview_expired", "restore_confirmation_required", "restore_external_state_inconsistent",
  "restore_unsupported_domain", "restore_operation_in_progress", "restore_not_authoritative",
  "commit_indeterminate", "operator_restore_contract_required", "private_state_projection_unavailable",
]);

function restoreErrorCode(detail: string | null): string | null {
  if (!detail) return null;
  try {
    const value = JSON.parse(detail) as unknown;
    if (!isRecord(value) || !isRecord(value.detail) || typeof value.detail.code !== "string") return null;
    return RESTORE_ERROR_CODES.has(value.detail.code) ? value.detail.code : null;
  } catch {
    return RESTORE_ERROR_CODES.has(detail) ? detail : null;
  }
}

const ACTION_STATUSES = ["awaiting_approval", "approved", "dry_run", "executing", "retry_pending", "succeeded", "failed", "cancelled", "rejected", "compensated"] as const;
const ACTION_TOOL_NAMES = ["restricted_metadata_read", "document_search", "calendar_read", "local_notification_enqueue"] as const;

function parseOperatorSummary(value: unknown): ActionOperatorSummary {
  exactRecord(value, ["pending_approval_count", "operator_action_count", "risk_ceiling", "actions", "action_tools", "registry_tools"], "operator summary");
  return {
    pending_approval_count: nonnegativeInteger(value.pending_approval_count, "operator summary pending count"),
    operator_action_count: nonnegativeInteger(value.operator_action_count, "operator summary action count"),
    risk_ceiling: enumValue(value.risk_ceiling, ["read_only", "reversible_write", "external_write", "destructive", "high_impact"] as const, "operator summary risk ceiling"),
    actions: recordArray(value.actions, "operator action").map(parseOperatorAction),
    action_tools: recordArray(value.action_tools, "action tool").map(parseActionTool),
    registry_tools: recordArray(value.registry_tools, "registry tool").map(parseRegistryTool),
  };
}

function parseActionTool(value: Record<string, unknown>): ActionTool {
  exactRecord(value, ["name", "risk_class", "approval_required", "reversible", "effect_code", "validation_schema_revision", "enabled", "executable", "execution_authority"], "action tool");
  return { name: enumValue(value.name, ACTION_TOOL_NAMES, "action tool name"), risk_class: enumValue(value.risk_class, ["read_only", "reversible_write", "external_write", "destructive", "high_impact"] as const, "action tool risk"), approval_required: booleanValue(value.approval_required, "action tool approval"), reversible: booleanValue(value.reversible, "action tool reversible"), effect_code: boundedCode(value.effect_code, "action tool effect code"), validation_schema_revision: digest(value.validation_schema_revision, "action tool schema revision"), enabled: booleanValue(value.enabled, "action tool enabled"), executable: booleanValue(value.executable, "action tool executable"), execution_authority: enumValue(value.execution_authority, ["action_execution"] as const, "action tool authority") };
}

function parseRegistryTool(value: Record<string, unknown>): RegistryTool {
  exactRecord(value, ["name", "description", "tool_type", "status", "generated", "human_approved", "execution_authority"], "registry tool");
  return { name: safeText(value.name, "registry tool name"), description: value.description === null ? null : publicPreviewText(value.description, "registry tool description", 160), tool_type: boundedCode(value.tool_type, "registry tool type"), status: boundedCode(value.status, "registry tool status"), generated: booleanValue(value.generated, "registry tool generated"), human_approved: booleanValue(value.human_approved, "registry tool approval"), execution_authority: enumValue(value.execution_authority, ["registry_only"] as const, "registry tool authority") };
}

function parseOperatorAction(value: Record<string, unknown>): OperatorAction {
  exactRecord(value, ["intent_id", "revision", "status", "approval", "tool", "argument_summary", "policy", "preview", "budget", "provenance", "receipt", "verification", "idempotency_state", "available_commands", "confirmation"], "operator action");
  if (!isRecord(value.tool) || !isRecord(value.policy) || !isRecord(value.preview) || !isRecord(value.budget) || !isRecord(value.provenance)) invalid("operator action");
  exactRecord(value.policy, ["allowed", "approval_required", "reason_codes"], "action policy");
  exactRecord(value.preview, ["effect_code", "effect", "digest", "compensation_available"], "action preview");
  exactRecord(value.budget, ["max_attempts", "max_cost_units", "max_monetary_cost", "deadline_at", "attempts", "cost_units_used", "retry_at"], "action budget");
  exactRecord(value.provenance, ["decision_id", "plan_id", "plan_revision", "step_id", "triggering_event_id"], "action provenance");
  const approval = value.approval === null ? null : parseApproval(value.approval);
  const receipt = value.receipt === null ? null : parseSmallReceipt(value.receipt);
  const verification = value.verification === null ? null : parseVerification(value.verification);
  if (value.confirmation !== null && (!isRecord(value.confirmation))) invalid("action confirmation");
  if (value.confirmation !== null) exactRecord(value.confirmation, ["required", "phrase"], "action confirmation");
  if (value.confirmation !== null && booleanValue(value.confirmation.required, "action confirmation required") !== true) invalid("action confirmation required");
  const confirmation = value.confirmation === null ? null : {
    required: true as const,
    phrase: safeText(value.confirmation.phrase, "action confirmation phrase"),
  };
  return {
    intent_id: safeId(value.intent_id, "action intent"), revision: positiveInteger(value.revision, "action revision"), status: enumValue(value.status, ACTION_STATUSES, "action status"), approval,
    tool: parseActionTool(value.tool), argument_summary: parseArgumentSummary(value.argument_summary),
    policy: { allowed: booleanValue(value.policy.allowed, "action policy allowed"), approval_required: booleanValue(value.policy.approval_required, "action policy approval"), reason_codes: stringArray(value.policy.reason_codes, "action policy codes").map((code) => boundedCode(code, "action policy code")) },
    preview: { effect_code: boundedCode(value.preview.effect_code, "action preview effect"), effect: safeText(value.preview.effect, "action preview"), digest: digest(value.preview.digest, "action preview digest"), compensation_available: booleanValue(value.preview.compensation_available, "action compensation") },
    budget: { max_attempts: positiveInteger(value.budget.max_attempts, "action budget attempts"), max_cost_units: nonnegativeNumber(value.budget.max_cost_units, "action budget cost"), max_monetary_cost: nonnegativeNumber(value.budget.max_monetary_cost, "action budget monetary cost"), deadline_at: optionalIsoTimestamp(value.budget.deadline_at, "action budget deadline"), attempts: nonnegativeInteger(value.budget.attempts, "action attempts"), cost_units_used: nonnegativeNumber(value.budget.cost_units_used, "action cost used"), retry_at: optionalIsoTimestamp(value.budget.retry_at, "action retry timestamp") },
    provenance: { decision_id: safeId(value.provenance.decision_id, "action decision"), plan_id: optionalSafeId(value.provenance.plan_id, "action plan"), plan_revision: optionalPositiveInteger(value.provenance.plan_revision, "action plan revision"), step_id: optionalSafeId(value.provenance.step_id, "action step"), triggering_event_id: optionalSafeId(value.provenance.triggering_event_id, "action event") },
    receipt, verification, idempotency_state: enumValue(value.idempotency_state, ["reserved", "released", "completed", "unknown"] as const, "action idempotency state"), available_commands: enumArray(value.available_commands, ["approve", "reject", "cancel", "retry_now", "compensate"] as const, "action commands"),
    confirmation,
  };
}

function parseArgumentSummary(value: unknown): ArgumentSummary {
  if (!isRecord(value) || typeof value.kind !== "string") invalid("action argument summary");
  switch (value.kind) {
    case "metadata_read": exactRecord(value, ["kind", "namespace", "key"], "metadata argument summary"); return { kind: value.kind, namespace: safeText(value.namespace, "metadata namespace"), key: safeText(value.key, "metadata key") };
    case "document_search": exactRecord(value, ["kind", "scope_kind", "max_results", "query_length"], "document argument summary"); return { kind: value.kind, scope_kind: boundedCode(value.scope_kind, "document scope"), max_results: positiveInteger(value.max_results, "document max results"), query_length: nonnegativeInteger(value.query_length, "document query length") };
    case "calendar_read": exactRecord(value, ["kind", "starts_at", "ends_at", "max_results"], "calendar argument summary"); return { kind: value.kind, starts_at: isoTimestamp(value.starts_at, "calendar start"), ends_at: isoTimestamp(value.ends_at, "calendar end"), max_results: positiveInteger(value.max_results, "calendar max results") };
    case "notification": exactRecord(value, ["kind", "channel", "title", "body_preview"], "notification argument summary"); return { kind: value.kind, channel: boundedCode(value.channel, "notification channel"), title: publicPreviewText(value.title, "notification title", 120), body_preview: publicPreviewText(value.body_preview, "notification preview", 160) };
    default: return invalid("action argument summary");
  }
}

function parseApproval(value: unknown): NonNullable<OperatorAction["approval"]> { if (!isRecord(value)) invalid("action approval"); exactRecord(value, ["approval_id", "status", "requested_at"], "action approval"); return { approval_id: safeId(value.approval_id, "approval"), status: enumValue(value.status, ["pending", "approved", "rejected"] as const, "approval status"), requested_at: isoTimestamp(value.requested_at, "approval timestamp") }; }
function parseSmallReceipt(value: unknown): NonNullable<OperatorAction["receipt"]> { if (!isRecord(value)) invalid("action receipt"); exactRecord(value, ["receipt_id", "status"], "action receipt"); return { receipt_id: safeId(value.receipt_id, "receipt"), status: enumValue(value.status, ["succeeded", "failed", "timed_out", "cancelled", "compensated"] as const, "receipt status") }; }
function parseVerification(value: unknown): NonNullable<OperatorAction["verification"]> { if (!isRecord(value)) invalid("action verification"); exactRecord(value, ["verification_id", "success", "reason"], "action verification"); return { verification_id: safeId(value.verification_id, "verification"), success: booleanValue(value.success, "verification success"), reason: boundedCode(value.reason, "verification reason") }; }
function enumArray<const T extends readonly string[]>(value: unknown, allowed: T, label: string): T[number][] { if (!Array.isArray(value)) invalid(label); return value.map((item) => enumValue(item, allowed, label)); }
function parseOutboxMessage(value: unknown): OutboxMessage { return parseOutbox({ messages: [value] }).messages[0]; }
function parseOutbox(value: unknown): OutboxMessageListResponse { if (!isRecord(value)) invalid("outbox"); exactRecord(value, ["messages"], "outbox"); return { messages: recordArray(value.messages, "outbox message").map((message) => { exactRecord(message, ["message_id", "kind", "title", "urgency", "delivery_status", "acknowledgment_status", "created_at", "channel", "privacy_class", "last_failure_code", "body_preview", "references"], "outbox message"); if (!isRecord(message.references)) invalid("outbox references"); exactRecord(message.references, ["event_id", "goal_id", "plan_id", "decision_id", "action_id", "commitment_id"], "outbox references"); const kind = enumValue(message.kind, ["question", "approval_request", "commitment_deadline", "goal_state", "action_result", "anomaly", "renegotiation", "long_task_complete"] as const, "outbox kind"); const bodyPreview = message.body_preview === null ? null : publicPreviewText(message.body_preview, "outbox body preview", 160); if (kind !== "question" && kind !== "renegotiation" && bodyPreview !== null) invalid("outbox body preview"); return { message_id: safeId(message.message_id, "outbox message"), kind, title: safeText(message.title, "outbox title"), urgency: enumValue(message.urgency, ["low", "normal", "high", "critical"] as const, "outbox urgency"), delivery_status: enumValue(message.delivery_status, ["pending", "delivered", "failed", "expired", "cancelled"] as const, "outbox delivery"), acknowledgment_status: enumValue(message.acknowledgment_status, ["unacknowledged", "read", "replied", "approved", "rejected"] as const, "outbox acknowledgment"), created_at: isoTimestamp(message.created_at, "outbox timestamp"), channel: enumValue(message.channel, ["local"] as const, "outbox channel"), privacy_class: enumValue(message.privacy_class, ["public", "interlocutor", "operator"] as const, "outbox privacy"), last_failure_code: message.last_failure_code === null ? null : boundedCode(message.last_failure_code, "outbox failure"), body_preview: bodyPreview, references: { event_id: optionalSafeId(message.references.event_id, "outbox event"), goal_id: optionalSafeId(message.references.goal_id, "outbox goal"), plan_id: optionalSafeId(message.references.plan_id, "outbox plan"), decision_id: optionalSafeId(message.references.decision_id, "outbox decision"), action_id: optionalSafeId(message.references.action_id, "outbox action"), commitment_id: optionalSafeId(message.references.commitment_id, "outbox commitment") } }; }) }; }

function parseCockpitOutbox(value: unknown): CockpitOutboxResponse {
  if (!isRecord(value) || !Array.isArray(value.messages)) invalid("cockpit outbox");
  return {
    pending_count: nonnegativeInteger(value.pending_count, "cockpit outbox pending count"),
    critical_count: nonnegativeInteger(value.critical_count, "cockpit outbox critical count"),
    messages: value.messages.map((message) => {
    if (!isRecord(message) || !isRecord(message.references)) invalid("cockpit outbox message");
    return {
      message_id: id(message.message_id, "cockpit outbox message"),
      title: text(message.title, "cockpit outbox message title"),
      urgency: enumValue(message.urgency, ["low", "normal", "high", "critical"] as const, "cockpit outbox urgency"),
      delivery_status: enumValue(message.delivery_status, ["pending", "delivered", "failed", "expired", "cancelled"] as const, "cockpit outbox delivery"),
      acknowledgment_status: enumValue(message.acknowledgment_status, ["unacknowledged", "read", "replied", "approved", "rejected"] as const, "cockpit outbox acknowledgment"),
      references: {
        event_id: optionalId(message.references.event_id, "cockpit outbox event"),
        goal_id: optionalId(message.references.goal_id, "cockpit outbox goal"),
        plan_id: optionalId(message.references.plan_id, "cockpit outbox plan"),
        decision_id: optionalId(message.references.decision_id, "cockpit outbox decision"),
        action_id: optionalId(message.references.action_id, "cockpit outbox action"),
        commitment_id: optionalId(message.references.commitment_id, "cockpit outbox commitment"),
      },
    };
    }),
  };
}

function parseActionTrace(value: unknown): CockpitActionTraceResponse {
  if (!isRecord(value) || !Array.isArray(value.traces)) invalid("action trace");
  return {
    pending_approval_count: nonnegativeInteger(value.pending_approval_count, "action trace pending approval count"),
    retry_pending_count: nonnegativeInteger(value.retry_pending_count, "action trace retry count"),
    failed_count: nonnegativeInteger(value.failed_count, "action trace failed count"),
    traces: value.traces.map((trace): CockpitActionTrace => {
      if (!isRecord(trace) || !isRecord(trace.provenance) || !isRecord(trace.approval)) invalid("action trace");
      const provenance = trace.provenance;
      const approval = trace.approval;
      return {
        intent_id: id(trace.intent_id, "action intent"),
        revision: positiveInteger(trace.revision, "action intent revision"),
        tool_name: text(trace.tool_name, "action tool"),
        risk_class: enumValue(trace.risk_class, ["read_only", "reversible_write", "external_write", "destructive", "high_impact"] as const, "action risk class"),
        status: enumValue(trace.status, ["awaiting_approval", "approved", "dry_run", "executing", "retry_pending", "succeeded", "failed", "cancelled", "rejected", "compensated"] as const, "action intent"),
        dry_run: booleanValue(trace.dry_run, "action dry-run"),
        created_at: text(trace.created_at, "action created timestamp"),
        updated_at: text(trace.updated_at, "action updated timestamp"),
        failure_code: optionalBoundedCode(trace.failure_code, "action failure code"),
        provenance: {
          decision_id: id(provenance.decision_id, "action decision"),
          candidate_id: id(provenance.candidate_id, "action candidate"),
          triggering_event_id: optionalId(provenance.triggering_event_id, "action triggering event"),
          plan_id: optionalId(provenance.plan_id, "action plan"),
          plan_revision: optionalPositiveInteger(provenance.plan_revision, "action plan revision"),
          step_id: optionalId(provenance.step_id, "action step"),
        },
        approval: {
          approval_id: optionalId(approval.approval_id, "action approval"),
          status: approval.status === null ? null : enumValue(approval.status, ["pending", "approved", "rejected"] as const, "action approval"),
          requested_at: optionalText(approval.requested_at, "action approval requested timestamp"),
          resolved_at: optionalText(approval.resolved_at, "action approval resolved timestamp"),
          resolved_by_operator: booleanValue(approval.resolved_by_operator, "action approval operator resolution"),
        },
        receipt: parseActionReceipt(trace.receipt),
        related_receipts: recordArray(trace.related_receipts, "action related receipt").map((receipt) => ({
          receipt_id: id(receipt.receipt_id, "action related receipt"),
          status: enumValue(receipt.status, ["succeeded", "failed", "timed_out", "cancelled", "compensated"] as const, "action related receipt"),
        })),
        observation: parseActionObservation(trace.observation),
        verification: parseActionVerification(trace.verification),
      };
    }),
    pre_intent_failures: recordArray(value.pre_intent_failures, "pre-intent failure").map((failure) => ({
      failure_id: id(failure.failure_id, "pre-intent failure"),
      failure_type: enumValue(failure.failure_type, ["validation", "policy_rejection"] as const, "pre-intent failure"),
      decision_id: optionalId(failure.decision_id, "pre-intent failure decision"),
      candidate_id: optionalId(failure.candidate_id, "pre-intent failure candidate"),
      tool_name: optionalToolName(failure.tool_name, "pre-intent failure tool"),
      risk_class: failure.risk_class === null ? null : enumValue(failure.risk_class, ["read_only", "reversible_write", "external_write", "destructive", "high_impact"] as const, "pre-intent failure risk class"),
      error_codes: stringArray(failure.error_codes, "pre-intent failure error codes").map((code) => boundedCode(code, "pre-intent failure error code")),
      event_id: id(failure.event_id, "pre-intent failure event"),
      event_sequence: positiveInteger(failure.event_sequence, "pre-intent failure event sequence"),
      occurred_at: text(failure.occurred_at, "pre-intent failure timestamp"),
    })),
  };
}

function parseActionReceipt(value: unknown): CockpitActionReceipt | null {
  if (value === null) return null;
  if (!isRecord(value)) invalid("action receipt");
  return {
    receipt_id: id(value.receipt_id, "action receipt"),
    status: enumValue(value.status, ["succeeded", "failed", "timed_out", "cancelled", "compensated"] as const, "action receipt"),
    attempt: nonnegativeInteger(value.attempt, "action receipt attempt"),
    duration_ms: nonnegativeNumber(value.duration_ms, "action receipt duration"),
    event_id: optionalId(value.event_id, "action receipt event"),
    event_sequence: optionalPositiveInteger(value.event_sequence, "action receipt event sequence"),
    error_code: optionalBoundedCode(value.error_code, "action receipt error code"),
    compensation_of: optionalId(value.compensation_of, "action compensation receipt"),
  };
}

function parseActionObservation(value: unknown): CockpitActionObservation | null {
  if (value === null) return null;
  if (!isRecord(value)) invalid("action observation");
  const digest = text(value.result_digest, "action observation digest");
  if (!/^[0-9a-f]{64}$/.test(digest)) invalid("action observation digest");
  return {
    observation_id: id(value.observation_id, "action observation"),
    valid: booleanValue(value.valid, "action observation validity"),
    validation_errors: stringArray(value.validation_errors, "action observation validation errors").map((code) => boundedCode(code, "action observation validation error")),
    result_digest: digest,
  };
}

function parseActionVerification(value: unknown): CockpitActionVerification | null {
  if (value === null) return null;
  if (!isRecord(value)) invalid("action verification");
  return {
    verification_id: id(value.verification_id, "action verification"),
    success: booleanValue(value.success, "action verification result"),
    reason: boundedCode(value.reason, "action verification reason"),
  };
}

export function parseCockpitTrainingSummary(value: unknown): CockpitTrainingSummary {
  exactRecord(value, ["node_count", "online_node_count", "running_job_count", "failed_job_count", "importing_job_count", "active_adapter_count", "candidate_adapter_count", "nodes", "jobs", "adapters"], "cockpit training summary");
  return {
    node_count: nonnegativeInteger(value.node_count, "cockpit training node count"),
    online_node_count: nonnegativeInteger(value.online_node_count, "cockpit training online node count"),
    running_job_count: nonnegativeInteger(value.running_job_count, "cockpit training running job count"),
    failed_job_count: nonnegativeInteger(value.failed_job_count, "cockpit training failed job count"),
    importing_job_count: nonnegativeInteger(value.importing_job_count, "cockpit training importing job count"),
    active_adapter_count: nonnegativeInteger(value.active_adapter_count, "cockpit training active adapter count"),
    candidate_adapter_count: nonnegativeInteger(value.candidate_adapter_count, "cockpit training candidate adapter count"),
    nodes: recordArray(value.nodes, "cockpit training node").map(parseCockpitTrainingNode),
    jobs: recordArray(value.jobs, "cockpit training job").map(parseCockpitTrainingJob),
    adapters: recordArray(value.adapters, "cockpit adapter lineage").map(parseCockpitAdapterLineage),
  };
}

function parseCockpitTrainingNode(node: Record<string, unknown>): CockpitTrainingNode {
  exactRecord(node, ["node_id", "role", "backend", "status", "last_contact_at", "expected_model_id", "expected_model_revision", "expected_processor_revision", "observed_model_id", "observed_model_revision", "model_matches_expected", "gpu_name", "cuda_version", "driver_version"], "cockpit training node");
  return {
    node_id: safeId(node.node_id, "cockpit training node"),
    role: enumValue(node.role, ["inference", "worker"] as const, "cockpit training node role"),
    backend: safeText(node.backend, "cockpit training node backend"),
    status: enumValue(node.status, ["online", "unavailable"] as const, "cockpit training node health"),
    last_contact_at: optionalIsoTimestamp(node.last_contact_at, "cockpit training node last contact"),
    expected_model_id: optionalSafeText(node.expected_model_id, "cockpit training expected model"),
    expected_model_revision: optionalSafeText(node.expected_model_revision, "cockpit training expected revision"),
    expected_processor_revision: optionalSafeText(node.expected_processor_revision, "cockpit training expected processor revision"),
    observed_model_id: optionalSafeText(node.observed_model_id, "cockpit training observed model"),
    observed_model_revision: optionalSafeText(node.observed_model_revision, "cockpit training observed revision"),
    model_matches_expected: optionalBoolean(node.model_matches_expected, "cockpit training model match"),
    gpu_name: optionalSafeText(node.gpu_name, "cockpit training GPU"),
    cuda_version: optionalSafeText(node.cuda_version, "cockpit training CUDA"),
    driver_version: optionalSafeText(node.driver_version, "cockpit training driver"),
  };
}

function parseCockpitTrainingJob(job: Record<string, unknown>): CockpitTrainingJob {
  exactRecord(job, ["job_id", "attempt_id", "status", "backend", "created_at", "updated_at", "started_at", "completed_at", "source_event_start", "source_event_end", "selected_episode_count", "remote_job_id", "worker_node_id", "retry_count", "transferred_bytes", "failure_code", "candidate_adapter_id", "import_status", "bundle_digest", "result_digest"], "cockpit training job");
  return {
    job_id: safeId(job.job_id, "cockpit training job"),
    attempt_id: safeId(job.attempt_id, "cockpit training attempt"),
    status: enumValue(job.status, ["preparing", "ready", "dispatched", "running", "succeeded", "importing", "completed", "failed", "cancelled", "unavailable"] as const, "cockpit training job status"),
    backend: optionalSafeText(job.backend, "cockpit training job backend"),
    created_at: optionalIsoTimestamp(job.created_at, "cockpit training job created"),
    updated_at: optionalIsoTimestamp(job.updated_at, "cockpit training job updated"),
    started_at: optionalIsoTimestamp(job.started_at, "cockpit training job started"),
    completed_at: optionalIsoTimestamp(job.completed_at, "cockpit training job completed"),
    source_event_start: optionalNonnegativeInteger(job.source_event_start, "cockpit training source event start"),
    source_event_end: optionalNonnegativeInteger(job.source_event_end, "cockpit training source event end"),
    selected_episode_count: optionalNonnegativeInteger(job.selected_episode_count, "cockpit training selected episodes"),
    remote_job_id: optionalSafeId(job.remote_job_id, "cockpit training remote job"),
    worker_node_id: optionalSafeId(job.worker_node_id, "cockpit training worker"),
    retry_count: optionalNonnegativeInteger(job.retry_count, "cockpit training retry"),
    transferred_bytes: optionalNonnegativeInteger(job.transferred_bytes, "cockpit training transferred bytes"),
    failure_code: job.failure_code === null ? null : enumValue(job.failure_code, ["worker_unavailable", "ssh_failed", "transfer_failed", "timeout", "cancelled", "bundle_invalid", "result_invalid", "training_failed", "cuda_oom", "non_finite_metrics", "import_failed", "evaluation_failed", "unknown_failure"] as const, "cockpit training failure code"),
    candidate_adapter_id: optionalSafeId(job.candidate_adapter_id, "cockpit training candidate adapter"),
    import_status: enumValue(job.import_status, ["not_started", "importing", "completed", "failed", "unavailable"] as const, "cockpit training import status"),
    bundle_digest: optionalDigest(job.bundle_digest, "cockpit training bundle digest"),
    result_digest: optionalDigest(job.result_digest, "cockpit training result digest"),
  };
}

function parseCockpitAdapterLineage(adapter: Record<string, unknown>): CockpitAdapterLineage {
  exactRecord(adapter, ["adapter_id", "status", "adapter_hash", "base_model_id", "base_model_revision", "parent_adapter_id", "training_job_id", "training_node_id", "submitted_by_node_id", "imported_by_node_id", "evaluation_id", "evaluation_status", "approved", "active", "rollback_candidate", "activation_event_id", "activation_event_sequence", "rollback_event_id", "rollback_event_sequence"], "cockpit adapter lineage");
  return {
    adapter_id: safeId(adapter.adapter_id, "cockpit adapter lineage"),
    status: safeText(adapter.status, "cockpit adapter status"),
    adapter_hash: optionalDigest(adapter.adapter_hash, "cockpit adapter hash"),
    base_model_id: optionalSafeText(adapter.base_model_id, "cockpit adapter base model"),
    base_model_revision: optionalSafeText(adapter.base_model_revision, "cockpit adapter base revision"),
    parent_adapter_id: optionalSafeId(adapter.parent_adapter_id, "cockpit parent adapter"),
    training_job_id: optionalSafeId(adapter.training_job_id, "cockpit adapter training job"),
    training_node_id: optionalSafeId(adapter.training_node_id, "cockpit adapter training node"),
    submitted_by_node_id: optionalSafeId(adapter.submitted_by_node_id, "cockpit adapter submitter"),
    imported_by_node_id: optionalSafeId(adapter.imported_by_node_id, "cockpit adapter importer"),
    evaluation_id: optionalSafeId(adapter.evaluation_id, "cockpit adapter evaluation"),
    evaluation_status: enumValue(adapter.evaluation_status, ["passed", "failed", "stale", "corrupt", "unavailable"] as const, "cockpit adapter evaluation"),
    approved: booleanValue(adapter.approved, "cockpit adapter approved"),
    active: booleanValue(adapter.active, "cockpit adapter active"),
    rollback_candidate: booleanValue(adapter.rollback_candidate, "cockpit adapter rollback candidate"),
    activation_event_id: optionalSafeId(adapter.activation_event_id, "cockpit adapter activation event"),
    activation_event_sequence: optionalPositiveInteger(adapter.activation_event_sequence, "cockpit adapter activation sequence"),
    rollback_event_id: optionalSafeId(adapter.rollback_event_id, "cockpit adapter rollback event"),
    rollback_event_sequence: optionalPositiveInteger(adapter.rollback_event_sequence, "cockpit adapter rollback sequence"),
  };
}

function identityOrigin(value: unknown, label: string): string {
  if (!isRecord(value)) invalid(`${label} origin`);
  const actor = text(value.actor, `${label} origin`);
  const inputKind = text(value.input_kind, `${label} origin`);
  const endorsement = text(value.endorsement, `${label} origin`);
  return `${actor} / ${inputKind} / ${endorsement}`;
}

function decisionOutcomeStatus(value: unknown): Decision["outcome_status"] {
  if (value === null) return "pending";
  if (!isRecord(value) || typeof value.success !== "boolean" || typeof value.compensated !== "boolean") invalid("decision outcome");
  if (value.compensated) return "compensated";
  return value.success ? "succeeded" : "failed";
}

function responseArray(value: unknown, key: string, label: string): Record<string, unknown>[] {
  if (!isRecord(value)) invalid(label);
  return recordArray(value[key], label);
}

function recordArray(value: unknown, label: string): Record<string, unknown>[] {
  if (!Array.isArray(value) || !value.every(isRecord)) invalid(label);
  return value;
}

function id(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim() || /\s/.test(value)) invalid(`${label} ID`);
  return value;
}

function safeId(value: unknown, label: string): string {
  const result = id(value, label);
  if (result.length > 128 || !/^[A-Za-z0-9._:-]+$/.test(result)) invalid(`${label} ID`);
  return result;
}

/** Validate identifiers that are safe to expose as public links or references. */
function publicReference(value: unknown, label: string, kind?: string): string {
  const result = safeId(value, label);
  const privateMarker = /private[_-]*(?:sentinel|state|session|context)|hidden[_-]*thought|raw[_-]*prompt|chain[_-]*of[_-]*thought|api[_-]*(?:key|token|secret)|access[_-]*token|bearer|credential|token|secret|password|prompt|attachment[_-]*body/i;
  const filesystemPath = /^(?:[A-Za-z]:|~[\\/]|[\\/])/;
  if (privateMarker.test(result) || /[\p{Cc}\p{Cf}]/u.test(result) || filesystemPath.test(result)) invalid(`${label} ID`);
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(result);
  const opaque = uuid || /^[a-f0-9]{32,}$/i.test(result) || /^[A-Za-z][A-Za-z0-9_.:-]{0,96}[-:](?:\d+|[a-f0-9]{16,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i.test(result);
  if (!opaque) invalid(`${label} ID`);
  const prefixes: Record<string, string[]> = { goal: ["goal-"], commitment: ["commitment-"], decision: ["decision-", "goal-decision-"], plan: ["plan-"], action: ["action-", "intent-"], outbox: ["outbox-", "message-"], journal: ["event-", "operator-restore-", "journal-"], experience: ["experience-"], memory: ["memory-", "episode-", "semantic-"], belief: ["belief-"] };
  const opaqueTail = /^(?:\d+|[a-f0-9]{16,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i;
  if (kind && !uuid && !(prefixes[kind] ?? []).some((prefix) => result.toLowerCase().startsWith(prefix) && opaqueTail.test(result.slice(prefix.length)))) invalid(`${label} ID`);
  return result;
}

function optionalSafeId(value: unknown, label: string): string | null {
  return value === null ? null : safeId(value, label);
}

function optionalId(value: unknown, label: string): string | null {
  return value === null ? null : id(value, label);
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim()) invalid(label);
  return value;
}

function safeText(value: unknown, label: string): string {
  const result = text(value, label);
  if (result.length > 160 || !/^[A-Za-z0-9 ._+:/(),-]+$/.test(result)) invalid(label);
  return result;
}

function publicPreviewText(value: unknown, label: string, maximum: number): string {
  const result = text(value, label);
  const privateMarker = /<\/?think\b|hidden[\s_-]*thought|private[\s_-]*(?:state|session|context|sentinel)|raw[\s_-]*prompt|api[\s_-]*(?:key|token)|access[\s_-]*token|\b(?:credential|credentials|password|secret|secrets|token|tokens)\b|bearer\s+token|(?:^|[^A-Z0-9])sentinel(?:$|[^A-Z0-9])/i;
  const filesystemPath = /(?:[A-Za-z]:[\\/]|~[\\/]|(?:^|\s)\/(?:[^\s/]+\/)+[^\s]*)/;
  if (result.length > maximum || /[\p{Cc}\p{Cf}]/u.test(result) || privateMarker.test(result) || filesystemPath.test(result)) invalid(label);
  return result;
}

function optionalSafeText(value: unknown, label: string): string | null {
  return value === null ? null : safeText(value, label);
}

function isoTimestamp(value: unknown, label: string): string {
  const result = text(value, label);
  if (Number.isNaN(Date.parse(result))) invalid(label);
  return result;
}

function optionalIsoTimestamp(value: unknown, label: string): string | null {
  return value === null ? null : isoTimestamp(value, label);
}

function digest(value: unknown, label: string): string {
  const result = text(value, label);
  if (!/^[0-9a-f]{64}$/.test(result)) invalid(label);
  return result;
}

function optionalDigest(value: unknown, label: string): string | null {
  return value === null ? null : digest(value, label);
}

function boundedCode(value: unknown, label: string): string {
  const result = text(value, label);
  if (result.length > 128 || !/^[A-Za-z0-9_.-]+$/.test(result)) invalid(label);
  return result;
}

function optionalBoundedCode(value: unknown, label: string): string | null {
  return value === null ? null : boundedCode(value, label);
}

function optionalText(value: unknown, label: string): string | null {
  return value === null ? null : text(value, label);
}

function optionalToolName(value: unknown, label: string): typeof ACTION_TOOL_NAMES[number] | null {
  if (value === null) return null;
  if (typeof value !== "string" || !ACTION_TOOL_NAMES.includes(value as typeof ACTION_TOOL_NAMES[number])) invalid(label);
  return value as typeof ACTION_TOOL_NAMES[number];
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string" && item.length > 0)) invalid(label);
  return value;
}

function idArray(value: unknown, label: string): string[] {
  const values = stringArray(value, label);
  values.forEach((item) => id(item, label));
  return values;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) invalid(label);
  return value;
}

function nonnegativeNumber(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (result < 0) invalid(label);
  return result;
}

function optionalNumber(value: unknown, label: string): number | null {
  return value === null ? null : finiteNumber(value, label);
}

function optionalNonnegativeInteger(value: unknown, label: string): number | null {
  return value === null ? null : nonnegativeInteger(value, label);
}

function nonnegativeInteger(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (!Number.isInteger(result) || result < 0) invalid(label);
  return result;
}

function positiveInteger(value: unknown, label: string): number {
  const result = nonnegativeInteger(value, label);
  if (result < 1) invalid(label);
  return result;
}

function optionalPositiveInteger(value: unknown, label: string): number | null {
  return value === null ? null : positiveInteger(value, label);
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") invalid(label);
  return value;
}

function optionalBoolean(value: unknown, label: string): boolean | null {
  return value === null ? null : booleanValue(value, label);
}

function exactRecord<const T extends readonly string[]>(value: unknown, keys: T, label: string): asserts value is Record<T[number], unknown> {
  if (!isRecord(value)) invalid(label);
  const allowed = new Set(keys);
  const actual = Object.keys(value);
  if (actual.length !== keys.length || actual.some((key) => !allowed.has(key))) invalid(label);
}

function enumValue<const T extends readonly string[]>(value: unknown, values: T, label: string): T[number] {
  if (typeof value !== "string" || !values.includes(value)) invalid(`${label} status`);
  return value as T[number];
}

function invalid(label: string): never {
  throw new ApiError(`Backend returned an invalid ${label} response.`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function cancelChatJob(operationId: string): Promise<ChatCancelResponse> {
  const response = await request<ChatCancelResponse>(`/api/chat/jobs/${encodeURIComponent(operationId)}`, { method: "DELETE" });
  if (!["canceled", "cancel_requested", "already_completed", "failed"].includes(response.disposition)) {
    throw new ApiError("Unknown cancellation disposition");
  }
  return response;
}

function actionMutation(path: string, body: ActionMutationCommon | (ApproveActionRequest & { approved: boolean })): Promise<ActionMutationResponse> {
  return privateApiRequest<unknown>(path, { method: "POST", body: JSON.stringify(body) }).then(parseActionMutation).catch((error: unknown) => {
    if (error instanceof ApiError && error.status !== null) {
      const status = error.status;
      throw new ApiError(`Action request failed (${status}).`, status, null, `HTTP status ${status}.`);
    }
    throw error;
  });
}

function parseActionMutation(value: unknown): ActionMutationResponse {
  if (!isRecord(value)) invalid("action mutation");
  exactRecord(value, ["command", "event_id", "processing_sequence", "action", "disposition"], "action mutation");
  if (!isRecord(value.action)) invalid("action mutation action");
  return {
    command: enumValue(value.command, ["approve", "reject", "cancel", "retry_now", "compensate"] as const, "action mutation command"),
    event_id: safeId(value.event_id, "action mutation event"),
    processing_sequence: positiveInteger(value.processing_sequence, "action mutation sequence"),
    action: parseOperatorAction(value.action),
    disposition: enumValue(value.disposition, ["awaiting_scheduler", "rejected", "cancelled", "executed", "compensated"] as const, "action mutation disposition"),
  };
}

export const api = {
  chat: (body: ChatRequest) => request<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify(body) }),
  chatJobResult: (operationId: string) => request<ChatJobResult>(`/api/chat/jobs/${encodeURIComponent(operationId)}/result`),
  cancelChatJob,
  feedback: (body: FeedbackRequest) => request<FeedbackResponse>("/api/feedback", { method: "POST", body: JSON.stringify(body) }),
  debugChat: (body: ChatRequest) => privateApiRequest<DebugChatResponse>("/chat/debug", { method: "POST", body: JSON.stringify(body) }),
  emotion: () => privateApiRequest<Emotion>("/state/emotion"),
  workingMemory: async () => parseWorkingMemory(await privateApiRequest<unknown>("/state/working-memory")),
  operatorRestoreSummary: async (limit?: number) => parseOperatorRestoreSummary(await restoreRequest<unknown>(`/state/operator-restore/summary${limit === undefined ? "" : `?limit=${encodeURIComponent(String(restoreLimit(limit)))}`}`)),
  previewOperatorRestore: async (targetSequence: number) => parseOperatorRestorePreview(await restoreRequest<unknown>(`/state/operator-restore/preview/${encodeURIComponent(String(restoreSequence(targetSequence, "restore target sequence")))}`)),
  commitOperatorRestore: async (body: OperatorRestoreCommitRequest) => parseOperatorRestoreCommit(await restoreRequest<unknown>("/state/operator-restore/commit", { method: "POST", body: JSON.stringify(prepareOperatorRestoreCommit(body)) })),
  contexts: async () => parseContexts(await privateApiRequest<unknown>("/contexts")),
  goals: async () => parseGoals(await privateApiRequest<unknown>("/goals")),
  commitments: async () => parseCommitments(await privateApiRequest<unknown>("/commitments")),
  plans: async () => parsePlans(await privateApiRequest<unknown>("/plans")),
  decisions: async () => parseDecisions(await privateApiRequest<unknown>("/decisions")),
  cockpitOutbox: async () => parseCockpitOutbox(await privateApiRequest<unknown>("/outbox/summary")),
  actionTrace: async () => parseActionTrace(await privateApiRequest<unknown>("/actions/trace")),
  actionOperatorSummary: async () => parseOperatorSummary(await privateApiRequest<unknown>("/actions/operator-summary")),
  approveAction: (intentId: string, body: ApproveActionRequest) => actionMutation(`/actions/operator/intents/${encodeURIComponent(id(intentId, "action intent"))}/approval`, { ...body, approved: true }),
  rejectAction: (intentId: string, body: ApproveActionRequest) => actionMutation(`/actions/operator/intents/${encodeURIComponent(id(intentId, "action intent"))}/approval`, { ...body, approved: false }),
  cancelAction: (intentId: string, body: ActionMutationCommon) => actionMutation(`/actions/operator/intents/${encodeURIComponent(id(intentId, "action intent"))}/cancel`, body),
  retryAction: (intentId: string, body: ActionMutationCommon) => actionMutation(`/actions/operator/intents/${encodeURIComponent(id(intentId, "action intent"))}/retry`, body),
  compensateAction: (intentId: string, body: ActionMutationCommon) => actionMutation(`/actions/operator/intents/${encodeURIComponent(id(intentId, "action intent"))}/compensate`, body),
  cockpitTraining: async () => parseCockpitTrainingSummary(await privateApiRequest<unknown>("/training/cockpit-summary")),
  memorySearch: (query: string) => privateApiRequest<MemorySearchResponse>(`/memory/search?query=${encodeURIComponent(query)}`),
  archiveEpisodeMemory: (episodeId: string) => privateApiRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/archive`, { method: "POST" }),
  updateEpisodeMemoryMetadata: (episodeId: string, body: MemoryMetadataUpdate) => privateApiRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/metadata`, { method: "POST", body: JSON.stringify(body) }),
  reviewEpisodeMemory: (episodeId: string, body: MemoryReviewUpdate) => privateApiRequest<EpisodeMemory>(`/memory/episodes/${encodeURIComponent(episodeId)}/review`, { method: "POST", body: JSON.stringify(body) }),
  archiveSemanticMemory: (memoryId: string) => privateApiRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/archive`, { method: "POST" }),
  updateSemanticMemoryMetadata: (memoryId: string, body: MemoryMetadataUpdate) => privateApiRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/metadata`, { method: "POST", body: JSON.stringify(body) }),
  updateSemanticLifecycle: (memoryId: string, action: "archive" | "restore" | "forget", idempotencyKey: string) => privateApiRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/lifecycle`, { method: "POST", body: JSON.stringify({ action, idempotency_key: idempotencyKey }) }),
  semanticGraph: (memoryId: string) => privateApiRequest<{ records: SemanticMemory[] }>(`/memory/semantic/${encodeURIComponent(memoryId)}/graph`),
  relateSemanticMemory: (memoryId: string, targetId: string, relationship: "merge" | "contradiction" | "supersession" | "correction", idempotencyKey: string) => privateApiRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/relationships`, { method: "POST", body: JSON.stringify({ target_id: targetId, relationship, idempotency_key: idempotencyKey }) }),
  updateSemanticPolicy: (memoryId: string, body: { confidence: number; validity: "valid" | "disputed" | "invalid"; valid_from?: string | null; valid_until?: string | null; expires_at?: string | null; decay_rate: number; idempotency_key: string }) => privateApiRequest<SemanticMemory>(`/memory/semantic/${encodeURIComponent(memoryId)}/policy`, { method: "POST", body: JSON.stringify(body) }),
  createSleepJob: (idempotency_key?: string) => privateApiRequest<TrainingJob>("/sleep/jobs", { method: "POST", body: JSON.stringify({ idempotency_key }) }),
  sleepJobs: () => privateApiRequest<TrainingJobListResponse>("/sleep/jobs"),
  datasetRevisions: () => privateApiRequest<{ datasets: DatasetRevisionSummary[] }>("/training/datasets"),
  datasetRevision: (revision: string) => privateApiRequest<DatasetRevisionDetail>(`/training/datasets/${encodeURIComponent(revision)}`),
  datasetRevisionDiff: (fromRevision: string, toRevision: string) => privateApiRequest<DatasetRevisionDiff>(`/training/datasets/diff?from=${encodeURIComponent(fromRevision)}&to=${encodeURIComponent(toRevision)}`),
  cancelSleepJob: (jobId: string) => privateApiRequest<TrainingJob>(`/sleep/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }),
  retrySleepJob: (jobId: string) => privateApiRequest<TrainingJob>(`/sleep/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" }),
  reconcileSleepJob: (jobId: string) => privateApiRequest<TrainingJob>(`/sleep/jobs/${encodeURIComponent(jobId)}/reconcile`, { method: "POST" }),
  reconcileSleepJobs: () => privateApiRequest<{ jobs: TrainingJob[]; orphan_result_job_ids: string[]; orphan_remote_job_ids: string[] }>("/sleep/reconcile", { method: "POST" }),
  cleanupSleepArtifacts: () => privateApiRequest<{ removed: string[]; remote_removed: string[]; retention_days: number }>("/sleep/cleanup", { method: "POST" }),
  adapters: () => privateApiRequest<AdapterListResponse>("/adapters"),
  adapterProvenance: (adapterId: string) => privateApiRequest<AdapterProvenance>(`/adapters/${encodeURIComponent(adapterId)}/provenance`),
  adapterRuntime: () => privateApiRequest<AdapterRuntimeState>("/adapters/runtime"),
  evaluateAdapter: (adapterId: string) => privateApiRequest<AdapterEvaluateResponse>(`/adapters/${adapterId}/evaluate`, { method: "POST", body: JSON.stringify({}) }),
  adapterBehavioralStatus: (adapterId: string) => privateApiRequest<AdapterBehavioralStatus>(`/adapters/${encodeURIComponent(adapterId)}/behavioral-evaluation-status`),
  trialAdapter: (adapterId: string) => privateApiRequest<Adapter>(`/adapters/${adapterId}/trial`, { method: "POST" }),
  approveAdapter: (adapterId: string) => privateApiRequest<Adapter>(`/adapters/${adapterId}/approve`, { method: "POST" }),
  activateAdapter: (adapterId: string) => privateApiRequest<Adapter>(`/adapters/${adapterId}/activate`, { method: "POST" }),
  rollbackAdapter: () => privateApiRequest<AdapterActivationResponse>("/adapters/rollback", { method: "POST" }),
  rejectAdapter: (adapterId: string) => privateApiRequest<Adapter>(`/adapters/${adapterId}/reject`, { method: "POST" }),
  evaluations: () => privateApiRequest<EvaluationResultListResponse>("/evaluations"),
  adapterEvaluationHistory: (adapterId: string) => privateApiRequest<AdapterEvaluationHistoryResponse>(`/evaluations/adapters/${encodeURIComponent(adapterId)}/history`),
  evaluationResult: (filename: string) => privateApiRequest<EvaluationResultDetail>(`/evaluations/${encodeURIComponent(filename)}`),
  behavioralEvaluations: () => privateApiRequest<BehavioralEvaluationHistoryResponse>("/evaluations/behavioral"),
  behavioralEvaluation: (evaluationId: string) => privateApiRequest<BehavioralEvaluationDetail>(`/evaluations/behavioral/${encodeURIComponent(evaluationId)}`),
  behavioralFailure: (evaluationId: string, scenarioId: string) => privateApiRequest<BehavioralFailureArtifact>(`/evaluations/behavioral/${encodeURIComponent(evaluationId)}/failures/${encodeURIComponent(scenarioId)}.json`),
  rerunBehavioralEvaluation: (evaluationId: string, rerunId: string) => privateApiRequest<BehavioralRerunResponse>(`/evaluations/behavioral/${encodeURIComponent(evaluationId)}/rerun`, { method: "POST", body: JSON.stringify({ rerun_id: rerunId }) }),
  systemInfo: () => requestUrl<SystemInfoResponse>("/api-proxy/system/info"),
  runtimeEvents: () => privateApiRequest<RuntimeEventListResponse>("/system/events"),
  eventJournal: () => privateApiRequest<JournalRecordListResponse>("/system/journal"),
  experiences: () => privateApiRequest<ExperienceListResponse>("/experiences"),
  experience: (experienceId: string) => privateApiRequest<Experience>(`/experiences/${encodeURIComponent(experienceId)}`),
  beliefs: (activeOnly = false) => privateApiRequest<BeliefListResponse>(`/beliefs?active_only=${activeOnly}`),
  decisionExplanations: async () => parseDecisionExplanationResponse(await privateApiRequest<unknown>("/decisions/explanations")),
  motivation: () => privateApiRequest<MotivationState>("/motivation"),
  outboxMessages: async () => parseOutbox(await privateApiRequest<unknown>("/outbox/messages")),
  deliverOutbox: async () => parseOutbox(await privateApiRequest<unknown>("/outbox/deliveries", { method: "POST" })),
  respondToOutbox: async (messageId: string, kind: "read" | "reply", text?: string) => parseOutboxMessage(await privateApiRequest<unknown>(`/outbox/messages/${encodeURIComponent(id(messageId, "outbox message"))}/responses`, { method: "POST", body: JSON.stringify({ kind, text }) })),
};
