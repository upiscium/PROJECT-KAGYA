import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { CockpitClient } from "./cockpit-client";

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    api: {
      systemInfo: vi.fn(), emotion: vi.fn(), workingMemory: vi.fn(), contexts: vi.fn(), goals: vi.fn(), commitments: vi.fn(), plans: vi.fn(), decisions: vi.fn(), cockpitOutbox: vi.fn(), actionTrace: vi.fn(), eventJournal: vi.fn(), adapters: vi.fn(),
    },
  };
});

const mockedApi = vi.mocked(api);

beforeEach(() => {
  vi.clearAllMocks();
  resolveRepresentativeData();
});

describe("CockpitClient", () => {
  it("renders representative safe sections and causal entity links", async () => {
    const { queryClient } = renderCockpit();

    expect(await screen.findByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("report.result")).toBeInTheDocument();
    expect(screen.getByText((_, element) => element?.tagName === "P" && element.textContent?.endsWith("Report result") === true)).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();
    expect(screen.getByText("Newest message")).toBeInTheDocument();
    expect(screen.queryByText("Oldest message")).not.toBeInTheDocument();
    expect(screen.getByText("goal_update")).toBeInTheDocument();
    expect(screen.getByText("Action Execution")).toBeInTheDocument();
    expect(screen.getByText("Pending approvals").parentElement).toHaveTextContent("2");
    expect(screen.getByText("Retry pending").parentElement).toHaveTextContent("1");
    expect(screen.getByText("Failed").parentElement).toHaveTextContent("5");
    expect(screen.getByText("Validation rejected")).toBeInTheDocument();
    expect(screen.getByText("Policy rejected")).toBeInTheDocument();
    expect(screen.getByText("awaiting_approval")).toBeInTheDocument();
    expect(screen.getByText("retry_pending")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getAllByText("compensated").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "plan-1" })[0]).toHaveAttribute("href", "#plan-plan-1");
    expect(screen.getAllByRole("link", { name: "decision-1" })[0]).toHaveAttribute("href", "#decision-decision-1");
    expect(screen.getByRole("link", { name: "message-1" })).toHaveAttribute("href", "#outbox-message-1");
    expect(screen.getAllByRole("link", { name: "action-1" })[0]).toHaveAttribute("href", "#action-action-1");
    const decisionSection = screen.getByText("Recent Decisions").closest(".ui-card") as HTMLElement;
    expect(within(decisionSection).getByRole("link", { name: "action-1" })).toHaveAttribute("href", "#action-action-1");
    expect(within(decisionSection).queryByRole("link", { name: "action-old" })).not.toBeInTheDocument();
    const actionSection = screen.getByText("Action Execution").closest(".ui-card") as HTMLElement;
    const journalSection = screen.getByText("Recent Journal").closest(".ui-card") as HTMLElement;
    const receiptRecord = within(actionSection).getByRole("link", { name: "receipt-1" }).closest(".step-row") as HTMLElement;
    expect(within(receiptRecord).getByRole("link", { name: "receipt-1" })).toHaveAttribute("href", "#receipt-receipt-1");
    expect(within(receiptRecord).getByRole("link", { name: "event-1" })).toHaveAttribute("href", "#journal-event-1");
    expect(within(journalSection).getByRole("link", { name: "action-1" })).toHaveAttribute("href", "#action-action-1");
    expect(within(journalSection).getByRole("link", { name: "receipt-1" })).toHaveAttribute("href", "#receipt-receipt-1");
    expect(screen.getByRole("link", { name: "observation-1" })).toHaveAttribute("href", "#observation-observation-1");
    expect(screen.getByRole("link", { name: "verification-1" })).toHaveAttribute("href", "#verification-verification-1");
    const compensation = screen.getByText((_, element) => element?.tagName === "P" && element.textContent === "Compensates: receipt-original");
    expect(within(compensation).getByRole("link", { name: "receipt-original" })).toHaveAttribute("href", "#receipt-receipt-original");
    expect(document.body.textContent).not.toContain("PRIVATE_SENTINEL");
    expect(document.body.textContent).not.toContain("<script>");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "outbox"]))).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "actions"]))).not.toContain("PRIVATE_SENTINEL");
  });

  it("shows action loading without hiding loaded sections", async () => {
    mockedApi.actionTrace.mockReturnValue(new Promise(() => undefined));
    renderCockpit();

    expect(await screen.findByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("Loading action execution...")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });

  it("shows explicit empty states for independent sections", async () => {
    mockedApi.contexts.mockResolvedValue({ contexts: [] });
    mockedApi.goals.mockResolvedValue({ goals: [], decisions: [] });
    mockedApi.commitments.mockResolvedValue({ commitments: [] });
    mockedApi.plans.mockResolvedValue({ plans: [] });
    mockedApi.decisions.mockResolvedValue({ decisions: [] });
    mockedApi.cockpitOutbox.mockResolvedValue({ pending_count: 0, critical_count: 0, messages: [] });
    mockedApi.actionTrace.mockResolvedValue({ pending_approval_count: 0, retry_pending_count: 0, failed_count: 0, traces: [], pre_intent_failures: [] });
    mockedApi.eventJournal.mockResolvedValue({ records: [] });
    renderCockpit();

    expect(await screen.findByText("No contexts recorded.")).toBeInTheDocument();
    expect(screen.getByText("No active or candidate goals.")).toBeInTheDocument();
    expect(screen.getByText("No current commitments.")).toBeInTheDocument();
    expect(screen.getByText("No active plans.")).toBeInTheDocument();
    expect(screen.getByText("No decisions recorded.")).toBeInTheDocument();
    expect(screen.getByText("No proactive messages.")).toBeInTheDocument();
    expect(screen.getByText("No action execution traces.")).toBeInTheDocument();
    expect(screen.getByText("No Journal records.")).toBeInTheDocument();
  });

  it("keeps successful sections visible after an action-only failure", async () => {
    mockedApi.actionTrace.mockRejectedValue(new Error("actions unavailable"));
    renderCockpit();

    expect(await screen.findByText("actions unavailable")).toBeInTheDocument();
    expect(screen.getByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("Report results")).toBeInTheDocument();
    expect(screen.getByText("conversation")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });

  it("links a decision to its pre-intent failure when no intent exists", async () => {
    mockedApi.actionTrace.mockResolvedValue({ pending_approval_count: 0, retry_pending_count: 0, failed_count: 2, traces: [], pre_intent_failures: rawActionFailures.map(cockpitFailureProjection) as never });
    const { queryClient } = renderCockpit();

    expect(await screen.findByText("Validation rejected")).toBeInTheDocument();
    expect(screen.getByText("Policy rejected")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "validation-1" })).toHaveAttribute("href", "#action-failure-validation-1");
    const actionSection = screen.getByText("Action Execution").closest(".ui-card") as HTMLElement;
    expect(within(actionSection).getByRole("link", { name: "event-1" })).toHaveAttribute("href", "#journal-event-1");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "actions"]))).not.toContain("PRIVATE_SENTINEL");
  });
});

function renderCockpit() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return { ...render(<QueryClientProvider client={queryClient}><CockpitClient /></QueryClientProvider>), queryClient };
}

function resolveRepresentativeData() {
  mockedApi.systemInfo.mockResolvedValue({ project: "KAGYA", status: "ok", build: { version: "1.0", commit: "abc123" }, runtime: { environment: "test", provider: "dummy", primary_model_id: "model-1", fallback_configured: false, transformers_4bit: false, qlora_dry_run: true, admin_token_configured: true } });
  mockedApi.emotion.mockResolvedValue({ valence: 0.4, arousal: 0.3, optimal_loss: 1 });
  mockedApi.workingMemory.mockResolvedValue({ item_count: 2, token_count: 80, item_capacity: 10, token_capacity: 1000 });
  mockedApi.adapters.mockResolvedValue({ adapters: [{ adapter_id: "adapter-1", adapter_hash: "adapter-hash", status: "active" } as never] });
  mockedApi.contexts.mockResolvedValue({ contexts: [{ context_id: "context-1", context_type: "conversation", source_channel: "chat", source_session_id: "session-1", participant_ids: ["operator"], active_topic: null, active_task: null, status: "active", hidden_thought: "PRIVATE_SENTINEL" } as never] });
  mockedApi.goals.mockResolvedValue({ goals: [{ goal_id: "goal-1", goal_type: "intrinsic", description: "Ship safely", priority: 0.8, urgency: 0.6, confidence: 0.9, origin: "self / internal_state / endorsed", status: "active", dependency_ids: [], conflict_ids: [], deadline: null, needs_information: false, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", raw_prompt: "PRIVATE_SENTINEL" } as never], decisions: [] });
  mockedApi.commitments.mockResolvedValue({ commitments: [{ commitment_id: "commitment-1", description: "Report results", related_goal_id: "goal-1", status: "active", beneficiary: "operator", scope: "Release report", deadline: null, cost: 0.1, burden: 0.1, fulfillability: "fulfillable", fulfillability_reason: "ready", decision_refs: ["decision-1"], created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", secret: "PRIVATE_SENTINEL" } as never] });
  mockedApi.plans.mockResolvedValue({ plans: [{ plan_id: "plan-1", goal_id: "goal-1", revision: 1, status: "active", steps: [{ step_id: "step-1", action_type: "respond", action_code: "report.result", dependency_ids: [], status: "in_progress", attempt_count: 1, started_at: "2026-01-01T00:00:00Z", retry_at: null, completed_at: null }], created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }] });
  mockedApi.decisions.mockResolvedValue({ decisions: [{ decision_id: "decision-1", context_id: "context-1", active_goal_ids: ["goal-1"], selected_candidate_id: "candidate-1", selected_candidate: { candidate_id: "candidate-1", candidate_type: "respond", proposed_action: "Report result", plan_id: "plan-1", plan_revision: 1, step_id: "step-1", goal_refs: ["goal-1"], commitment_refs: ["commitment-1"] }, selection_confidence: 0.8, status: "awaiting_outcome", outcome_status: "pending", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }] });
  mockedApi.cockpitOutbox.mockResolvedValue({ pending_count: 42, critical_count: 9, messages: rawOutboxFixtures.map(cockpitOutboxProjection) });
  mockedApi.actionTrace.mockResolvedValue({ pending_approval_count: 2, retry_pending_count: 1, failed_count: 5, traces: rawActionFixtures.map(cockpitActionProjection) as never, pre_intent_failures: rawActionFailures.map(cockpitFailureProjection) as never });
  mockedApi.eventJournal.mockResolvedValue({ records: [{ record_id: "record-1", timestamp: "2026-01-01T00:00:00Z", lifecycle: "completed", event_id: "event-1", event_type: "goal_update", source: "runtime", processing_sequence: 1, snapshot_sequence: 1, causation_id: null, correlation_id: null, state_hash_before: null, state_hash_after: null, snapshot_hash: null, failure_category: null, actor_id: null, actor_role: null, target: "goal:goal-1", reauthenticated: null, previous_record_hash: null, record_hash: "hash", private_replay: "PRIVATE_SENTINEL" } as never] });
}

const rawOutboxFixtures = [
  rawOutboxFixture("message-1", "Newest message"),
  rawOutboxFixture("message-2", "Recent message 2"),
  rawOutboxFixture("message-3", "Recent message 3"),
  rawOutboxFixture("message-4", "Recent message 4"),
  rawOutboxFixture("message-5", "Recent message 5"),
  rawOutboxFixture("message-6", "Oldest message"),
];

function rawOutboxFixture(messageId: string, title: string) {
  return {
    message_id: messageId,
    urgency: "critical" as const,
    delivery_status: "pending" as const,
    acknowledgment_status: "unacknowledged" as const,
    title,
    references: { decision_id: "decision-1", action_id: "action-1", event_id: null, goal_id: "goal-1", plan_id: "plan-1", commitment_id: "commitment-1" },
    body: "PRIVATE_SENTINEL",
    responses: [{ text: "PRIVATE_SENTINEL" }],
    attempts: [{ failure_code: "PRIVATE_SENTINEL" }],
  };
}

function cockpitOutboxProjection(message: ReturnType<typeof rawOutboxFixture>) {
  return {
    message_id: message.message_id,
    urgency: message.urgency,
    delivery_status: message.delivery_status,
    acknowledgment_status: message.acknowledgment_status,
    title: message.title,
    references: message.references,
  };
}

const rawActionFixtures = [
  rawActionFixture("action-1", "succeeded", "receipt-1", "succeeded"),
  { ...rawActionFixture("action-old", "succeeded", null, null), revision: 99, updated_at: "2026-07-29T00:00:00Z", provenance: { decision_id: "decision-1", candidate_id: "candidate-1", triggering_event_id: "event-old", plan_id: "plan-1", plan_revision: 1, step_id: "step-1" } },
  rawActionFixture("action-pending", "awaiting_approval", null, null),
  rawActionFixture("action-retry", "retry_pending", null, null),
  rawActionFixture("action-failed", "failed", "receipt-failed", "timed_out"),
  rawActionFixture("action-compensated", "compensated", "receipt-compensated", "compensated"),
];

function rawActionFixture(intentId: string, status: string, receiptId: string | null, receiptStatus: string | null) {
  return {
    intent_id: intentId,
    revision: 2,
    tool_name: "document_search",
    risk_class: "read_only",
    status,
    dry_run: false,
    created_at: "2026-07-30T00:00:00Z",
    updated_at: "2026-07-30T00:00:01Z",
    failure_code: status === "failed" || status === "retry_pending" ? "timeout" : null,
    provenance: { decision_id: intentId === "action-1" ? "decision-1" : `decision-${intentId}`, candidate_id: "candidate-1", triggering_event_id: "event-0", plan_id: "plan-1", plan_revision: 1, step_id: "step-1" },
    approval: { approval_id: status === "awaiting_approval" ? "approval-1" : null, status: status === "awaiting_approval" ? "pending" : null, requested_at: status === "awaiting_approval" ? "2026-07-30T00:00:00Z" : null, resolved_at: null, resolved_by_operator: false, reason: "PRIVATE_SENTINEL" },
    receipt: receiptId && receiptStatus ? { receipt_id: receiptId, status: receiptStatus, attempt: 1, duration_ms: 12.5, event_id: intentId === "action-1" ? "event-1" : null, event_sequence: intentId === "action-1" ? 45 : null, error_code: status === "failed" ? "timeout" : null, compensation_of: status === "compensated" ? "receipt-original" : null, idempotency_key: "PRIVATE_SENTINEL" } : null,
    related_receipts: status === "compensated" ? [{ receipt_id: "receipt-original", status: "succeeded", idempotency_key: "PRIVATE_SENTINEL" }] : [],
    observation: intentId === "action-1" ? { observation_id: "observation-1", valid: true, validation_errors: [], result_digest: "a".repeat(64), data: { result: "PRIVATE_SENTINEL" } } : null,
    verification: intentId === "action-1" ? { verification_id: "verification-1", success: true, reason: "observation_schema_valid", private_replay: "PRIVATE_SENTINEL" } : null,
    arguments: { query: "PRIVATE_SENTINEL" },
    preview: { arguments: { query: "PRIVATE_SENTINEL" } },
    idempotency_key: "PRIVATE_SENTINEL",
  };
}

function cockpitActionProjection(action: ReturnType<typeof rawActionFixture>) {
  return {
    intent_id: action.intent_id,
    revision: action.revision,
    tool_name: action.tool_name,
    risk_class: action.risk_class,
    status: action.status,
    dry_run: action.dry_run,
    created_at: action.created_at,
    updated_at: action.updated_at,
    failure_code: action.failure_code,
    provenance: action.provenance,
    approval: {
      approval_id: action.approval.approval_id,
      status: action.approval.status,
      requested_at: action.approval.requested_at,
      resolved_at: action.approval.resolved_at,
      resolved_by_operator: action.approval.resolved_by_operator,
    },
    receipt: action.receipt && {
      receipt_id: action.receipt.receipt_id,
      status: action.receipt.status,
      attempt: action.receipt.attempt,
      duration_ms: action.receipt.duration_ms,
      event_id: action.receipt.event_id,
      event_sequence: action.receipt.event_sequence,
      error_code: action.receipt.error_code,
      compensation_of: action.receipt.compensation_of,
    },
    related_receipts: action.related_receipts.map((receipt) => ({ receipt_id: receipt.receipt_id, status: receipt.status })),
    observation: action.observation && {
      observation_id: action.observation.observation_id,
      valid: action.observation.valid,
      validation_errors: action.observation.validation_errors,
      result_digest: action.observation.result_digest,
    },
    verification: action.verification && {
      verification_id: action.verification.verification_id,
      success: action.verification.success,
      reason: action.verification.reason,
    },
  };
}

const rawActionFailures = [
  { failure_id: "validation-1", failure_type: "validation", decision_id: "decision-1", candidate_id: null, tool_name: "document_search", risk_class: "read_only", error_codes: ["arguments_schema_invalid"], event_id: "event-1", event_sequence: 42, occurred_at: "2026-07-30T00:00:02Z", idempotency_key: "PRIVATE_SENTINEL", request_digest: "PRIVATE_SENTINEL", canonical_arguments_digest: "PRIVATE_SENTINEL", arguments: { secret: "PRIVATE_SENTINEL" } },
  { failure_id: "rejection-1", failure_type: "policy_rejection", decision_id: "decision-2", candidate_id: "candidate-2", tool_name: "PRIVATE_SENTINEL<script>", risk_class: "reversible_write", error_codes: ["risk_class_exceeds_budget"], event_id: "event-2", event_sequence: 43, occurred_at: "2026-07-30T00:00:01Z", idempotency_key: "PRIVATE_SENTINEL" },
];

function cockpitFailureProjection(failure: typeof rawActionFailures[number]) {
  return {
    failure_id: failure.failure_id,
    failure_type: failure.failure_type,
    decision_id: failure.decision_id,
    candidate_id: failure.candidate_id,
    tool_name: failure.tool_name === "document_search" ? failure.tool_name : null,
    risk_class: failure.risk_class,
    error_codes: failure.error_codes,
    event_id: failure.event_id,
    event_sequence: failure.event_sequence,
    occurred_at: failure.occurred_at,
  };
}
