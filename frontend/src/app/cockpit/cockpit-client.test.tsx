import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { CockpitClient } from "./cockpit-client";

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    api: {
      systemInfo: vi.fn(), emotion: vi.fn(), workingMemory: vi.fn(), contexts: vi.fn(), goals: vi.fn(), commitments: vi.fn(), plans: vi.fn(), decisions: vi.fn(), cockpitOutbox: vi.fn(), eventJournal: vi.fn(), adapters: vi.fn(),
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
    expect(screen.getAllByRole("link", { name: "plan-1" })[0]).toHaveAttribute("href", "#plan-plan-1");
    expect(screen.getAllByRole("link", { name: "decision-1" })[0]).toHaveAttribute("href", "#decision-decision-1");
    expect(screen.getByRole("link", { name: "message-1" })).toHaveAttribute("href", "#outbox-message-1");
    expect(document.body.textContent).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "outbox"]))).not.toContain("PRIVATE_SENTINEL");
  });

  it("shows section loading without hiding loaded sections", async () => {
    mockedApi.contexts.mockReturnValue(new Promise(() => undefined));
    renderCockpit();

    expect(await screen.findByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("Loading contexts...")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });

  it("shows explicit empty states for independent sections", async () => {
    mockedApi.contexts.mockResolvedValue({ contexts: [] });
    mockedApi.goals.mockResolvedValue({ goals: [], decisions: [] });
    mockedApi.commitments.mockResolvedValue({ commitments: [] });
    mockedApi.plans.mockResolvedValue({ plans: [] });
    mockedApi.decisions.mockResolvedValue({ decisions: [] });
    mockedApi.cockpitOutbox.mockResolvedValue({ pending_count: 0, critical_count: 0, messages: [] });
    mockedApi.eventJournal.mockResolvedValue({ records: [] });
    renderCockpit();

    expect(await screen.findByText("No contexts recorded.")).toBeInTheDocument();
    expect(screen.getByText("No active or candidate goals.")).toBeInTheDocument();
    expect(screen.getByText("No current commitments.")).toBeInTheDocument();
    expect(screen.getByText("No active plans.")).toBeInTheDocument();
    expect(screen.getByText("No decisions recorded.")).toBeInTheDocument();
    expect(screen.getByText("No proactive messages.")).toBeInTheDocument();
    expect(screen.getByText("No Journal records.")).toBeInTheDocument();
  });

  it("keeps successful sections visible after a partial failure", async () => {
    mockedApi.goals.mockRejectedValue(new Error("goals unavailable"));
    renderCockpit();

    expect(await screen.findByText("goals unavailable")).toBeInTheDocument();
    expect(screen.getByText("Report results")).toBeInTheDocument();
    expect(screen.getByText("conversation")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
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
