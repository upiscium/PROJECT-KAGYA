import { describe, expect, it, vi, beforeEach } from "vitest";
import { ChatJobCanceledError, ChatJobFailedError, api, streamChatJob } from "./api";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function jsonResponse(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
}

function errorResponse(status: number, statusText: string, body: unknown) {
  return Promise.resolve({ ok: false, status, statusText, text: () => Promise.resolve(JSON.stringify(body)) });
}

describe("api client", () => {
  it("/chat sends requests to /api/chat", async () => {
    fetchMock.mockReturnValue(jsonResponse({ context_id: "c", episode_id: "e", experience_id: "x", response: "ok", emotion: { valence: 0, arousal: 0, optimal_loss: 1 }, model: { model_id: "m", adapter_id: null, adapter_hash: null, activation_sequence: null, fallback_used: false } }));

    await api.chat({ text: "hello", attachments: [], debug: false });

    expect(fetchMock).toHaveBeenCalledWith("/api-proxy/chat", expect.objectContaining({ method: "POST" }));
  });

  it("/debug sends requests through the server-side admin proxy", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.debugChat({ text: "hello", attachments: [], debug: true });

    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/chat/debug", expect.objectContaining({ method: "POST" }));
    expect(fetchMock.mock.calls[0][1]?.headers).not.toHaveProperty("X-KAGYA-Admin-Token");
  });

  it("admin actions call backend admin endpoints", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.evaluateAdapter("a");
    await api.trialAdapter("a");
    await api.approveAdapter("a");
    await api.activateAdapter("a");
    await api.rejectAdapter("a");
    await api.evaluations();
    await api.evaluationResult("adapter-a.json");

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "/admin-proxy/adapters/a/evaluate",
      "/admin-proxy/adapters/a/trial",
      "/admin-proxy/adapters/a/approve",
      "/admin-proxy/adapters/a/activate",
      "/admin-proxy/adapters/a/reject",
      "/admin-proxy/evaluations",
      "/admin-proxy/evaluations/adapter-a.json",
    ]);
  });

  it("sleep page uses asynchronous job endpoints", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.createSleepJob("request-1");
    await api.sleepJobs();

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "/admin-proxy/sleep/jobs",
      "/admin-proxy/sleep/jobs",
    ]);
  });

  it("system metadata calls public and admin system endpoints", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.systemInfo();
    await api.runtimeEvents();

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "/api-proxy/system/info",
      "/admin-proxy/system/events",
    ]);
  });

  it("validates the backend decision explanation schema before UI use", async () => {
    const explanation = {
      explanation_id: "explanation-1", revision: 1, decision_id: "decision-1", decision_revision: 1,
      selected: {}, major_alternatives: [], contributions: [], evidence_refs: [], uncertainty: [], information_gap_codes: [], omitted_reference_count: 0,
      risk: {}, tradeoff_refs: [], conflict_codes: [], boundary: null, reason_codes: [], outcome: {}, change: {},
      renderer: { offered_clause_ids: ["disposition.no_op.v1"], ordered_clause_ids: ["disposition.no_op.v1"], visible_explanation: "Disposition: no op." },
    };
    fetchMock.mockReturnValue(jsonResponse({ explanations: [explanation] }));

    const response = await api.decisionExplanations();

    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/decisions/explanations", expect.anything());
    expect(response.explanations[0].renderer.ordered_clause_ids).toEqual(["disposition.no_op.v1"]);

    fetchMock.mockReturnValue(jsonResponse({ explanations: [{ ...explanation, renderer: { visible_explanation: "invented prose" } }] }));
    await expect(api.decisionExplanations()).rejects.toThrow("invalid decision explanation renderer");
  });

  it("parses whitelisted cockpit projections through admin proxy paths", async () => {
    const cases = [
      [api.contexts, contextPayload, "/admin-proxy/contexts", "contexts"],
      [api.goals, goalPayload, "/admin-proxy/goals", "goals"],
      [api.commitments, commitmentPayload, "/admin-proxy/commitments", "commitments"],
      [api.plans, planPayload, "/admin-proxy/plans", "plans"],
      [api.decisions, decisionPayload, "/admin-proxy/decisions", "decisions"],
      [api.workingMemory, workingMemoryPayload, "/admin-proxy/state/working-memory", "item_count"],
      [api.cockpitOutbox, cockpitOutboxPayload, "/admin-proxy/outbox/summary", "messages"],
    ] as const;

    for (const [client, payload, path, projectionKey] of cases) {
      fetchMock.mockReturnValueOnce(jsonResponse(payload));
      const result = await client();
      expect(fetchMock).toHaveBeenLastCalledWith(path, expect.anything());
      expect(result).toHaveProperty(projectionKey);
      expect(JSON.stringify(result)).not.toContain("PRIVATE_SENTINEL");
    }
  });

  it.each([
    ["contexts", api.contexts, { contexts: [{ ...contextPayload.contexts[0], status: "invented" }] }],
    ["goals", api.goals, { ...goalPayload, goals: [{ ...goalPayload.goals[0], goal_id: "" }] }],
    ["goal decisions", api.goals, { ...goalPayload, decisions: [{ ...goalPayload.decisions[0], conflicting_goal_ids: "goal-2" }] }],
    ["commitments", api.commitments, { commitments: [{ ...commitmentPayload.commitments[0], related_goal_id: 3 }] }],
    ["plans", api.plans, { plans: [{ ...planPayload.plans[0], step_states: [] }] }],
    ["plan steps", api.plans, { plans: [{ ...planPayload.plans[0], revisions: [{ revision: 1, steps: [{ ...planPayload.plans[0].revisions[0].steps[0], action_type: "invented" }] }] }] }],
    ["decisions", api.decisions, { decisions: [{ ...decisionPayload.decisions[0], selected_candidate_id: "missing" }] }],
    ["decision references", api.decisions, { decisions: [{ ...decisionPayload.decisions[0], considered_candidates: [{ candidate: { ...decisionPayload.decisions[0].considered_candidates[0].candidate, plan_revision: null } }] }] }],
    ["working memory", api.workingMemory, { ...workingMemoryPayload, token_count: -1 }],
  ])("rejects malformed %s payloads", async (_label, client, payload) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(client()).rejects.toMatchObject({ name: "ApiError" });
  });

  it.each([
    ["root", api.contexts, null],
    ["collection", api.contexts, { contexts: {} }],
    ["missing ID", api.contexts, { contexts: [{ ...contextPayload.contexts[0], context_id: undefined }] }],
  ])("rejects malformed cockpit %s", async (_label, client, payload) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(client()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("parses only public-safe cockpit outbox summary fields", async () => {
    fetchMock.mockReturnValue(jsonResponse(cockpitOutboxPayload));

    const result = await api.cockpitOutbox();

    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/outbox/summary", expect.anything());
    expect(result).toEqual({ messages: [{
      message_id: "message-1",
      title: "Release ready",
      urgency: "critical",
      delivery_status: "pending",
      acknowledgment_status: "unacknowledged",
      references: {
        event_id: null,
        goal_id: "goal-1",
        plan_id: "plan-1",
        decision_id: "decision-1",
        action_id: "action-1",
        commitment_id: "commitment-1",
      },
    }] });
    expect(JSON.stringify(result)).not.toContain("PRIVATE_SENTINEL");
    expect(result.messages[0]).not.toHaveProperty("body");
    expect(result.messages[0]).not.toHaveProperty("responses");
    expect(result.messages[0]).not.toHaveProperty("attempts");
  });

  it.each([
    ["root", null],
    ["messages collection", { messages: {} }],
    ["message ID", { messages: [{ ...cockpitOutboxPayload.messages[0], message_id: undefined }] }],
    ["urgency", { messages: [{ ...cockpitOutboxPayload.messages[0], urgency: "urgent" }] }],
    ["delivery status", { messages: [{ ...cockpitOutboxPayload.messages[0], delivery_status: "invented" }] }],
    ["acknowledgment status", { messages: [{ ...cockpitOutboxPayload.messages[0], acknowledgment_status: "invented" }] }],
    ["reference", { messages: [{ ...cockpitOutboxPayload.messages[0], references: { ...cockpitOutboxPayload.messages[0].references, goal_id: 7 } }] }],
  ])("rejects malformed cockpit outbox %s", async (_label, payload) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(api.cockpitOutbox()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("formats backend JSON error details", async () => {
    fetchMock.mockReturnValue(errorResponse(500, "Internal Server Error", { detail: "Fallback model produced an empty visible response" }));

    await expect(api.chat({ text: "hello" })).rejects.toThrow("Backend failed: Fallback model produced an empty visible response");
  });

  it("formats unavailable backend errors", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    await expect(api.chat({ text: "hello" })).rejects.toThrow("Backend unavailable");
  });

  it("uses public wording for Chat service 503 responses", async () => {
    fetchMock.mockReturnValue(errorResponse(503, "Service Unavailable", { detail: "Chat job registry is not ready" }));

    await expect(api.chat({ text: "hello" })).rejects.toThrow("Chat service is temporarily unavailable: Chat job registry is not ready");
  });

  it("rejects unknown cancellation dispositions", async () => {
    fetchMock.mockReturnValue(jsonResponse({ disposition: "invented", operation: {} }));

    await expect(api.cancelChatJob("job-1")).rejects.toThrow("Unknown cancellation disposition");
  });

  it("reconnects SSE with Last-Event-ID after a reader disconnect", async () => {
    const operation = {
      schema_version: 1 as const, operation_id: "operation-1", event_id: "event-1", status: "running" as const,
      status_sequence: 2, queue_position: null, submitted_at: "2026-01-01T00:00:00Z", started_at: "2026-01-01T00:00:01Z",
      finalizing_at: null, completed_at: null, updated_at: "2026-01-01T00:00:01Z", error_code: null, cancel_code: null,
      cancel_requested: false, result_available: false,
    };
    const encoder = new TextEncoder();
    const disconnected = {
      getReader: () => ({
        read: vi.fn()
          .mockResolvedValueOnce({ done: false, value: encoder.encode(`id: 1\nevent: status\ndata: ${JSON.stringify(operation)}\n\n`) })
          .mockRejectedValueOnce(new Error("disconnected")),
      }),
    };
    const result = { context_id: "c", episode_id: "e", experience_id: "x", response: "ok", emotion: { valence: 0, arousal: 0, optimal_loss: 1 }, model: { model_id: "m", adapter_id: null, adapter_hash: null, activation_sequence: null, fallback_used: false } };
    const completed = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(`id: 2\nevent: final\ndata: ${JSON.stringify(result)}\n\n`));
        controller.close();
      },
    });
    fetchMock
      .mockReturnValueOnce(jsonResponse({ operation, status_url: "/api/chat/jobs/operation-1", result_url: "/api/chat/jobs/operation-1/result", events_url: "/api/chat/jobs/operation-1/events", duplicate: false }))
      .mockResolvedValueOnce({ ok: true, body: disconnected })
      .mockResolvedValueOnce({ ok: true, body: completed });

    await expect(streamChatJob({ text: "hello" }, { status: vi.fn(), token: vi.fn() })).resolves.toEqual(result);
    expect(fetchMock.mock.calls[2][1].headers).toEqual({ "Last-Event-ID": "1" });
  });

  it.each([
    ["canceled", "timeout", ChatJobCanceledError],
    ["error", "provider_error", ChatJobFailedError],
  ])("treats %s SSE as terminal without reconnect or result fetch", async (event, code, errorType) => {
    const operation = {
      schema_version: 1 as const, operation_id: "operation-1", event_id: "event-1", status: "running" as const,
      status_sequence: 2, queue_position: null, submitted_at: "2026-01-01T00:00:00Z", started_at: "2026-01-01T00:00:01Z",
      finalizing_at: null, completed_at: null, updated_at: "2026-01-01T00:00:01Z", error_code: null, cancel_code: null,
      cancel_requested: false, result_available: false,
    };
    const encoder = new TextEncoder();
    const terminal = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(`id: 1\nevent: ${event}\ndata: ${JSON.stringify({ code })}\n\n`));
        controller.close();
      },
    });
    fetchMock
      .mockReturnValueOnce(jsonResponse({ operation, status_url: "/api/chat/jobs/operation-1", result_url: "/api/chat/jobs/operation-1/result", events_url: "/api/chat/jobs/operation-1/events", duplicate: false }))
      .mockResolvedValueOnce({ ok: true, body: terminal });

    await expect(streamChatJob({ text: "hello" }, { status: vi.fn(), token: vi.fn() })).rejects.toBeInstanceOf(errorType);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

const contextPayload = {
  contexts: [{ context_id: "context-1", context_type: "conversation", source_channel: "chat", source_session_id: "session-1", participant_ids: ["person-1"], active_topic: "Release", active_task: null, status: "active", hidden_thought: "PRIVATE_SENTINEL" }],
};

const goalPayload = {
  goals: [{ goal_id: "goal-1", goal_type: "intrinsic", description: "Ship safely", priority: 0.8, urgency: 0.6, confidence: 0.9, identity_origin: { actor: "self", input_kind: "internal_state", endorsement: "endorsed" }, status: "active", dependency_ids: [], conflict_ids: [], deadline: null, needs_information: false, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", raw_prompt: "PRIVATE_SENTINEL" }],
  decisions: [{ decision_id: "goal-decision-1", action: "activate", goal_id: "goal-1", score: 0.8, reasons: ["ready"], conflicting_goal_ids: [], created_at: "2026-01-01T00:00:00Z", secret: "PRIVATE_SENTINEL" }],
  intrinsic_deliberations: [{ private_replay: "PRIVATE_SENTINEL" }],
};

const commitmentPayload = {
  commitments: [{ commitment_id: "commitment-1", description: "Report results", related_goal_id: "goal-1", status: "active", beneficiary: "operator", scope: "Release report", deadline: null, cost: 0.2, burden: 0.1, fulfillability: "fulfillable", fulfillability_reason: "Resources available", decision_refs: ["decision-1"], created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", hidden_thought: "PRIVATE_SENTINEL" }],
};

const planPayload = {
  plans: [{ plan_id: "plan-1", goal_id: "goal-1", revision: 1, status: "active", revisions: [{ revision: 1, raw_prompt: "PRIVATE_SENTINEL", steps: [{ step_id: "step-1", action_type: "respond", action_code: "report.result", dependency_ids: [], parameters: { private_replay: "PRIVATE_SENTINEL" } }] }], step_states: [{ step_id: "step-1", status: "in_progress", attempt_count: 1, started_at: "2026-01-01T00:00:00Z", retry_at: null, completed_at: null, evidence: [{ secret: "PRIVATE_SENTINEL" }] }], created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }],
};

const decisionPayload = {
  decisions: [{ decision_id: "decision-1", context_id: "context-1", active_goal_ids: ["goal-1"], selected_candidate_id: "candidate-1", considered_candidates: [{ candidate: { candidate_id: "candidate-1", candidate_type: "respond", proposed_action: "Report result", plan_id: "plan-1", plan_revision: 1, step_id: "step-1", goal_refs: ["goal-1"], commitment_refs: ["commitment-1"], parameters: { raw_prompt: "PRIVATE_SENTINEL" } }, hidden_thought: "PRIVATE_SENTINEL" }], selection_confidence: 0.8, status: "awaiting_outcome", actual_outcome: null, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", private_replay: "PRIVATE_SENTINEL" }],
};

const workingMemoryPayload = { item_count: 3, token_count: 120, item_capacity: 12, token_capacity: 2000, items: [{ hidden_thought: "PRIVATE_SENTINEL" }] };

const cockpitOutboxPayload = {
  messages: [{
    message_id: "message-1",
    title: "Release ready",
    urgency: "critical",
    delivery_status: "pending",
    acknowledgment_status: "unacknowledged",
    references: { event_id: null, goal_id: "goal-1", plan_id: "plan-1", decision_id: "decision-1", action_id: "action-1", commitment_id: "commitment-1" },
    body: "PRIVATE_SENTINEL",
    responses: [{ text: "PRIVATE_SENTINEL" }],
    attempts: [{ failure_code: "PRIVATE_SENTINEL" }],
  }],
};
