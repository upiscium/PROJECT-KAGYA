import { createHash } from "node:crypto";
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
      [api.actionTrace, actionTracePayload, "/admin-proxy/actions/trace", "traces"],
      [api.cockpitTraining, cockpitTrainingPayload, "/admin-proxy/training/cockpit-summary", "nodes"],
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
    expect(result).toEqual({ pending_count: 42, critical_count: 9, messages: [{
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
    ["messages collection", { ...cockpitOutboxPayload, messages: {} }],
    ["message ID", { ...cockpitOutboxPayload, messages: [{ ...cockpitOutboxPayload.messages[0], message_id: undefined }] }],
    ["urgency", { ...cockpitOutboxPayload, messages: [{ ...cockpitOutboxPayload.messages[0], urgency: "urgent" }] }],
    ["delivery status", { ...cockpitOutboxPayload, messages: [{ ...cockpitOutboxPayload.messages[0], delivery_status: "invented" }] }],
    ["acknowledgment status", { ...cockpitOutboxPayload, messages: [{ ...cockpitOutboxPayload.messages[0], acknowledgment_status: "invented" }] }],
    ["reference", { ...cockpitOutboxPayload, messages: [{ ...cockpitOutboxPayload.messages[0], references: { ...cockpitOutboxPayload.messages[0].references, goal_id: 7 } }] }],
    ["negative pending count", { ...cockpitOutboxPayload, pending_count: -1 }],
    ["fractional critical count", { ...cockpitOutboxPayload, critical_count: 1.5 }],
    ["missing pending count", { critical_count: 9, messages: cockpitOutboxPayload.messages }],
    ["missing critical count", { pending_count: 42, messages: cockpitOutboxPayload.messages }],
  ])("rejects malformed cockpit outbox %s", async (_label, payload) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(api.cockpitOutbox()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("parses only public-safe action trace fields", async () => {
    fetchMock.mockReturnValue(jsonResponse(actionTracePayload));

    const result = await api.actionTrace();

    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/actions/trace", expect.anything());
    expect(result.pending_approval_count).toBe(2);
    expect(result.retry_pending_count).toBe(1);
    expect(result.failed_count).toBe(3);
    expect(result.traces[0].receipt?.status).toBe("succeeded");
    expect(result.traces[0].related_receipts).toEqual([{ receipt_id: "receipt-original", status: "succeeded" }]);
    expect(result.traces[0].observation?.result_digest).toBe("a".repeat(64));
    expect(result.pre_intent_failures.map((failure) => failure.failure_type)).toEqual(["validation", "policy_rejection"]);
    expect(result.pre_intent_failures[0].tool_name).toBe("document_search");
    expect(result.pre_intent_failures[1].tool_name).toBeNull();
    expect(result.pre_intent_failures[1].candidate_id).toBe("candidate-2");
    expect(JSON.stringify(result)).not.toContain("PRIVATE_SENTINEL");
    expect(result.traces[0]).not.toHaveProperty("arguments");
    expect(result.traces[0]).not.toHaveProperty("preview");
    expect(result.traces[0]).not.toHaveProperty("idempotency_key");
    expect(result.traces[0].approval).not.toHaveProperty("reason");
    expect(result.traces[0].observation).not.toHaveProperty("data");
  });

  it("parses only public-safe cockpit training summary fields", async () => {
    fetchMock.mockReturnValue(jsonResponse(cockpitTrainingPayload));

    const result = await api.cockpitTraining();

    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/training/cockpit-summary", expect.anything());
    expect(result.node_count).toBe(1);
    expect(result.nodes[0].status).toBe("online");
    expect(result.jobs[0].failure_code).toBeNull();
    expect(result.adapters[0].activation_event_id).toBe("event-1");
    expect(JSON.stringify(result)).not.toContain("PRIVATE_SENTINEL");
    expect(result.jobs[0]).not.toHaveProperty("stderr");
    expect(result.adapters[0]).not.toHaveProperty("private_path");
  });

  it.each([
    ["unavailable worker", { ...cockpitTrainingPayload, nodes: [{ ...cockpitTrainingPayload.nodes[0], status: "unavailable", last_contact_at: null, observed_model_id: null, observed_model_revision: null, model_matches_expected: null }] }],
    ["split projection", { ...cockpitTrainingPayload, nodes: [{ ...cockpitTrainingPayload.nodes[0], role: "inference" }], jobs: [{ ...cockpitTrainingPayload.jobs[0], worker_node_id: null, candidate_adapter_id: null }], adapters: [{ ...cockpitTrainingPayload.adapters[0], training_job_id: null, training_node_id: null }] }],
  ])("accepts cockpit training %s", async (_label, payload) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(api.cockpitTraining()).resolves.toHaveProperty("nodes");
  });

  it.each([
    ["root", null],
    ["unknown field", { ...cockpitTrainingPayload, private: "PRIVATE_SENTINEL" }],
    ["missing field", { nodes: cockpitTrainingPayload.nodes, jobs: cockpitTrainingPayload.jobs, adapters: cockpitTrainingPayload.adapters }],
    ["malformed enum", { ...cockpitTrainingPayload, jobs: [{ ...cockpitTrainingPayload.jobs[0], status: "invented" }] }],
    ["negative count", { ...cockpitTrainingPayload, failed_job_count: -1 }],
    ["invalid digest", { ...cockpitTrainingPayload, jobs: [{ ...cockpitTrainingPayload.jobs[0], bundle_digest: "bad" }] }],
    ["invalid timestamp", { ...cockpitTrainingPayload, nodes: [{ ...cockpitTrainingPayload.nodes[0], last_contact_at: "not-time" }] }],
    ["unsafe code", { ...cockpitTrainingPayload, jobs: [{ ...cockpitTrainingPayload.jobs[0], failure_code: "raw worker stderr PRIVATE_SENTINEL" }] }],
    ["unsafe GPU", { ...cockpitTrainingPayload, nodes: [{ ...cockpitTrainingPayload.nodes[0], gpu_name: "<script>PRIVATE_SENTINEL</script>" }] }],
  ])("rejects malformed cockpit training %s", async (_label, payload) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(api.cockpitTraining()).rejects.toMatchObject({ name: "ApiError" });
  });

  it.each([
    ["root", null],
    ["traces collection", { ...actionTracePayload, traces: {} }],
    ["intent ID", actionTraceWith({ intent_id: undefined })],
    ["intent status", actionTraceWith({ status: "invented" })],
    ["risk class", actionTraceWith({ risk_class: "invented" })],
    ["receipt status", actionTraceWith({ receipt: { ...actionTracePayload.traces[0].receipt, status: "invented" } })],
    ["negative count", { ...actionTracePayload, failed_count: -1 }],
    ["fractional count", { ...actionTracePayload, retry_pending_count: 1.5 }],
    ["duration", actionTraceWith({ receipt: { ...actionTracePayload.traces[0].receipt, duration_ms: -0.1 } })],
    ["digest", actionTraceWith({ observation: { ...actionTracePayload.traces[0].observation, result_digest: "invalid" } })],
    ["event sequence", actionTraceWith({ receipt: { ...actionTracePayload.traces[0].receipt, event_sequence: 0 } })],
    ["validation errors", actionTraceWith({ observation: { ...actionTracePayload.traces[0].observation, validation_errors: {} } })],
    ["failure type", actionFailureWith({ failure_type: "invented" })],
    ["failure error code", actionFailureWith({ error_codes: ["not bounded code"] })],
    ["failure event sequence", actionFailureWith({ event_sequence: 0 })],
    ["failure decision ID", actionFailureWith({ decision_id: undefined })],
    ["tool whitespace", actionFailureWith({ tool_name: "tool name" })],
    ["tool newline", actionFailureWith({ tool_name: "tool\nname" })],
    ["tool HTML", actionFailureWith({ tool_name: "<script>alert(1)</script>" })],
    ["tool uppercase", actionFailureWith({ tool_name: "Document_Search" })],
    ["tool hyphen", actionFailureWith({ tool_name: "document-search" })],
    ["tool private sentinel", actionFailureWith({ tool_name: "PRIVATE_SENTINEL" })],
    ["intent failure code", actionTraceWith({ failure_code: "not bounded code" })],
    ["receipt error code", actionTraceWith({ receipt: { ...actionTracePayload.traces[0].receipt, error_code: "not bounded code" } })],
    ["verification reason", actionTraceWith({ verification: { ...actionTracePayload.traces[0].verification, reason: "not bounded code" } })],
  ])("rejects malformed action trace %s", async (_label, payload) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(api.actionTrace()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("accepts backend fail-closed cross-record projections", async () => {
    const payload = {
      ...actionTracePayload,
      traces: [{
        ...actionTracePayload.traces[0],
        receipt: null,
        related_receipts: [],
        observation: null,
        verification: null,
      }],
      pre_intent_failures: [{
        ...actionTracePayload.pre_intent_failures[1],
        tool_name: null,
      }],
    };
    fetchMock.mockReturnValue(jsonResponse(payload));

    const result = await api.actionTrace();

    expect(result.traces[0].receipt).toBeNull();
    expect(result.traces[0].related_receipts).toEqual([]);
    expect(result.traces[0].observation).toBeNull();
    expect(result.traces[0].verification).toBeNull();
    expect(result.pre_intent_failures[0].tool_name).toBeNull();
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

  it("strictly parses the action operator contract and rejects raw private fields", async () => {
    fetchMock.mockReturnValue(jsonResponse(actionOperatorPayload));
    const result = await api.actionOperatorSummary();
    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/actions/operator-summary", expect.anything());
    expect(result.actions[0].argument_summary).toEqual(expect.objectContaining({ kind: "notification", title: "Safe title" }));
    expect(JSON.stringify(result)).not.toContain("PRIVATE_SENTINEL");

    fetchMock.mockReturnValue(jsonResponse({
      ...actionOperatorPayload,
      actions: [{ ...actionOperatorPayload.actions[0], arguments: { body: "PRIVATE_SENTINEL" } }],
    }));
    await expect(api.actionOperatorSummary()).rejects.toMatchObject({ name: "ApiError" });
  });

  it.each([
    ["candidate argument", { arguments: { body: "PRIVATE_SENTINEL" } }],
    ["preview argument", { preview: { ...actionOperatorPayload.actions[0].preview, arguments: { body: "PRIVATE_SENTINEL" } } }],
    ["candidate preview field", { preview: { ...actionOperatorPayload.actions[0].preview, private_replay: "PRIVATE_SENTINEL" } }],
  ])("rejects private sentinel in unknown action %s fields", async (_label, update) => {
    fetchMock.mockReturnValue(jsonResponse({
      ...actionOperatorPayload,
      actions: [{ ...actionOperatorPayload.actions[0], ...update }],
    }));
    await expect(api.actionOperatorSummary()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("binds approval mutations to revision, digest, and approval ID", async () => {
    fetchMock.mockReturnValue(jsonResponse({ command: "approve", event_id: "event-operator-1", processing_sequence: 12, action: actionOperatorPayload.actions[0], disposition: "awaiting_scheduler" }));
    await api.approveAction("intent-operator-1", {
      expected_intent_revision: 3,
      expected_preview_digest: "a".repeat(64),
      expected_approval_id: "approval-operator-1",
    });
    const [, options] = fetchMock.mock.calls[0];
    expect(fetchMock.mock.calls[0][0]).toBe("/admin-proxy/actions/operator/intents/intent-operator-1/approval");
    expect(JSON.parse(String(options.body))).toEqual({ expected_intent_revision: 3, expected_preview_digest: "a".repeat(64), expected_approval_id: "approval-operator-1", approved: true });
  });

  it.each([
    ["reject", api.rejectAction, "/admin-proxy/actions/operator/intents/intent-operator-1/approval", { approved: false, reason: "not yet" }],
    ["cancel", api.cancelAction, "/admin-proxy/actions/operator/intents/intent-operator-1/cancel", {}],
    ["retry", api.retryAction, "/admin-proxy/actions/operator/intents/intent-operator-1/retry", {}],
    ["compensate", api.compensateAction, "/admin-proxy/actions/operator/intents/intent-operator-1/compensate", {}],
  ] as const)("sends the exact %s mutation endpoint and body", async (_label, client, path, extra) => {
    fetchMock.mockReturnValue(jsonResponse({ command: _label === "retry" ? "retry_now" : _label, event_id: "event-operator-1", processing_sequence: 12, action: actionOperatorPayload.actions[0], disposition: _label === "reject" ? "rejected" : "cancelled" }));
    await client("intent-operator-1", { expected_intent_revision: 3, expected_preview_digest: "a".repeat(64), ...extra } as never);
    const [, options] = fetchMock.mock.calls[0];
    expect(fetchMock.mock.calls[0][0]).toBe(path);
    expect(JSON.parse(String(options.body))).toEqual({ expected_intent_revision: 3, expected_preview_digest: "a".repeat(64), ...extra });
  });

  it("accepts only the safe outbox projection", async () => {
    const safe = {
      message_id: "message-safe-1", kind: "approval_request", title: "Approval required", urgency: "high",
      delivery_status: "pending", acknowledgment_status: "unacknowledged", created_at: "2026-01-01T00:00:00Z",
      channel: "local", privacy_class: "operator", last_failure_code: null, body_preview: null,
      references: { event_id: null, goal_id: null, plan_id: "plan-1", decision_id: "decision-1", action_id: "intent-operator-1", commitment_id: null },
    };
    fetchMock.mockReturnValue(jsonResponse({ messages: [safe] }));
    expect((await api.outboxMessages()).messages[0].kind).toBe("approval_request");

    fetchMock.mockReturnValue(jsonResponse({ messages: [{ ...safe, body: "PRIVATE_SENTINEL" }] }));
    await expect(api.outboxMessages()).rejects.toMatchObject({ name: "ApiError" });

    fetchMock.mockReturnValue(jsonResponse({ messages: [{ ...safe, kind: "question", body_preview: "Which local option should be used?" }] }));
    expect((await api.outboxMessages()).messages[0].body_preview).toBe("Which local option should be used?");

    fetchMock.mockReturnValue(jsonResponse({ messages: [{ ...safe, kind: "renegotiation", body_preview: "Move the deadline by one day." }] }));
    expect((await api.outboxMessages()).messages[0].body_preview).toBe("Move the deadline by one day.");

    for (const body_preview of ["PRIVATE_SENTINEL", "hidden thought", "credential=secret", "x".repeat(161)]) {
      fetchMock.mockReturnValue(jsonResponse({ messages: [{ ...safe, kind: "question", body_preview }] }));
      await expect(api.outboxMessages()).rejects.toMatchObject({ name: "ApiError" });
    }

    fetchMock.mockReturnValue(jsonResponse({ messages: [{ ...safe, body_preview: "Action details" }] }));
    await expect(api.outboxMessages()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("does not retain a raw or deterministic document query fingerprint", async () => {
    const query = "release status";
    const queryHash = createHash("sha256").update(query).digest("hex");
    const documentAction = {
      ...actionOperatorPayload.actions[0],
      tool: { ...actionOperatorPayload.actions[0].tool, name: "document_search", risk_class: "read_only", approval_required: false, reversible: false, effect_code: "documents.search" },
      argument_summary: { kind: "document_search", scope_kind: "all", max_results: 5, query_length: query.length },
      approval: null,
      available_commands: [],
    };
    fetchMock.mockReturnValue(jsonResponse({ ...actionOperatorPayload, actions: [documentAction] }));

    const result = await api.actionOperatorSummary();
    const serialized = JSON.stringify(result);
    expect(serialized).not.toContain(query);
    expect(serialized).not.toContain(queryHash);

    fetchMock.mockReturnValue(jsonResponse({ ...actionOperatorPayload, actions: [{ ...documentAction, argument_summary: { ...documentAction.argument_summary, query_digest: queryHash } }] }));
    await expect(api.actionOperatorSummary()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("accepts only safe bounded registry descriptions", async () => {
    const safe = { ...actionOperatorPayload, registry_tools: [{ ...actionOperatorPayload.registry_tools[0], description: "Reads public metadata." }] };
    fetchMock.mockReturnValue(jsonResponse(safe));
    expect((await api.actionOperatorSummary()).registry_tools[0].description).toBe("Reads public metadata.");

    for (const description of [
      "x".repeat(161),
      "hidden thought: PRIVATE_SENTINEL",
      "See /home/kagya/private/config.yaml",
      "credential token secret",
      "SCHEMA_SENTINEL",
    ]) {
      fetchMock.mockReturnValue(jsonResponse({ ...actionOperatorPayload, registry_tools: [{ ...actionOperatorPayload.registry_tools[0], description }] }));
      await expect(api.actionOperatorSummary()).rejects.toMatchObject({ name: "ApiError" });
    }
  });

  it("sanitizes private conflict bodies into bounded public ApiErrors", async () => {
    const secret = `PRIVATE_SENTINEL ${"x".repeat(2000)}`;
    fetchMock.mockReturnValue(errorResponse(409, "Conflict", { detail: secret, hidden_thought: secret }));

    const error = await api.cancelAction("intent-operator-1", {
      expected_intent_revision: 3,
      expected_preview_digest: "a".repeat(64),
    }).catch((value) => value);
    expect(error).toMatchObject({ name: "ApiError", status: 409 });
    expect(String(error)).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(error)).not.toContain("PRIVATE_SENTINEL");
    expect(String(error).length).toBeLessThan(500);
  });

  it.each([
    ["arguments", { arguments: { value: "PRIVATE_SENTINEL" } }],
    ["preview", { preview: { ...actionOperatorPayload.actions[0].preview, raw_prompt: "PRIVATE_SENTINEL" } }],
  ])("rejects unsafe unknown %s fields in successful action mutation responses", async (_label, unsafe) => {
    fetchMock.mockReturnValue(jsonResponse({
      command: "cancel",
      event_id: "event-operator-unsafe",
      processing_sequence: 12,
      action: { ...actionOperatorPayload.actions[0], ...unsafe },
      disposition: "cancelled",
    }));

    await expect(api.cancelAction("intent-operator-1", {
      expected_intent_revision: 3,
      expected_preview_digest: "a".repeat(64),
    })).rejects.toMatchObject({ name: "ApiError" });
  });

  it("calls governed restore summary and preview endpoints and parses their public responses", async () => {
    fetchMock.mockReturnValueOnce(jsonResponse(operatorRestoreSummaryPayload));
    const summary = await api.operatorRestoreSummary(25);
    expect(fetchMock).toHaveBeenLastCalledWith("/admin-proxy/state/operator-restore/summary?limit=25", expect.anything());
    expect(summary.targets[0]).toEqual(operatorRestoreSummaryPayload.targets[0]);
    expect(summary.latest_operation?.external_side_effects_replayed).toBe(false);
    expect(JSON.stringify(summary)).not.toContain("PRIVATE_SENTINEL");

    fetchMock.mockReturnValueOnce(jsonResponse(operatorRestorePreviewPayload));
    const preview = await api.previewOperatorRestore(7);
    expect(fetchMock).toHaveBeenLastCalledWith("/admin-proxy/state/operator-restore/preview/7", expect.anything());
    expect(preview.external_effects.artifacts[0].refs).toEqual([externalArtifactHandle]);
    expect(preview.confirmation_phrase).toBe("RESTORE TARGET 7");
    expect(preview.external_side_effects_replayed).toBe(false);
  });

  it("serializes the exact governed restore commit body and parses the response", async () => {
    fetchMock.mockReturnValue(jsonResponse(operatorRestoreCommitPayload));
    const body = {
      target_sequence: 7,
      expected_target_hash: "a".repeat(64),
      expected_semantic_revision: 12,
      expected_current_logical_digest: "b".repeat(64),
      expected_preview_digest: "c".repeat(64),
      expected_external_effect_digest: "d".repeat(64),
      confirmation_phrase: "RESTORE TARGET 7",
    };

    const response = await api.commitOperatorRestore(body);
    const [, options] = fetchMock.mock.calls[0];
    expect(fetchMock.mock.calls[0][0]).toBe("/admin-proxy/state/operator-restore/commit");
    expect(options.method).toBe("POST");
    expect(JSON.parse(String(options.body))).toEqual(body);
    expect(response).toEqual(operatorRestoreCommitPayload);
    expect(response.external_side_effects_replayed).toBe(false);
  });

  it.each([
    ["unknown summary field", { ...operatorRestoreSummaryPayload, private: "PRIVATE_SENTINEL" }, api.operatorRestoreSummary],
    ["unknown preview field", { ...operatorRestorePreviewPayload, private: "PRIVATE_SENTINEL" }, () => api.previewOperatorRestore(7)],
    ["unknown commit response field", { ...operatorRestoreCommitPayload, private: "PRIVATE_SENTINEL" }, () => api.commitOperatorRestore(restoreCommitRequest)],
  ])("rejects %s", async (_label, payload, client) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(client()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("rejects bounded and private restore fields", async () => {
    const cases: Array<[unknown, () => Promise<unknown>]> = [
      [{ ...operatorRestoreSummaryPayload, targets: [{ ...operatorRestoreSummaryPayload.targets[0], reason_codes: ["x".repeat(129)] }] }, () => api.operatorRestoreSummary()],
      [{ ...operatorRestorePreviewPayload, confirmation_phrase: "PRIVATE_SENTINEL" }, () => api.previewOperatorRestore(7)],
      [{ ...operatorRestorePreviewPayload, domains: [{ ...operatorRestorePreviewPayload.domains[0], domain: "x".repeat(129) }] }, () => api.previewOperatorRestore(7)],
      [{ ...operatorRestoreCommitPayload, error_code: "not a bounded code" }, () => api.commitOperatorRestore(restoreCommitRequest)],
    ];
    for (const [payload, client] of cases) {
      fetchMock.mockReturnValue(jsonResponse(payload));
      await expect(client()).rejects.toMatchObject({ name: "ApiError" });
    }
  });

  it.each([
    ["goal", { refs: [{ kind: "goal", id: "PRIVATE_SENTINEL" }] }],
    ["human-readable goal", { refs: [{ kind: "goal", id: "AliceSmith1" }] }],
    ["prefixed human-readable goal", { refs: [{ kind: "goal", id: "goal-AliceSmith-1" }] }],
    ["decision", { refs: [{ kind: "decision", id: "hidden_thought" }] }],
    ["intent", { refs: [{ kind: "action", id: "credential_token" }] }],
    ["message", { refs: [{ kind: "outbox", id: "api_secret" }] }],
    ["memory", { refs: [{ kind: "memory", id: "password" }] }],
    ["belief", { refs: [{ kind: "belief", id: "raw_prompt" }] }],
    ["event", { refs: [{ kind: "journal", id: "event-prompt-1" }] }],
    ["path", { refs: [{ kind: "outbox", id: "/var/lib/kagya/message-1" }] }],
    ["control", { refs: [{ kind: "memory", id: "memory-1\u0000" }] }],
    ["format", { refs: [{ kind: "journal", id: "event\u202e-1" }] }],
  ])("rejects unsafe public restore %s references", async (_label, refs) => {
    fetchMock.mockReturnValue(jsonResponse({
      ...operatorRestorePreviewPayload,
      domains: [{ ...operatorRestorePreviewPayload.domains[0], refs }],
    }));
    await expect(api.previewOperatorRestore(7)).rejects.toMatchObject({ name: "ApiError" });
  });

  it.each([[[]], [[externalArtifactHandle]]])("accepts empty or opaque external artifact references", async (refs) => {
    fetchMock.mockReturnValue(jsonResponse({
      ...operatorRestorePreviewPayload,
      external_effects: {
        ...operatorRestorePreviewPayload.external_effects,
        artifacts: [{ ...operatorRestorePreviewPayload.external_effects.artifacts[0], refs }],
      },
    }));
    await expect(api.previewOperatorRestore(7)).resolves.toMatchObject({ external_effects: { artifacts: [{ refs }] } });
  });

  it.each(["artifact-transaction-1", `sha256:${restoreDigestA}`, "PRIVATE_SENTINEL"])("rejects a raw or non-opaque external artifact reference: %s", async (reference) => {
    fetchMock.mockReturnValue(jsonResponse({
      ...operatorRestorePreviewPayload,
      external_effects: {
        ...operatorRestorePreviewPayload.external_effects,
        artifacts: [{ ...operatorRestorePreviewPayload.external_effects.artifacts[0], refs: [reference] }],
      },
    }));
    await expect(api.previewOperatorRestore(7)).rejects.toMatchObject({ name: "ApiError" });
  });

  it.each([
    ["summary operation UUID", { ...operatorRestoreSummaryPayload, latest_operation: { ...operatorRestoreSummaryPayload.latest_operation, operation_id: "restore-op-1" } }, api.operatorRestoreSummary],
    ["summary operation event binding", { ...operatorRestoreSummaryPayload, latest_operation: { ...operatorRestoreSummaryPayload.latest_operation, event_id: "operator-restore-22222222-2222-2222-2222-222222222222" } }, api.operatorRestoreSummary],
    ["commit operation UUID", { ...operatorRestoreCommitPayload, operation_id: "7" }, () => api.commitOperatorRestore(restoreCommitRequest)],
    ["commit event binding", { ...operatorRestoreCommitPayload, event_id: "operator-restore-22222222-2222-2222-2222-222222222222" }, () => api.commitOperatorRestore(restoreCommitRequest)],
  ])("rejects restore operation forgery: %s", async (_label, payload, client) => {
    fetchMock.mockReturnValue(jsonResponse(payload));
    await expect(client()).rejects.toMatchObject({ name: "ApiError" });
  });

  it("sanitizes governed restore HTTP errors", async () => {
    const secret = `PRIVATE_SENTINEL ${"x".repeat(2000)}`;
    fetchMock.mockReturnValue(errorResponse(409, "Conflict", { detail: secret, hidden_thought: secret }));

    const error = await api.commitOperatorRestore(restoreCommitRequest).catch((value) => value);
    expect(error).toMatchObject({ name: "ApiError", status: 409, detail: "conflict" });
    expect(String(error)).toBe("ApiError: Restore request failed (409).");
    expect(JSON.stringify(error)).not.toContain("PRIVATE_SENTINEL");
  });

  it("preserves only allowlisted governed restore error codes", async () => {
    fetchMock.mockReturnValue(errorResponse(503, "Unavailable", { detail: { code: "commit_indeterminate" } }));
    await expect(api.commitOperatorRestore(restoreCommitRequest)).rejects.toMatchObject({
      name: "ApiError",
      status: 503,
      detail: "commit_indeterminate",
    });
  });
});

const actionOperatorPayload = {
  pending_approval_count: 1,
  operator_action_count: 1,
  risk_ceiling: "reversible_write",
  actions: [{
    intent_id: "intent-operator-1", revision: 3, status: "awaiting_approval",
    approval: { approval_id: "approval-operator-1", status: "pending", requested_at: "2026-01-01T00:00:00Z" },
    tool: { name: "local_notification_enqueue", risk_class: "reversible_write", approval_required: true, reversible: true, effect_code: "notification.enqueue", validation_schema_revision: "b".repeat(64), enabled: true, executable: true, execution_authority: "action_execution" },
    argument_summary: { kind: "notification", channel: "local", title: "Safe title", body_preview: "Safe preview" },
    policy: { allowed: true, approval_required: true, reason_codes: ["human_approval_required"] },
    preview: { effect_code: "notification.enqueue", effect: "Enqueue a notification", digest: "a".repeat(64), compensation_available: true },
    budget: { max_attempts: 2, max_cost_units: 1, max_monetary_cost: 0, deadline_at: "2026-01-02T00:00:00Z", attempts: 0, cost_units_used: 0, retry_at: null },
    provenance: { decision_id: "decision-1", plan_id: "plan-1", plan_revision: 1, step_id: "step-1", triggering_event_id: "event-1" },
    receipt: null, verification: null, idempotency_state: "reserved", available_commands: ["approve", "reject", "cancel"], confirmation: null,
  }],
  action_tools: [{ name: "local_notification_enqueue", risk_class: "reversible_write", approval_required: true, reversible: true, effect_code: "notification.enqueue", validation_schema_revision: "b".repeat(64), enabled: true, executable: true, execution_authority: "action_execution" }],
  registry_tools: [{ name: "registry-tool", description: null, tool_type: "metadata", status: "declared", generated: false, human_approved: false, execution_authority: "registry_only" }],
};

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
  pending_count: 42,
  critical_count: 9,
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

const actionTracePayload = {
  pending_approval_count: 2,
  retry_pending_count: 1,
  failed_count: 3,
  traces: [{
    intent_id: "action-1",
    revision: 2,
    tool_name: "document_search",
    risk_class: "read_only",
    status: "succeeded",
    dry_run: false,
    created_at: "2026-07-30T00:00:00Z",
    updated_at: "2026-07-30T00:00:01Z",
    failure_code: null,
    provenance: { decision_id: "decision-1", candidate_id: "candidate-1", triggering_event_id: "event-0", plan_id: "plan-1", plan_revision: 1, step_id: "step-1" },
    approval: { approval_id: "approval-1", status: "approved", requested_at: "2026-07-30T00:00:00Z", resolved_at: "2026-07-30T00:00:01Z", resolved_by_operator: true, reason: "PRIVATE_SENTINEL" },
    receipt: { receipt_id: "receipt-1", status: "succeeded", attempt: 1, duration_ms: 12.5, event_id: "event-1", event_sequence: 45, error_code: null, compensation_of: null, idempotency_key: "PRIVATE_SENTINEL" },
    related_receipts: [{ receipt_id: "receipt-original", status: "succeeded", idempotency_key: "PRIVATE_SENTINEL" }],
    observation: { observation_id: "observation-1", valid: true, validation_errors: [], result_digest: "a".repeat(64), data: { result: "PRIVATE_SENTINEL" } },
    verification: { verification_id: "verification-1", success: true, reason: "observation_schema_valid", private_replay: "PRIVATE_SENTINEL" },
    arguments: { query: "PRIVATE_SENTINEL" },
    preview: { arguments: { query: "PRIVATE_SENTINEL" } },
    idempotency_key: "PRIVATE_SENTINEL",
  }],
  pre_intent_failures: [
    { failure_id: "validation-1", failure_type: "validation", decision_id: "decision-1", candidate_id: null, tool_name: "document_search", risk_class: "read_only", error_codes: ["arguments_schema_invalid"], event_id: "event-1", event_sequence: 42, occurred_at: "2026-07-30T00:00:00Z", idempotency_key: "PRIVATE_SENTINEL", request_digest: "PRIVATE_SENTINEL", canonical_arguments_digest: "PRIVATE_SENTINEL", arguments: { secret: "PRIVATE_SENTINEL" } },
    { failure_id: "rejection-1", failure_type: "policy_rejection", decision_id: "decision-2", candidate_id: "candidate-2", tool_name: null, risk_class: "reversible_write", error_codes: ["risk_class_exceeds_budget"], event_id: "event-2", event_sequence: 43, occurred_at: "2026-07-30T00:00:01Z", idempotency_key: "PRIVATE_SENTINEL" },
  ],
};

const cockpitTrainingPayload = {
  node_count: 1,
  online_node_count: 1,
  running_job_count: 1,
  failed_job_count: 0,
  importing_job_count: 0,
  active_adapter_count: 1,
  candidate_adapter_count: 1,
  nodes: [{
    node_id: "node-1",
    role: "worker",
    backend: "ssh",
    status: "online",
    last_contact_at: "2026-01-01T00:00:00Z",
    expected_model_id: "model-1",
    expected_model_revision: "rev-1",
    expected_processor_revision: "proc-1",
    observed_model_id: "model-1",
    observed_model_revision: "rev-1",
    model_matches_expected: true,
    gpu_name: "NVIDIA GPU 1",
    cuda_version: "12.1",
    driver_version: "550",
  }],
  jobs: [{
    job_id: "job-1",
    attempt_id: "attempt-1",
    status: "running",
    backend: "ssh",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:01Z",
    started_at: "2026-01-01T00:00:01Z",
    completed_at: null,
    source_event_start: 1,
    source_event_end: 5,
    selected_episode_count: 3,
    remote_job_id: "job-1",
    worker_node_id: "node-1",
    retry_count: 0,
    transferred_bytes: 2048,
    failure_code: null,
    candidate_adapter_id: "adapter-1",
    import_status: "not_started",
    bundle_digest: "a".repeat(64),
    result_digest: "b".repeat(64),
  }],
  adapters: [{
    adapter_id: "adapter-1",
    status: "active",
    adapter_hash: "b".repeat(64),
    base_model_id: "model-1",
    base_model_revision: "rev-1",
    parent_adapter_id: null,
    training_job_id: "job-1",
    training_node_id: "node-1",
    submitted_by_node_id: "submitter-1",
    imported_by_node_id: "node-1",
    evaluation_id: "eval-1",
    evaluation_status: "passed",
    approved: true,
    active: true,
    rollback_candidate: false,
    activation_event_id: "event-1",
    activation_event_sequence: 1,
    rollback_event_id: null,
    rollback_event_sequence: null,
  }],
};

const restoreDigestA = "a".repeat(64);
const restoreDigestB = "b".repeat(64);
const restoreDigestC = "c".repeat(64);
const restoreDigestD = "d1e2f3a4b5c697887766554433221100fedcba98765432100123456789abcdef";
const externalArtifactHandle = "9f4a7c2e1b6d8a50392716e0f5c4b3a29182736455463728190abcdef1234567";
const restoreCommitRequest = {
  target_sequence: 7,
  expected_target_hash: restoreDigestA,
  expected_semantic_revision: 12,
  expected_current_logical_digest: restoreDigestB,
  expected_preview_digest: restoreDigestC,
  expected_external_effect_digest: restoreDigestD,
  confirmation_phrase: "RESTORE TARGET 7",
};

const operatorRestoreSummaryPayload = {
  schema_version: 1,
  current_sequence: 10,
  current_snapshot_hash: restoreDigestA,
  current_logical_digest: restoreDigestB,
  semantic_revision: 12,
  retained_min_sequence: 1,
  retained_max_sequence: 10,
  targets: [{
    target_sequence: 7,
    target_snapshot_hash: restoreDigestC,
    checkpoint_kind: "checkpoint",
    timestamp: "2026-01-01T00:00:00Z",
    event_type: "checkpoint_created",
    eligible: true,
    reason_codes: ["eligible"],
  }],
  latest_operation: {
    operation_id: "11111111-1111-1111-1111-111111111111",
    target_sequence: 7,
    target_snapshot_hash: restoreDigestC,
    preview_digest: restoreDigestD,
    requested_at: "2026-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
    event_id: "operator-restore-11111111-1111-1111-1111-111111111111",
    processing_sequence: 11,
    state: "previewed",
    error_code: null,
    external_side_effects_replayed: false,
  },
  external_side_effects_replayed: false,
};

const operatorRestorePreviewPayload = {
  schema_version: 1,
  operation_id: "11111111-1111-1111-1111-111111111111",
  preview_digest: restoreDigestD,
  created_at: "2026-01-01T00:00:00Z",
  expires_at: "2026-01-01T01:00:00Z",
  current_logical_digest: restoreDigestB,
  semantic_revision: 12,
  display_sequence: 10,
  target_sequence: 7,
  target_snapshot_hash: restoreDigestC,
  newer_authoritative_event_count: 3,
  domains: [{
    domain: "motivation",
    before_count: 2,
    after_count: 1,
    added_count: 0,
    removed_count: 1,
    changed_count: 1,
    changed_revision_count: 1,
    newer_state_loss_count: 1,
    refs: [{ kind: "goal", id: "goal-1" }],
    truncated: false,
    reason_code: null,
  }],
  external_effects: {
    consistency_status: "consistent",
    artifacts: [{ artifact_type: "outbox", count: 1, refs: [externalArtifactHandle], truncated: false }],
    retained_not_replayed_count: 1,
    pending_count: 0,
    orphaned_count: 0,
    retryable_count: 0,
    effect_digest: restoreDigestA,
    external_side_effects_replayed: false,
  },
  restoreable: true,
  reason_codes: ["eligible"],
  external_side_effects_replayed: false,
  confirmation_phrase: "RESTORE TARGET 7",
};

const operatorRestoreCommitPayload = {
  command: "restore",
  disposition: "completed",
  operation_id: "11111111-1111-1111-1111-111111111111",
  event_id: "operator-restore-11111111-1111-1111-1111-111111111111",
  processing_sequence: 12,
  restored_target_sequence: 7,
  restored_target_hash: restoreDigestC,
  post_restore_sequence: 13,
  post_restore_hash: restoreDigestA,
  operation_status: "completed",
  error_code: null,
  external_side_effects_replayed: false,
};

function actionTraceWith(update: Record<string, unknown>) {
  return { ...actionTracePayload, traces: [{ ...actionTracePayload.traces[0], ...update }] };
}

function actionFailureWith(update: Record<string, unknown>) {
  return { ...actionTracePayload, pre_intent_failures: [{ ...actionTracePayload.pre_intent_failures[0], ...update }] };
}
