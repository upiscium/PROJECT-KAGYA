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

  it("formats backend JSON error details", async () => {
    fetchMock.mockReturnValue(errorResponse(500, "Internal Server Error", { detail: "Fallback model produced an empty visible response" }));

    await expect(api.chat({ text: "hello" })).rejects.toThrow("Backend failed: Fallback model produced an empty visible response");
  });

  it("formats unavailable backend errors", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    await expect(api.chat({ text: "hello" })).rejects.toThrow("Backend unavailable");
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
