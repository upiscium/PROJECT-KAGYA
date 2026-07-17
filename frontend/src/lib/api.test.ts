import { describe, expect, it, vi, beforeEach } from "vitest";
import { api } from "./api";

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

  it("formats backend JSON error details", async () => {
    fetchMock.mockReturnValue(errorResponse(500, "Internal Server Error", { detail: "Fallback model produced an empty visible response" }));

    await expect(api.chat({ text: "hello" })).rejects.toThrow("Backend failed: Fallback model produced an empty visible response");
  });

  it("formats unavailable backend errors", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    await expect(api.chat({ text: "hello" })).rejects.toThrow("Backend unavailable");
  });
});
