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

describe("api client", () => {
  it("/chat sends requests to /api/chat", async () => {
    fetchMock.mockReturnValue(jsonResponse({ episode_id: "e", response: "ok", emotion: { valence: 0, arousal: 0, optimal_loss: 1 }, model: { model_id: "m", adapter_id: null } }));

    await api.chat({ message: "hello", attachments: [], debug: false });

    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8000/api/chat", expect.objectContaining({ method: "POST" }));
  });

  it("/debug sends requests to /api/chat/debug", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.debugChat({ message: "hello", attachments: [], debug: true });

    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8000/api/chat/debug", expect.objectContaining({ method: "POST" }));
  });

  it("adapter actions call backend lifecycle endpoints", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.evaluateAdapter("a");
    await api.trialAdapter("a");
    await api.approveAdapter("a");
    await api.activateAdapter("a");
    await api.rejectAdapter("a");

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "http://127.0.0.1:8000/api/adapters/a/evaluate",
      "http://127.0.0.1:8000/api/adapters/a/trial",
      "http://127.0.0.1:8000/api/adapters/a/approve",
      "http://127.0.0.1:8000/api/adapters/a/activate",
      "http://127.0.0.1:8000/api/adapters/a/reject",
    ]);
  });

  it("sleep page action calls /api/sleep/run", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.sleepRun();

    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8000/api/sleep/run", expect.objectContaining({ method: "POST" }));
  });
});
