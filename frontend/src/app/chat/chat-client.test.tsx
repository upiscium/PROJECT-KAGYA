import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ChatClient } from "./chat-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><ChatClient /></QueryClientProvider>);
}

describe("ChatClient", () => {
  it("renders normal chat response without hidden thought fields", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        context_id: "ctx-1",
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Visible answer",
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null, fallback_used: false },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Visible answer")).toBeInTheDocument();
    expect(screen.getByText("Primary model")).toBeInTheDocument();
    expect(screen.queryByText("Fallback model")).not.toBeInTheDocument();
    expect(screen.queryByText(/hidden_thought/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/raw prompt/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/retrieved memory/i)).not.toBeInTheDocument();
  });

  it("sends selected attachments with the chat request", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        context_id: "ctx-1",
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Visible answer",
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null, fallback_used: false },
      }),
    });
    renderWithQuery();

    await userEvent.selectOptions(screen.getByLabelText("Attachment type"), "audio");
    await userEvent.type(screen.getByPlaceholderText("Attachment URL"), "file:///tmp/sample.wav");
    await userEvent.type(screen.getByPlaceholderText("Optional attachment name"), "sample.wav");
    await userEvent.click(screen.getByRole("button", { name: "Add Attachment" }));
    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "please inspect this");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    const request = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(fetchMock.mock.calls[0][0]).toBe("/api-proxy/chat/jobs");
    expect(request).toEqual({
      text: "please inspect this",
      attachments: [{ type: "audio", url: "file:///tmp/sample.wav", name: "sample.wav" }],
      debug: false,
    });
    expect(await screen.findByText(/Attachments: sample.wav/)).toBeInTheDocument();
  });

  it("shows a clear public error when backend generation fails", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      text: async () => JSON.stringify({ detail: "Fallback model produced an empty visible response" }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Backend failed: Fallback model produced an empty visible response")).toBeInTheDocument();
    expect(screen.queryByText(/hidden_thought/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/raw prompt/i)).not.toBeInTheDocument();
  });

  it("shows fallback model usage without debug internals", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        context_id: "ctx-1",
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Fallback answer",
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E2B", adapter_id: null, fallback_used: true },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Fallback answer")).toBeInTheDocument();
    expect(screen.getAllByText("Fallback model").length).toBeGreaterThan(0);
    expect(screen.getByText("The primary model was unavailable for this response, so KAGYA used the configured fallback model without an adapter.")).toBeInTheDocument();
    expect(screen.queryByText(/hidden_thought/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/raw prompt/i)).not.toBeInTheDocument();
  });

  it("shows a generating indicator while waiting for the response", async () => {
    fetchMock.mockReturnValue(new Promise(() => undefined));
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByLabelText("KAGYA is generating a response")).toBeInTheDocument();
    expect(screen.getByText("Generating response...")).toBeInTheDocument();
  });

  it("does not offer cancellation while a response is finalizing", async () => {
    const operation = operationStatus("finalizing");
    const encoder = new TextEncoder();
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => ({ operation, status_url: "/api/chat/jobs/job-1", result_url: "/api/chat/jobs/job-1/result", events_url: "/api/chat/jobs/job-1/events", duplicate: false }) })
      .mockResolvedValueOnce({
        ok: true,
        body: new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(`id: 1\nevent: status\ndata: ${JSON.stringify(operation)}\n\n`));
          },
        }),
      });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Finalizing response...")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });

  it("ends pending UI and clears streamed text after cancellation", async () => {
    const running = operationStatus("running");
    const canceled = { ...operationStatus("canceled"), cancel_code: "client_request" as const };
    const encoder = new TextEncoder();
    let streamSignal: AbortSignal | undefined;
    let reads = 0;
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => ({ operation: running, status_url: "/api/chat/jobs/job-1", result_url: "/api/chat/jobs/job-1/result", events_url: "/api/chat/jobs/job-1/events", duplicate: false }) })
      .mockImplementationOnce((_url, init) => {
        streamSignal = init?.signal;
        return Promise.resolve({
          ok: true,
          body: {
            getReader: () => ({
              read: () => {
                reads += 1;
                if (reads === 1) return Promise.resolve({ done: false, value: encoder.encode(`id: 1\nevent: status\ndata: ${JSON.stringify(running)}\n\nid: 2\nevent: token\ndata: {"text":"partial"}\n\n`) });
                return new Promise((_resolve, reject) => streamSignal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true }));
              },
            }),
          },
        });
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ disposition: "canceled", operation: canceled }) });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("partial")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("Canceled")).toBeInTheDocument();
    expect(screen.queryByText("partial")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("KAGYA is generating a response")).not.toBeInTheDocument();
  });

  it("keeps SSE active after cancel_requested and accepts normal completion", async () => {
    const running = operationStatus("running");
    const result = chatResult("Completed after cancel race");
    const encoder = new TextEncoder();
    let stream: ReadableStreamDefaultController<Uint8Array>;
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => ({ operation: running, status_url: "/api/chat/jobs/job-1", result_url: "/api/chat/jobs/job-1/result", events_url: "/api/chat/jobs/job-1/events", duplicate: false }) })
      .mockResolvedValueOnce({ ok: true, body: new ReadableStream({ start(controller) { stream = controller; controller.enqueue(encoder.encode(`id: 1\nevent: status\ndata: ${JSON.stringify(running)}\n\n`)); } }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ disposition: "cancel_requested", operation: { ...running, cancel_requested: true } }) });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(await screen.findByText("Cancellation requested...")).toBeInTheDocument();
    stream!.enqueue(encoder.encode(`id: 2\nevent: final\ndata: ${JSON.stringify(result)}\n\n`));
    stream!.close();

    expect(await screen.findByText("Completed after cancel race")).toBeInTheDocument();
    expect(screen.queryByText("Canceled")).not.toBeInTheDocument();
  });

  it("recovers already_completed result without showing canceled", async () => {
    const running = operationStatus("running");
    const completed = { ...operationStatus("canceled"), status: "completed" as const, cancel_code: null, result_available: true };
    const result = chatResult("Recovered completed result");
    const encoder = new TextEncoder();
    let stream: ReadableStreamDefaultController<Uint8Array>;
    let resolveResult: ((value: unknown) => void) | undefined;
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => ({ operation: running, status_url: "/api/chat/jobs/job-1", result_url: "/api/chat/jobs/job-1/result", events_url: "/api/chat/jobs/job-1/events", duplicate: false }) })
      .mockResolvedValueOnce({ ok: true, body: new ReadableStream({ start(controller) { stream = controller; controller.enqueue(encoder.encode(`id: 1\nevent: status\ndata: ${JSON.stringify(running)}\n\n`)); } }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ disposition: "already_completed", operation: completed }) })
      .mockReturnValueOnce(new Promise((resolve) => { resolveResult = resolve; }));
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));
    stream!.enqueue(encoder.encode(`id: 2\nevent: final\ndata: ${JSON.stringify(result)}\n\n`));
    stream!.close();
    resolveResult!({ ok: true, json: async () => ({ operation: completed, result }) });

    expect(await screen.findByText("Recovered completed result")).toBeInTheDocument();
    expect(screen.getAllByText("Recovered completed result")).toHaveLength(1);
    expect(screen.queryByText("Canceled")).not.toBeInTheDocument();
    expect(fetchMock.mock.calls[3][0]).toBe("/api-proxy/chat/jobs/job-1/result");
  });

  it("explains when cancellation loses the finalizing race", async () => {
    const running = operationStatus("running");
    const encoder = new TextEncoder();
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => ({ operation: running, status_url: "/api/chat/jobs/job-1", result_url: "/api/chat/jobs/job-1/result", events_url: "/api/chat/jobs/job-1/events", duplicate: false }) })
      .mockResolvedValueOnce({
        ok: true,
        body: new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(`id: 1\nevent: status\ndata: ${JSON.stringify(running)}\n\n`));
          },
        }),
      })
      .mockResolvedValueOnce({ ok: false, status: 409, statusText: "Conflict", text: async () => JSON.stringify({ detail: "already_finalizing" }) });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("The response is already being committed and can no longer be canceled.")).toBeInTheDocument();
  });

  it("shows a typed failure for failed cancellation disposition", async () => {
    const running = operationStatus("running");
    const encoder = new TextEncoder();
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => ({ operation: running, status_url: "/api/chat/jobs/job-1", result_url: "/api/chat/jobs/job-1/result", events_url: "/api/chat/jobs/job-1/events", duplicate: false }) })
      .mockResolvedValueOnce({ ok: true, body: new ReadableStream({ start(controller) { controller.enqueue(encoder.encode(`id: 1\nevent: status\ndata: ${JSON.stringify(running)}\n\n`)); } }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ disposition: "failed", operation: { ...running, status: "failed" } }) });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("Cancellation failed")).toBeInTheDocument();
  });

  it("reuses the server context id for the next message", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        context_id: "ctx-stable",
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Visible answer",
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "model", adapter_id: null, fallback_used: false },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "first");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("Visible answer");
    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "second");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    const secondRequest = JSON.parse(fetchMock.mock.calls[1][1].body as string);
    expect(secondRequest.context_id).toBe("ctx-stable");
  });

  it("submits typed feedback with response provenance", async () => {
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          context_id: "ctx-feedback",
          episode_id: "episode-feedback",
          experience_id: "experience-feedback",
          response: "Visible answer",
          emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
          model: { model_id: "model", adapter_id: null, fallback_used: false },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          feedback_id: "feedback-1",
          current_revision: 1,
          revisions: [{ revision: 1, status: "active", signals: ["do_not_remember"], propagation: { training_disposition: "exclude", correction_memory_id: null } }],
        }),
      });
    renderWithQuery();
    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("Visible answer");

    await userEvent.selectOptions(screen.getByLabelText("Feedback type"), "do_not_remember");
    await userEvent.click(screen.getByRole("button", { name: "Submit feedback" }));

    expect(await screen.findByText("Feedback recorded")).toBeInTheDocument();
    expect(fetchMock.mock.calls[1][0]).toBe("/api-proxy/feedback");
    const feedbackRequest = JSON.parse(fetchMock.mock.calls[1][1].body as string);
    expect(feedbackRequest.target).toEqual({
      target_type: "response",
      target_id: "episode-feedback",
      episode_id: "episode-feedback",
      experience_id: "experience-feedback",
      context_id: "ctx-feedback",
    });
    expect(feedbackRequest.signals).toEqual(["do_not_remember"]);
    expect(feedbackRequest).not.toHaveProperty("reward");
  });
});

function operationStatus(status: "running" | "finalizing" | "canceled") {
  const terminal = status === "canceled";
  return {
    schema_version: 1 as const,
    operation_id: "job-1",
    event_id: "event-1",
    status,
    status_sequence: terminal ? 3 : 2,
    queue_position: null,
    submitted_at: "2026-01-01T00:00:00Z",
    started_at: "2026-01-01T00:00:01Z",
    finalizing_at: status === "finalizing" ? "2026-01-01T00:00:02Z" : null,
    completed_at: terminal ? "2026-01-01T00:00:03Z" : null,
    updated_at: "2026-01-01T00:00:03Z",
    error_code: null,
    cancel_code: null,
    cancel_requested: false,
    result_available: false,
  };
}

function chatResult(response: string) {
  return {
    context_id: "ctx-1",
    episode_id: "episode-1",
    experience_id: "experience-1",
    response,
    emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
    model: { model_id: "model", adapter_id: null, adapter_hash: null, activation_sequence: null, fallback_used: false },
  };
}
