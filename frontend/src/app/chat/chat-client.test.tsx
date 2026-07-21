import { render, screen } from "@testing-library/react";
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
    expect(fetchMock.mock.calls[0][0]).toBe("/api-proxy/chat");
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
