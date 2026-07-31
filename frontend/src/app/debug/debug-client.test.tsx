import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { DebugClient } from "./debug-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><DebugClient /></QueryClientProvider>);
}

describe("DebugClient", () => {
  it("renders debug-only fields", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Visible answer",
        hidden_thought: "internal thought",
        loss: 1.25,
        prompt: "raw prompt",
        attachments: [],
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null, fallback_used: false },
        retrieved_memory: { db1_results: [], db2_results: [] },
        generation_params: { max_new_tokens: 8, temperature: 0.7, top_p: 0.95, do_sample: true, repetition_penalty: 1.1, no_repeat_ngram_size: 6 },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Debug a message"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Run Debug Chat" }));

    expect(await screen.findByText("internal thought")).toBeInTheDocument();
    expect(screen.getByText("raw prompt")).toBeInTheDocument();
    expect(screen.getByText(/Loss 1.250/)).toBeInTheDocument();
    expect(screen.getByText("Fallback: no")).toBeInTheDocument();
  });

  it("sends and renders debug attachments", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Visible answer",
        hidden_thought: "internal thought",
        loss: 1.25,
        prompt: "raw prompt",
        attachments: [{ type: "audio", url: "file:///tmp/sample.wav", name: "sample.wav", content_type: "audio/wav" }],
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null, fallback_used: false },
        retrieved_memory: { db1_results: [], db2_results: [] },
        generation_params: { max_new_tokens: 8, temperature: 0.7, top_p: 0.95, do_sample: true, repetition_penalty: 1.1, no_repeat_ngram_size: 6 },
      }),
    });
    renderWithQuery();

    await userEvent.selectOptions(screen.getByLabelText("Attachment type"), "audio");
    await userEvent.type(screen.getByPlaceholderText("Attachment URL"), "file:///tmp/sample.wav");
    await userEvent.type(screen.getByPlaceholderText("Optional attachment name"), "sample.wav");
    await userEvent.click(screen.getByRole("button", { name: "Add Attachment" }));
    await userEvent.type(screen.getByPlaceholderText("Debug a message"), "please inspect this");
    await userEvent.click(screen.getByRole("button", { name: "Run Debug Chat" }));

    const request = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(fetchMock.mock.calls[0][0]).toBe("/admin-proxy/chat/debug");
    expect(request).toEqual({
      text: "please inspect this",
      attachments: [{ type: "audio", url: "file:///tmp/sample.wav", name: "sample.wav" }],
      debug: true,
    });
    expect(await screen.findAllByText(/sample.wav/)).toHaveLength(2);
    expect(screen.getByText(/audio\/wav/)).toBeInTheDocument();
  });

  it("shows empty retrieved memory states", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Visible answer",
        hidden_thought: "",
        loss: 1.25,
        prompt: "raw prompt",
        attachments: [],
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null, fallback_used: false },
        retrieved_memory: { db1_results: [], db2_results: [] },
        generation_params: { max_new_tokens: 8, temperature: 0.7, top_p: 0.95, do_sample: true, repetition_penalty: 1.1, no_repeat_ngram_size: 6 },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Debug a message"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Run Debug Chat" }));

    expect(await screen.findByText("No DB1 episodes retrieved.")).toBeInTheDocument();
    expect(screen.getByText("No DB2 semantic memories retrieved.")).toBeInTheDocument();
  });

  it("shows a clear backend error", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      text: async () => JSON.stringify({ detail: "debug service unavailable" }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Debug a message"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Run Debug Chat" }));

    expect(await screen.findByText("Chat service is temporarily unavailable: debug service unavailable")).toBeInTheDocument();
  });

  it("shows fallback model status explicitly", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        episode_id: "episode-1",
        experience_id: "experience-1",
        response: "Fallback visible answer",
        hidden_thought: "fallback thought",
        loss: 1.25,
        prompt: "raw prompt",
        attachments: [],
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E2B", adapter_id: null, fallback_used: true },
        retrieved_memory: { db1_results: [], db2_results: [] },
        generation_params: { max_new_tokens: 8, temperature: 0.7, top_p: 0.95, do_sample: true, repetition_penalty: 1.1, no_repeat_ngram_size: 6 },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Debug a message"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Run Debug Chat" }));

    expect(await screen.findByText("Fallback: yes")).toBeInTheDocument();
    expect(screen.getByText("Fallback responses run without active adapters.")).toBeInTheDocument();
    expect(screen.getByText("google/gemma-4-E2B")).toBeInTheDocument();
  });
});
