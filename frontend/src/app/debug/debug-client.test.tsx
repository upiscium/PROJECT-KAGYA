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
        response: "Visible answer",
        hidden_thought: "internal thought",
        loss: 1.25,
        prompt: "raw prompt",
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null },
        retrieved_memory: { db1_results: [], db2_results: [] },
        generation_params: { max_new_tokens: 8, temperature: 0.7, top_p: 0.95, do_sample: true },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Debug a message"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Run Debug Chat" }));

    expect(await screen.findByText("internal thought")).toBeInTheDocument();
    expect(screen.getByText("raw prompt")).toBeInTheDocument();
    expect(screen.getByText(/Loss 1.250/)).toBeInTheDocument();
  });
});
