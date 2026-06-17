import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SleepClient } from "./sleep-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><SleepClient /></QueryClientProvider>);
}

describe("SleepClient", () => {
  it("shows empty sleep cycle states", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        selected_episode_ids: [],
        semantic_memory_ids: [],
        dream_dataset_path: null,
        adapter_id: null,
        adapter_status: null,
        dry_run: null,
      }),
    });
    renderWithQuery();

    await userEvent.click(screen.getByRole("button", { name: "Run Sleep Cycle" }));

    expect(await screen.findByText("No high-emotion DB1 episodes met the sleep threshold.")).toBeInTheDocument();
    expect(screen.getByText("No semantic memories were created in this cycle.")).toBeInTheDocument();
    expect(screen.getByText("No adapter created")).toBeInTheDocument();
  });
});
