import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryClient } from "./memory-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><MemoryClient /></QueryClientProvider>);
}

describe("MemoryClient", () => {
  it("distinguishes no query from empty search results", async () => {
    fetchMock.mockResolvedValue({ ok: true, json: async () => ({ db1_results: [], db2_results: [] }) });
    renderWithQuery();

    expect(screen.getAllByText("Enter a query to search memory.")).toHaveLength(2);
    await userEvent.type(screen.getByPlaceholderText("Search memory"), "missing");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("No DB1 episodes matched this query.")).toBeInTheDocument();
    expect(screen.getByText("No DB2 semantic memories matched this query.")).toBeInTheDocument();
  });

  it("posts tag and archive controls for episode results", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        db1_results: [{ id: "episode-1", user_input: "hello", response: "world", record_type: "episodic_log", archived: false, tags: [], operator_metadata: {}, loss: 0, emotion_valence: 0, emotion_arousal: 0, created_at: "" }],
        db2_results: [],
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Search memory"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("hello");
    await userEvent.type(screen.getByPlaceholderText("Add tag"), "review");
    await userEvent.click(screen.getByRole("button", { name: "Add tag" }));
    await userEvent.click(screen.getByRole("button", { name: "Archive" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/admin-proxy/memory/episodes/episode-1/metadata",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ tags: ["review"] }) }),
    ));
    expect(fetchMock).toHaveBeenCalledWith(
      "/admin-proxy/memory/episodes/episode-1/archive",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
