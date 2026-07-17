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
    const job = {
      job_id: "job-1", attempt_id: "attempt-1", idempotency_key: "request-1", status: "completed",
      bundle_path: null, bundle_hash: null, base_model_id: "model", base_model_revision: "revision",
      parent_adapter_id: null, source_event_sequence_start: 0, source_event_sequence_end: 0,
      backend: "local", remote_job_id: null, candidate_adapter_id: null,
      selected_episode_ids: [], semantic_memory_ids: [], created_at: "now", updated_at: "now", error: null, retry_count: 0,
      phase_started_at: "now", phase_durations_seconds: {}, transferred_bytes: 0,
      remote_last_contact: null, worker_node_id: null, worker_hostname: null,
      failure_category: null, retryable: null, import_status: "completed", correlation_id: "request-1",
      processor_revision: "processor-revision", training_metrics: {},
      total_duration_seconds: 0,
      stale: false,
    };
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ jobs: [] }),
    }).mockResolvedValueOnce({
      ok: true,
      json: async () => job,
    }).mockResolvedValue({
      ok: true,
      json: async () => ({ jobs: [job] }),
    });
    renderWithQuery();

    await userEvent.click(screen.getByRole("button", { name: "Create Sleep Job" }));

    expect(await screen.findByText("No high-emotion DB1 episodes met the sleep threshold.")).toBeInTheDocument();
    expect(screen.getByText("No semantic memories were created in this cycle.")).toBeInTheDocument();
    expect(await screen.findByText("job-1")).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
  });
});
