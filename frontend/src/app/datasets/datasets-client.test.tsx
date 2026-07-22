import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetsClient } from "./datasets-client";

const fetchMock = vi.fn();
const revision = "a".repeat(64);

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><DatasetsClient /></QueryClientProvider>);
}

describe("DatasetsClient", () => {
  it("browses governed records and lineage", async () => {
    fetchMock.mockImplementation(async (url: string) => ({
      ok: true,
      json: async () => url.endsWith("/training/datasets") ? { datasets: [{ revision, parent_revision: null, created_at: "2026-01-01T00:00:00Z", source_job_id: "job-1", record_count: 1, disposition_counts: { included: 1 }, split_counts: { train: 1 }, quality_findings: [], record_ids: ["record-1"], manifest_hash: "b".repeat(64) }] } : { manifest: { revision, parent_revision: null, created_at: "2026-01-01T00:00:00Z", source_job_id: "job-1", record_count: 1, disposition_counts: { included: 1 }, split_counts: { train: 1 }, quality_findings: [], record_ids: ["record-1"], manifest_hash: "b".repeat(64) }, records: [{ record_id: "record-1", schema_version: 1, input: "input", thought: "", output: "output", provenance: { source_kind: "verified_episode", source_id: "episode-1", source_event_ids: ["event-1"], source_memory_ids: ["episode-1"], source_decision_ids: ["decision-1"], source_feedback_ids: ["feedback-1"] }, inclusion_reason: "verified source", consent: "allowed", privacy: "internal", disposition: "included", split: "train", content_hash: "c".repeat(64), quarantine_reasons: [], exclusion_reasons: [], quality_checks: [], context_id: null, interlocutor_id: null }] },
    }));

    renderWithQuery();

    expect(await screen.findByText("verified_episode: episode-1")).toBeInTheDocument();
    expect(screen.getByText(/decisions decision-1/)).toBeInTheDocument();
    expect(screen.getByText("included")).toBeInTheDocument();
  });
});
