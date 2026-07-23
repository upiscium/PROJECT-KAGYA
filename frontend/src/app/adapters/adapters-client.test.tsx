import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AdaptersClient } from "./adapters-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><AdaptersClient /></QueryClientProvider>);
}

describe("AdaptersClient", () => {
  it("shows empty states for adapter lifecycle columns", async () => {
    fetchMock.mockResolvedValue({ ok: true, json: async () => ({ adapters: [] }) });

    renderWithQuery();

    expect(await screen.findAllByText("No adapters.")).toHaveLength(6);
  });

  it("shows a clear admin access error", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 401,
      statusText: "Unauthorized",
      text: async () => JSON.stringify({ detail: "Invalid admin token" }),
    });

    renderWithQuery();

    expect(await screen.findByText("Admin access denied: Invalid admin token")).toBeInTheDocument();
  });

  it("labels deterministic and real behavioral evidence separately", async () => {
    fetchMock.mockResolvedValue({ ok: true, json: async () => ({ adapters: [{
      adapter_id: "candidate", base_model: "base", path: "/private/candidate", status: "candidate", dataset_path: "dataset", dataset_hash: "hash", eval_score: null, eval_result_path: null, created_at: "", updated_at: "", notes: "", base_model_revision: "rev", adapter_hash: "abc", parent_adapter_id: null, parent_adapter_hash: null, activation_sequence: null, dataset_repetition_count: 0, dataset_overlap_count: 0, dataset_overlap_ratio: 0, holdout_score: null, holdout_baseline_score: null, holdout_regression: false, drift_scores: null, quality_gate_passed: true, holdout_gate_passed: true, drift_gate_passed: true, activation_gate_passed: false, behavioral_evaluation_id: "det", behavioral_evaluation_path: null, behavioral_result_hash: null, behavioral_gate_passed: true, behavioral_candidate_adapter_hash: "abc", behavioral_base_model_revision: "rev", subject_revision: "subject", fixture_set_hash: "fixture", real_model_behavioral_evaluation_id: null, real_model_behavioral_gate_passed: null, real_model_behavioral_artifact_state: "unbound", activation_eligibility_reason: "real_model_behavioral_unevaluated", real_model_behavioral_required: true, legacy_activation_warning: false, rollout_state: "candidate", canary_failures: 0, rollback_target_id: null,
    }] }) });
    renderWithQuery();
    expect(await screen.findByText("Behavioral deterministic: passed")).toBeInTheDocument();
    expect(screen.getByText("Behavioral real: not run (required)")).toBeInTheDocument();
    expect(screen.getByText("Activation eligibility: real_model_behavioral_unevaluated")).toBeInTheDocument();
    expect(screen.queryByText("/private/candidate")).not.toBeInTheDocument();
  });
});
