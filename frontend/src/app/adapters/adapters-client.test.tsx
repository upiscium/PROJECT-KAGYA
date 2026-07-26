import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Adapter, BehavioralArtifactStatus } from "@/lib/api";
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

function adapter(overrides: Partial<Adapter> = {}): Adapter {
  return {
    adapter_id: "candidate", base_model: "base", path: "/private/candidate", status: "candidate", dataset_path: "dataset", dataset_hash: "hash", eval_score: null, eval_result_path: null, created_at: "", updated_at: "", notes: "", base_model_revision: "rev", adapter_hash: "raw-adapter-hash", parent_adapter_id: null, parent_adapter_hash: null, activation_sequence: null, dataset_repetition_count: 0, dataset_overlap_count: 0, dataset_overlap_ratio: 0, holdout_score: null, holdout_baseline_score: null, holdout_regression: false, drift_scores: null, quality_gate_passed: true, holdout_gate_passed: true, drift_gate_passed: true, activation_gate_passed: false, behavioral_evaluation_id: null, behavioral_evaluation_path: null, behavioral_result_hash: null, behavioral_gate_passed: null, behavioral_candidate_adapter_hash: null, behavioral_base_model_revision: null, subject_revision: null, fixture_set_hash: null, behavioral_artifact_state: "unbound", deterministic_coverage_status: "not_evaluated", deterministic_behavioral_artifact_status: "not_run", real_model_behavioral_evaluation_id: null, real_model_behavioral_gate_passed: null, real_model_behavioral_artifact_state: "unbound", real_model_coverage_status: "not_evaluated", real_model_behavioral_artifact_status: "not_run", behavioral_artifact_hash_match: "not_run", activation_eligibility_reason: "behavioral_unevaluated", real_model_behavioral_required: false, behavioral_activation_policy: "deterministic_runtime_only", legacy_activation_warning: false, rollout_state: "candidate", canary_failures: 0, rollback_target_id: null, identity_integrity_status: "not_evaluated", real_model_identity_integrity_status: "not_evaluated", rollback_reason: null,
    ...overrides,
  };
}

function mockAdapter(value: Adapter) {
  fetchMock.mockResolvedValue({ ok: true, json: async () => ({ adapters: [value] }) });
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

  it.each([
    ["not_run", true, "not run", "not run", "not run"],
    ["valid", false, "failed", "valid", "passed"],
    ["valid", true, "passed", "valid", "passed"],
    ["hash_mismatch", true, "stale / hash mismatch", "hash mismatch", "failed"],
    ["corrupt", true, "corrupt", "corrupt", "failed"],
    ["orphan", true, "missing", "orphan", "failed"],
  ] as const)("renders bounded %s artifact status", async (status, gate, gateText, artifactText, hashText) => {
    mockAdapter(
      adapter({
        behavioral_evaluation_id: status === "not_run" ? null : "deterministic",
        behavioral_gate_passed: gate,
        deterministic_behavioral_artifact_status: status as BehavioralArtifactStatus,
        behavioral_artifact_hash_match: hashText === "not run" ? "not_run" : hashText,
        activation_eligibility_reason: status === "hash_mismatch" ? "adapter_artifact_mismatch" : status === "corrupt" ? "behavioral_result_corrupt" : "behavioral_failed",
      }),
    );
    renderWithQuery();
    expect(await screen.findByText(`Runtime architecture gate: ${gateText}`)).toBeInTheDocument();
    expect(screen.getByText(`Architecture artifact: ${artifactText}`)).toBeInTheDocument();
    expect(screen.getByText(`Artifact hash match: ${hashText}`)).toBeInTheDocument();
    expect(screen.queryByText("raw-adapter-hash")).not.toBeInTheDocument();
  });

  it("shows real-model status separately and preserves exact eligibility reason", async () => {
    mockAdapter(
      adapter({
        deterministic_coverage_status: "complete",
        real_model_coverage_status: "incomplete",
        real_model_behavioral_required: true,
        activation_eligibility_reason: "real_model_not_run",
      }),
    );
    renderWithQuery();
    expect(await screen.findByText("Candidate runtime artifact: not run")).toBeInTheDocument();
    expect(screen.getByText("Candidate adapter runtime gate: not run (required)")).toBeInTheDocument();
    expect(screen.getByText("Activation eligibility: real_model_not_run")).toBeInTheDocument();
    expect(screen.getByText("Runtime architecture coverage: complete")).toBeInTheDocument();
    expect(screen.getByText("Candidate adapter coverage: incomplete")).toBeInTheDocument();
    expect(screen.queryByText("/private/candidate")).not.toBeInTheDocument();
  });

  it("never renders an unbound synthetic claim as behavioral passed", async () => {
    mockAdapter(
      adapter({
        behavioral_evaluation_id: "synthetic-contract",
        behavioral_gate_passed: true,
      }),
    );
    renderWithQuery();
    expect(await screen.findByText("Runtime architecture gate: not run")).toBeInTheDocument();
    expect(screen.queryByText("Runtime architecture gate: passed")).not.toBeInTheDocument();
  });
});
