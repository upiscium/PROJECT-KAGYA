import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvaluationsClient } from "./evaluations-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><EvaluationsClient /></QueryClientProvider>);
}

describe("EvaluationsClient", () => {
  it("lists evaluation results and loads selected JSON", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url === "/admin-proxy/evaluations/behavioral") {
        return {
          ok: true,
          json: async () => ({
            results: [{
              evaluation_id: "behavior-133",
              baseline_id: "base",
              candidate_id: "candidate",
              baseline_score: 1,
              candidate_score: 0.8,
              baseline_dimensions: { tool_safety: 1 },
              candidate_dimensions: { tool_safety: 0 },
              dimension_deltas: { tool_safety: -1 },
              activation_gate_passed: false,
              regression_dimensions: ["tool_safety"],
              threshold_failure_dimensions: ["tool_safety"],
              hard_gate_failures: ["action_policy_bypass"],
              tool_execution_dimensions_complete: true,
              created_at: "2026-07-23T00:00:00+00:00",
            }],
          }),
        };
      }
      if (url === "/admin-proxy/evaluations/behavioral/behavior-133") {
        return { ok: true, json: async () => ({ evaluation_id: "behavior-133", payload: { reproduction_artifacts: ["failures/behavior-133/tool.json"] } }) };
      }
      if (url === "/admin-proxy/evaluations/behavioral/behavior-133/failures/tool.json") {
        return { ok: true, json: async () => ({ evaluation_id: "behavior-133", scenario_id: "tool", payload: { candidate_result: { passed: false } } }) };
      }
      if (url === "/admin-proxy/evaluations/behavioral/behavior-133/rerun") {
        return { ok: true, json: async () => ({ source_evaluation_id: "behavior-133", evaluation_id: "behavior-133.rerun-1", fixture_hashes_match: true, activation_gate_passed: false }) };
      }
      if (url === "/admin-proxy/evaluations") {
        return {
          ok: true,
          json: async () => ({
            results: [
              {
                filename: "adapter-a.json",
                adapter_id: "adapter-a",
                score: 0.9,
                previous_score: 0.7,
                score_delta: 0.2,
                regression: false,
                decision: "trial_active",
                status_before: "candidate",
                status_after: "trial_active",
                case_count: 2,
                updated_at: "2026-06-08T00:00:00+00:00",
              },
              {
                filename: "adapter-b.json",
                adapter_id: "adapter-b",
                score: 0.2,
                previous_score: 0.5,
                score_delta: -0.3,
                regression: true,
                decision: "rejected",
                status_before: "candidate",
                status_after: "rejected",
                case_count: 1,
                updated_at: "2026-06-07T00:00:00+00:00",
              },
            ],
          }),
        };
      }
      const adapterId = url.endsWith("adapter-b.json") ? "adapter-b" : "adapter-a";
      const decision = adapterId === "adapter-b" ? "rejected" : "trial_active";
      return {
        ok: true,
        json: async () => ({ filename: `${adapterId}.json`, payload: { adapter_id: adapterId, decision } }),
      };
    });

    renderWithQuery();

    expect(await screen.findByText("adapter-a")).toBeInTheDocument();
    expect(screen.getByText("behavior-133")).toBeInTheDocument();
    expect(screen.getByText("tool safety")).toBeInTheDocument();
    expect(screen.getByText("adapter-b")).toBeInTheDocument();
    expect(screen.getByText(/-0\.300 regression/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "View JSON" }));

    expect(await screen.findByText(/"adapter_id": "adapter-b"/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "tool" }));
    expect(await screen.findByText(/"passed": false/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Rerun fixture" }));
    expect(await screen.findByText(/hashes verified/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/evaluations", expect.any(Object));
    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/evaluations/adapter-b.json", expect.any(Object));
  });

  it("shows an empty state when there are no evaluation results", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ results: [] }),
    });

    renderWithQuery();

    expect(await screen.findByText("No evaluation results yet.")).toBeInTheDocument();
    expect(screen.getByText("Select an evaluation result to inspect its JSON payload.")).toBeInTheDocument();
  });
});
