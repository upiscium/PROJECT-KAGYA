import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DecisionsClient } from "./decisions-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

it("renders structured safe explanation revisions and renderer status", async () => {
  fetchMock.mockResolvedValue({ ok: true, json: async () => ({ explanations: [{
    explanation_id: "explanation-1", revision: 2, decision_id: "decision-1", decision_revision: 3,
    decision_status: "resolved", disposition: "action_failed",
    selected: { candidate_id: "candidate-1", action_type: "respond", eligible: true, score: 0.4, uncertainty: 0.3, risk: 0.2, disposition_code: "selected", reason_codes: [] },
    major_alternatives: [{ candidate_id: "candidate-2", action_type: "defer", eligible: true, score: 0.1, uncertainty: 0.5, risk: 0, disposition_code: "rejected", reason_codes: [] }],
    contributions: [{ source_type: "value", source_id: "care", source_revision: 2, contribution: 0.4, evidence_refs: ["evidence-1"], origin_ref: null, availability: "available" }],
    evidence_refs: ["evidence-1"],
    uncertainty: [{ code: "selected_candidate_uncertainty", severity: 0.3, refs: ["candidate-1"] }], information_gap_codes: ["evidence_missing"], omitted_reference_count: 1,
    risk: { risk_class: "read_only", policy_status: "allowed", approval_status: "not_required", policy_ref: "policy-1", approval_ref: null, action_intent_ref: "intent-1", validation_ref: "validation-1", receipt_ref: "receipt-1", observation_ref: "observation-1", verification_ref: "verification-1", policy_reason_codes: ["tool_allowlisted"] },
    tradeoff_refs: ["tradeoff-1"], conflict_codes: ["value_conflict"], boundary: { assessment_id: "boundary-1", assessment_revision: 1, classification: "care", recommendation: "allow", disposition: "care", reason_codes: ["care_supported"] }, reason_codes: ["selected"],
    outcome: { status: "failed", utility: -0.3, prediction_error: -0.5, observed_event_ref: "event-1", post_assessment_ref: "assessment-1" }, change: { previous_explanation_revision: 1, changed_fields: ["outcome"], reason_codes: [] },
    renderer: { state: "failed", deterministic_template: "decision_explanation.action_failed.resolved.failed.v1", offered_clause_ids: ["disposition.action_failed.v1"], ordered_clause_ids: ["disposition.action_failed.v1"], visible_explanation: "Disposition: action failed.", failure_code: "renderer_failed" },
  }] }) });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><DecisionsClient /></QueryClientProvider>);

  expect(await screen.findByText(/Decision decision-1@3/)).toBeInTheDocument();
  expect(screen.getByText(/Information gaps: evidence_missing/)).toBeInTheDocument();
  expect(screen.getByText(/omitted references 1/)).toBeInTheDocument();
  expect(screen.getByText(/Tradeoffs: 1; conflicts: 1/)).toBeInTheDocument();
  expect(screen.getByText(/Boundary: care/)).toBeInTheDocument();
  expect(screen.getByText(/Renderer: renderer_failed/)).toBeInTheDocument();
  expect(screen.queryByText(/hidden thought|raw prompt|secret/i)).not.toBeInTheDocument();
});
