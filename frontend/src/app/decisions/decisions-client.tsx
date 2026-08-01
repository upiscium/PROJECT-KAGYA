"use client";

import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Card, CardTitle } from "@/components/ui/card";
import { api, errorMessage } from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

export function DecisionsClient() {
  const query = useQuery({ queryKey: queryKeys.decisionExplanations, queryFn: api.decisionExplanations });

  return (
    <div className="page">
      <header>
        <h1 className="page-title">Decision Explanations</h1>
        <p className="page-subtitle">Deterministic public-safe projections. Natural-language rendering is non-authoritative.</p>
      </header>
      {query.error ? <p className="error">{errorMessage(query.error)}</p> : null}
      {!query.isPending && !query.data?.explanations.length ? <p className="muted">No explanations recorded.</p> : null}
      <div className="grid">
        {query.data?.explanations.map((explanation) => (
          <Card className="wide" key={`${explanation.explanation_id}@${explanation.revision}`}>
            <CardTitle>{explanation.disposition.replaceAll("_", " ")}</CardTitle>
            <div className="metadata-row">
              <Badge>{explanation.decision_status}</Badge>
              <Badge>{explanation.renderer.state}</Badge>
              <span>Decision {explanation.decision_id}@{explanation.decision_revision}</span>
              <span>Explanation r{explanation.revision}</span>
            </div>
            <p>{explanation.renderer.visible_explanation}</p>
            <p>Selected {explanation.selected.candidate_id} ({explanation.selected.action_type}), uncertainty {Math.round(explanation.selected.uncertainty * 100)}%, risk {Math.round(explanation.selected.risk * 100)}%</p>
            <p className="muted">Information gaps: {explanation.information_gap_codes.join(", ") || "none"}; omitted references {explanation.omitted_reference_count}</p>
            <p className="muted">Uncertainty: {explanation.uncertainty.map((item) => `${item.code} ${Math.round(item.severity * 100)}%`).join(", ") || "none"}</p>
            <p className="muted">Risk: {explanation.risk.risk_class}; policy {explanation.risk.policy_status}; approval {explanation.risk.approval_status}; reasons {explanation.risk.policy_reason_codes.join(", ") || "none"}</p>
            <p className="muted">Action refs: intent {explanation.risk.action_intent_ref ?? "none"}; receipt {explanation.risk.receipt_ref ?? "none"}; observation {explanation.risk.observation_ref ?? "none"}; verification {explanation.risk.verification_ref ?? "none"}</p>
            <p className="muted">Sources: {explanation.contributions.map((item) => `${item.source_type}:${item.source_id}@${item.source_revision} (${item.availability})`).join(", ") || "none"}</p>
            <p className="muted">Evidence: {explanation.evidence_refs.join(", ") || "none"}</p>
            <p className="muted">Alternatives: {explanation.major_alternatives.map((item) => `${item.candidate_id}:${item.disposition_code}`).join(", ") || "none"}</p>
            <p className="muted">Tradeoffs: {explanation.tradeoff_refs.length}; conflicts: {explanation.conflict_codes.length}</p>
            <p className="muted">Boundary: {explanation.boundary ? `${explanation.boundary.disposition} (${explanation.boundary.reason_codes.join(", ") || "no reasons"})` : "none"}</p>
            <p className="muted">Outcome: {explanation.outcome.status}; event {explanation.outcome.observed_event_ref ?? "none"}; assessment {explanation.outcome.post_assessment_ref ?? "none"}</p>
            <p className="muted">Reasons: {explanation.reason_codes.join(", ") || "none"}; changed: {explanation.change.changed_fields.join(", ") || "none"}; previous revision {explanation.change.previous_explanation_revision ?? "none"}</p>
            <p className="muted">Renderer clauses: {explanation.renderer.ordered_clause_ids.join(" -> ")}; template {explanation.renderer.deterministic_template}</p>
            {explanation.renderer.failure_code ? <p className="error">Renderer: {explanation.renderer.failure_code}</p> : null}
          </Card>
        ))}
      </div>
    </div>
  );
}
