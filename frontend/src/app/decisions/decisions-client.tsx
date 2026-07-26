"use client";

import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Card, CardTitle } from "@/components/ui/card";
import { api, errorMessage } from "@/lib/api";

export function DecisionsClient() {
  const query = useQuery({ queryKey: ["decision-explanations"], queryFn: api.decisionExplanations });

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
            <p className="muted">Information gaps: {explanation.information_gap_codes.join(", ") || "none"}</p>
            <p className="muted">Risk: {explanation.risk.risk_class}; policy {explanation.risk.policy_status}; approval {explanation.risk.approval_status}</p>
            <p className="muted">Sources: {explanation.contributions.map((item) => `${item.source_type}:${item.source_id}@${item.source_revision} (${item.availability})`).join(", ") || "none"}</p>
            <p className="muted">Evidence: {explanation.evidence_refs.join(", ") || "none"}</p>
            <p className="muted">Alternatives: {explanation.major_alternatives.map((item) => `${item.candidate_id}:${item.disposition_code}`).join(", ") || "none"}</p>
            <p className="muted">Outcome: {explanation.outcome.status}; changed: {explanation.change.changed_fields.join(", ") || "none"}; previous revision {explanation.change.previous_explanation_revision ?? "none"}</p>
            {explanation.renderer.failure_code ? <p className="error">Renderer: {explanation.renderer.failure_code}</p> : null}
          </Card>
        ))}
      </div>
    </div>
  );
}
