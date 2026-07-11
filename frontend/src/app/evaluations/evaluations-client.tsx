"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api, errorMessage, type EvaluationResultSummary } from "@/lib/api";
import { formatNumber, statusTone } from "@/lib/format";

export function EvaluationsClient() {
  const results = useQuery({ queryKey: ["evaluations"], queryFn: api.evaluations });
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const selected = selectedFilename ?? results.data?.results[0]?.filename ?? null;
  const detail = useQuery({
    queryKey: ["evaluation", selected],
    queryFn: () => api.evaluationResult(selected ?? ""),
    enabled: selected !== null,
  });

  return (
    <div className="page">
      <header>
        <h1 className="page-title">Evaluations</h1>
        <p className="page-subtitle">Browse persisted adapter evaluation results from the admin backend.</p>
      </header>
      {results.error ? <p className="error">{errorMessage(results.error)}</p> : null}
      {detail.error ? <p className="error">{errorMessage(detail.error)}</p> : null}
      <div className="grid">
        <Card>
          <CardTitle>Result Files</CardTitle>
          {results.isLoading ? <p className="muted">Loading evaluations...</p> : null}
          {results.data?.results.length === 0 ? <p className="muted">No evaluation results yet.</p> : null}
          {results.data?.results.map((result) => (
            <EvaluationRow
              key={result.filename}
              result={result}
              selected={result.filename === selected}
              onSelect={() => setSelectedFilename(result.filename)}
            />
          ))}
        </Card>
        <Card className="wide">
          <CardTitle>Result JSON</CardTitle>
          {detail.isLoading ? <p className="muted">Loading result detail...</p> : null}
          {detail.data ? <pre>{JSON.stringify(detail.data.payload, null, 2)}</pre> : null}
          {!selected && !detail.isLoading ? <p className="muted">Select an evaluation result to inspect its JSON payload.</p> : null}
        </Card>
      </div>
    </div>
  );
}

function EvaluationRow({ result, selected, onSelect }: { result: EvaluationResultSummary; selected: boolean; onSelect: () => void }) {
  return (
    <article className="record">
      <div className="metadata-row">
        <Badge data-tone={statusTone(result.decision ?? "")}>{result.decision ?? "unknown"}</Badge>
        <span className="mono">{result.filename}</span>
      </div>
      <h3>{result.adapter_id}</h3>
      <p>Score: {result.score === null ? "n/a" : formatNumber(result.score)}</p>
      <p>Delta: {result.score_delta === null ? "n/a" : formatNumber(result.score_delta)}{result.regression ? " regression" : ""}</p>
      <p>Status: {result.status_before ?? "n/a"} -&gt; {result.status_after ?? "n/a"}</p>
      <p>Cases: {result.case_count ?? "n/a"}</p>
      <p className="muted">Updated: {new Date(result.updated_at).toLocaleString()}</p>
      <Button onClick={onSelect} disabled={selected}>{selected ? "Selected" : "View JSON"}</Button>
    </article>
  );
}
