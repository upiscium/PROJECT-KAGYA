"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api, errorMessage, type BehavioralEvaluationSummary, type EvaluationResultSummary } from "@/lib/api";
import { evaluationAnchor, evaluationIdFromHash } from "@/lib/anchors";
import { formatNumber, statusTone } from "@/lib/format";

export function EvaluationsClient() {
  const queryClient = useQueryClient();
  const results = useQuery({ queryKey: ["evaluations"], queryFn: api.evaluations });
  const behavioral = useQuery({ queryKey: ["behavioral-evaluations"], queryFn: api.behavioralEvaluations });
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const selected = selectedFilename ?? results.data?.results[0]?.filename ?? null;
  const detail = useQuery({
    queryKey: ["evaluation", selected],
    queryFn: () => api.evaluationResult(selected ?? ""),
    enabled: selected !== null,
  });
  const [selectedBehavioralId, setSelectedBehavioralId] = useState<string | null>(null);
  const behavioralId = selectedBehavioralId ?? (behavioral.data?.results ?? [])[0]?.evaluation_id ?? null;
  useEffect(() => {
    const selectFromLocation = () => {
      const id = evaluationIdFromHash(window.location.hash);
      const selected = id !== null && behavioral.data?.results.some((result) => result.evaluation_id === id)
        ? id
        : null;
      setSelectedBehavioralId(selected);
      setFailureScenario(null);
    };
    selectFromLocation();
    window.addEventListener("hashchange", selectFromLocation);
    return () => window.removeEventListener("hashchange", selectFromLocation);
  }, [behavioral.data?.results]);
  useEffect(() => {
    const linkedId = evaluationIdFromHash(window.location.hash);
    if (linkedId !== behavioralId) return;
    document.getElementById(evaluationAnchor(linkedId))?.scrollIntoView?.({ block: "nearest" });
  }, [behavioralId]);
  const behavioralDetail = useQuery({
    queryKey: ["behavioral-evaluation", behavioralId],
    queryFn: () => api.behavioralEvaluation(behavioralId ?? ""),
    enabled: behavioralId !== null,
  });
  const [failureScenario, setFailureScenario] = useState<string | null>(null);
  const failure = useQuery({
    queryKey: ["behavioral-failure", behavioralId, failureScenario],
    queryFn: () => api.behavioralFailure(behavioralId ?? "", failureScenario ?? ""),
    enabled: behavioralId !== null && failureScenario !== null,
  });
  const rerun = useMutation({
    mutationFn: () => api.rerunBehavioralEvaluation(behavioralId ?? "", `${behavioralId}.rerun-${Date.now()}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["behavioral-evaluations"] }),
  });
  const artifactPaths = Array.isArray(behavioralDetail.data?.payload.reproduction_artifacts)
    ? behavioralDetail.data.payload.reproduction_artifacts.filter((item): item is string => typeof item === "string")
    : [];

  return (
    <div className="page">
      <header>
        <h1 className="page-title">Evaluations</h1>
        <p className="page-subtitle">Browse persisted adapter evaluation results from the admin backend.</p>
      </header>
      {results.error ? <p className="error">{errorMessage(results.error)}</p> : null}
      {detail.error ? <p className="error">{errorMessage(detail.error)}</p> : null}
      {behavioral.error ? <p className="error">{errorMessage(behavioral.error)}</p> : null}
      {behavioralDetail.error ? <p className="error">{errorMessage(behavioralDetail.error)}</p> : null}
      <section className="stack" aria-labelledby="behavioral-heading">
        <div className="page-header">
          <div>
            <h2 id="behavioral-heading">Subject behavioral gates</h2>
            <p className="muted">Structured model declarations are reconciled against authoritative runtime effects; declarations never override observed actions or state.</p>
          </div>
          <Button onClick={() => rerun.mutate()} disabled={!behavioralId || rerun.isPending}>
            {rerun.isPending ? "Rerunning..." : "Rerun fixture"}
          </Button>
        </div>
        {rerun.error ? <p className="error">{errorMessage(rerun.error)}</p> : null}
        {rerun.data ? <p className="muted">Rerun {rerun.data.evaluation_id}: hashes {rerun.data.fixture_hashes_match ? "verified" : "mismatched"}</p> : null}
        <div className="grid">
          <EvidenceSection title="Synthetic evaluator contract" kind="synthetic_evaluator_contract" results={behavioral.data?.results ?? []} selectedId={behavioralId} onSelect={(id) => { selectBehavioralEvaluation(id); setSelectedBehavioralId(id); setFailureScenario(null); }} />
          <EvidenceSection title="Deterministic runtime evaluation" kind="deterministic_runtime" results={behavioral.data?.results ?? []} selectedId={behavioralId} onSelect={(id) => { selectBehavioralEvaluation(id); setSelectedBehavioralId(id); setFailureScenario(null); }} />
          <EvidenceSection title="Real-model runtime evaluation" kind="real_model_runtime" results={behavioral.data?.results ?? []} selectedId={behavioralId} onSelect={(id) => { selectBehavioralEvaluation(id); setSelectedBehavioralId(id); setFailureScenario(null); }} />
          <Card>
            <CardTitle>Hardware evidence</CardTitle>
            <p className="muted">Pending an explicit real-model run on the configured model and adapter artifacts. Deterministic evidence does not satisfy the production hardware gate.</p>
          </Card>
          <Card>
            <CardTitle>Dimension deltas</CardTitle>
            {behavioral.data?.results.find((item) => item.evaluation_id === behavioralId) ? (
              <div className="dimension-grid">
                {Object.entries(behavioral.data.results.find((item) => item.evaluation_id === behavioralId)?.dimension_deltas ?? {}).map(([name, delta]) => (
                  <div className="dimension-cell" key={name}><span>{name.replaceAll("_", " ")}</span><strong>{formatNumber(delta)}</strong></div>
                ))}
              </div>
            ) : <p className="muted">Select a behavioral evaluation.</p>}
          </Card>
          <Card className="wide">
            <CardTitle>Failure artifacts</CardTitle>
            <div className="action-row">
              {artifactPaths.map((path) => {
                const scenario = path.split("/").at(-1)?.replace(/\.json$/, "") ?? "";
                return <Button key={path} onClick={() => setFailureScenario(scenario)} disabled={scenario === failureScenario}>{scenario}</Button>;
              })}
              {artifactPaths.length === 0 ? <p className="muted">No candidate failures in this run.</p> : null}
            </div>
            {failure.error ? <p className="error">{errorMessage(failure.error)}</p> : null}
            {failure.data ? <pre>{JSON.stringify(failure.data.payload, null, 2)}</pre> : null}
          </Card>
        </div>
      </section>
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

function EvidenceSection({ title, kind, results, selectedId, onSelect }: { title: string; kind: BehavioralEvaluationSummary["runtime_kind"]; results: BehavioralEvaluationSummary[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const matching = results.filter((result) => result.runtime_kind === kind);
  return <Card><CardTitle>{title}</CardTitle>{matching.length === 0 ? <p className="muted">Not run.</p> : matching.map((result) => <BehavioralRow key={result.evaluation_id} result={result} selected={result.evaluation_id === selectedId} onSelect={() => onSelect(result.evaluation_id)} />)}</Card>;
}

function BehavioralRow({ result, selected, onSelect }: { result: BehavioralEvaluationSummary; selected: boolean; onSelect: () => void }) {
  return (
    <article className="record" id={evaluationAnchor(result.evaluation_id)}>
      <div className="metadata-row">
        <Badge data-tone={result.activation_gate_passed ? "success" : "danger"}>{result.runtime_kind === "synthetic_evaluator_contract" ? (result.activation_gate_passed ? "contract passed" : "contract failed") : (result.activation_gate_passed ? "gate passed" : "blocked")}</Badge>
        <span className="mono">{result.evaluation_id}</span>
      </div>
      <p>{result.baseline_id} {formatNumber(result.baseline_score)} / {result.candidate_id} {formatNumber(result.candidate_score)}</p>
      <p>Artifact: {result.artifact_integrity} · Source: {result.source_integrity} · Model: {result.model_integrity}</p>
      <p>State: {result.evaluation_state}{result.failure_code ? ` (${result.failure_code})` : ""}</p>
      <p className="muted">{new Date(result.created_at).toLocaleString()}</p>
      <Button onClick={onSelect} disabled={selected}>{selected ? "Selected" : "Inspect"}</Button>
    </article>
  );
}

function selectBehavioralEvaluation(id: string): void {
  if (typeof window === "undefined") return;
  window.history.replaceState(null, "", `#${evaluationAnchor(id)}`);
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
