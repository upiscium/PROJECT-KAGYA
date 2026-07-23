"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage, type Adapter, type AdapterEvaluateResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { statusTone } from "@/lib/format";

const actions = ["evaluate", "trial", "approve", "activate", "reject"] as const;
type Action = (typeof actions)[number];

export function AdaptersClient() {
  const queryClient = useQueryClient();
  const adapters = useQuery({ queryKey: ["adapters"], queryFn: api.adapters });
  const action = useMutation<Adapter | AdapterEvaluateResponse, Error, { adapter: Adapter; action: Action }>({
    mutationFn: ({ adapter, action }: { adapter: Adapter; action: Action }) => runAction(adapter.adapter_id, action),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["adapters"] }),
  });

  const grouped = groupByStatus(adapters.data?.adapters ?? []);
  return (
    <div className="page">
      <header><h1 className="page-title">Adapters</h1><p className="page-subtitle">Lifecycle controls for candidate, trial active, approved, active, rejected, and archived adapters.</p></header>
      {adapters.error ? <p className="error">{errorMessage(adapters.error)}</p> : null}
      {action.error ? <p className="error">{errorMessage(action.error)}</p> : null}
      <div className="grid">
        {Object.entries(grouped).map(([status, items]) => (
          <Card key={status}><CardTitle><Badge data-tone={statusTone(status)}>{status}</Badge></CardTitle>{items.length === 0 ? <p className="muted">No adapters.</p> : items.map((adapter) => <AdapterRow key={adapter.adapter_id} adapter={adapter} onAction={(next) => action.mutate({ adapter, action: next })} busy={action.isPending} />)}</Card>
        ))}
      </div>
    </div>
  );
}

function AdapterRow({ adapter, busy, onAction }: { adapter: Adapter; busy: boolean; onAction: (action: Action) => void }) {
  return <article className="record"><h3>{adapter.adapter_id}</h3><p>Parent: {adapter.parent_adapter_id ?? "base model"}</p><p>Quality: {formatGate(adapter.quality_gate_passed)}</p><p>Holdout: {formatGate(adapter.holdout_gate_passed)}</p><p>Drift: {formatGate(adapter.drift_gate_passed)}</p><p>Behavioral deterministic: {formatGate(adapter.behavioral_gate_passed)}</p><p>Behavioral real: {formatGate(adapter.real_model_behavioral_gate_passed)}{adapter.real_model_behavioral_required ? " (required)" : " (optional)"}</p><p>Artifact hash: {artifactStatus(adapter)}</p><p>Activation eligibility: {adapter.activation_eligibility_reason || "not evaluated"}</p>{adapter.legacy_activation_warning ? <p className="error">Legacy active adapter has no behavioral evaluation; reactivation and rollback promotion are blocked.</p> : null}<p>Dataset overlap: {adapter.dataset_overlap_count} ({Math.round(adapter.dataset_overlap_ratio * 100)}%) · repeats {adapter.dataset_repetition_count}</p><p>Drift scores: {formatDrift(adapter.drift_scores)}</p><p>Score: {adapter.eval_score ?? "n/a"}</p><a href={`/admin-proxy/adapters/${encodeURIComponent(adapter.adapter_id)}/provenance`} target="_blank" rel="noreferrer">Export provenance</a><div className="action-row">{actions.map((item) => <Button key={item} disabled={busy} onClick={() => onAction(item)}>{item}</Button>)}</div></article>;
}

function formatGate(value: boolean | null): string {
  return value === true ? "passed" : value === false ? "failed" : "not run";
}

function artifactStatus(adapter: Adapter): string {
  if (adapter.real_model_behavioral_artifact_state === "quarantined") return "corrupt";
  if (adapter.activation_eligibility_reason.includes("artifact_mismatch")) return "hash-mismatch";
  if (adapter.activation_eligibility_reason.includes("corrupt")) return "corrupt";
  return adapter.adapter_hash ? adapter.adapter_hash : "not run";
}

function formatDrift(scores: Record<string, number> | null): string {
  if (!scores || Object.keys(scores).length === 0) return "n/a";
  return Object.entries(scores).map(([name, score]) => `${name} ${score.toFixed(3)}`).join(", ");
}

function groupByStatus(adapters: Adapter[]): Record<string, Adapter[]> {
  const statuses = ["candidate", "trial_active", "approved", "active", "rejected", "archived"];
  return Object.fromEntries(statuses.map((status) => [status, adapters.filter((adapter) => adapter.status === status)]));
}

function runAction(adapterId: string, action: Action) {
  if (action === "evaluate") return api.evaluateAdapter(adapterId);
  if (action === "trial") return api.trialAdapter(adapterId);
  if (action === "approve") return api.approveAdapter(adapterId);
  if (action === "activate") return api.activateAdapter(adapterId);
  return api.rejectAdapter(adapterId);
}
