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
  return <article className="record"><h3>{adapter.adapter_id}</h3><p className="mono">{adapter.path}</p><p>Score: {adapter.eval_score ?? "n/a"}</p><div className="action-row">{actions.map((item) => <Button key={item} disabled={busy} onClick={() => onAction(item)}>{item}</Button>)}</div></article>;
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
