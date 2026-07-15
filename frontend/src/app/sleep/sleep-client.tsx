"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage, type TrainingJob } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function SleepClient() {
  const queryClient = useQueryClient();
  const jobs = useQuery({ queryKey: ["sleep-jobs"], queryFn: api.sleepJobs, refetchInterval: 2000 });
  const mutation = useMutation({ mutationFn: () => api.createSleepJob(), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sleep-jobs"] }) });
  const result = jobs.data?.jobs?.at(-1) ?? mutation.data;

  return (
    <div className="page">
      <header className="page-header"><div><h1 className="page-title">Sleep Jobs</h1><p className="page-subtitle">Queue consolidation and training without holding the request open.</p></div><Button disabled={mutation.isPending} onClick={() => mutation.mutate()}>{mutation.isPending ? "Queueing" : "Create Sleep Job"}</Button></header>
      {mutation.error ? <p className="error">{errorMessage(mutation.error)}</p> : null}
      <div className="grid">
        <Card><CardTitle>Target Episodes</CardTitle><p className="metric">{result?.selected_episode_ids.length ?? 0}</p><p className="muted">{result && result.selected_episode_ids.length === 0 ? "No high-emotion DB1 episodes met the sleep threshold." : "High-emotion DB1 episodes selected."}</p></Card>
        <Card><CardTitle>Semantic Memories</CardTitle><p className="metric">{result?.semantic_memory_ids.length ?? 0}</p>{result && result.semantic_memory_ids.length === 0 ? <p className="muted">No semantic memories were created in this cycle.</p> : null}</Card>
        <Card><CardTitle>Training Job</CardTitle>{result ? <SleepResult result={result} /> : <p className="muted">No sleep job has run.</p>}</Card>
        <Card><CardTitle>Immutable Bundle</CardTitle><p className="mono">{result?.bundle_path ?? "Bundle not prepared"}</p><p className="muted">Bundle and result artifacts are checksum validated.</p></Card>
      </div>
    </div>
  );
}

function SleepResult({ result }: { result: TrainingJob }) {
  return <div className="stack"><p>{result.candidate_adapter_id ?? result.job_id}</p><Badge data-tone="accent">{result.status}</Badge>{result.error ? <p className="error">{result.error}</p> : null}</div>;
}
