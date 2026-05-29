"use client";

import { useMutation } from "@tanstack/react-query";
import { api, type SleepRunResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function SleepClient() {
  const mutation = useMutation({ mutationFn: api.sleepRun });
  const result = mutation.data;

  return (
    <div className="page">
      <header className="page-header"><div><h1 className="page-title">Sleep</h1><p className="page-subtitle">Run high-emotion consolidation, dream dataset generation, and QLoRA dry-run.</p></div><Button disabled={mutation.isPending} onClick={() => mutation.mutate()}>{mutation.isPending ? "Sleeping" : "Run Sleep Cycle"}</Button></header>
      {mutation.error ? <p className="error">{mutation.error.message}</p> : null}
      <div className="grid">
        <Card><CardTitle>Target Episodes</CardTitle><p className="metric">{result?.selected_episode_ids.length ?? 0}</p><p className="muted">High-emotion DB1 episodes selected.</p></Card>
        <Card><CardTitle>Semantic Memories</CardTitle><p className="metric">{result?.semantic_memory_ids.length ?? 0}</p></Card>
        <Card><CardTitle>Adapter Candidate</CardTitle>{result ? <SleepResult result={result} /> : <p className="muted">No sleep cycle has run.</p>}</Card>
        <Card><CardTitle>Dream Dataset Preview</CardTitle><p className="mono">{result?.dream_dataset_path ?? "No dataset yet"}</p><p className="muted">Records are JSONL with input, thought, and output fields.</p></Card>
      </div>
    </div>
  );
}

function SleepResult({ result }: { result: SleepRunResponse }) {
  return <div className="stack"><p>{result.adapter_id ?? "No adapter created"}</p>{result.adapter_status ? <Badge data-tone="accent">{result.adapter_status}</Badge> : null}<p>Dry run: {String(result.dry_run)}</p></div>;
}
