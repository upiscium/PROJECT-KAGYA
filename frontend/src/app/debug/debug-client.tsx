"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, type DebugChatResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/format";

export function DebugClient() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<DebugChatResponse | null>(null);
  const mutation = useMutation({ mutationFn: api.debugChat, onSuccess: setResult });

  return (
    <div className="page">
      <header>
        <h1 className="page-title">Debug</h1>
        <p className="page-subtitle">Development-only view for hidden thought, prompt, loss, memory, and generation params.</p>
      </header>
      <form className="composer" onSubmit={(event) => { event.preventDefault(); if (message.trim()) mutation.mutate({ message, attachments: [], debug: true }); }}>
        <Textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Debug a message" />
        <Button disabled={mutation.isPending || !message.trim()} type="submit">Run Debug Chat</Button>
      </form>
      {mutation.error ? <p className="error">{mutation.error.message}</p> : null}
      {result ? <DebugResult result={result} /> : null}
    </div>
  );
}

function DebugResult({ result }: { result: DebugChatResponse }) {
  return (
    <div className="grid">
      <Card><CardTitle>Visible Response</CardTitle><p>{result.response}</p></Card>
      <Card><CardTitle>Hidden Thought</CardTitle><pre>{result.hidden_thought || "none"}</pre></Card>
      <Card><CardTitle>Loss + Emotion</CardTitle><p>Loss {formatNumber(result.loss)}</p><p>Valence {formatNumber(result.emotion.valence)}</p><p>Arousal {formatNumber(result.emotion.arousal)}</p><p>Optimal loss {formatNumber(result.emotion.optimal_loss)}</p></Card>
      <Card><CardTitle>Model</CardTitle><p>{result.model.model_id}</p><p>Adapter {result.model.adapter_id ?? "none"}</p></Card>
      <Card className="wide"><CardTitle>Raw Prompt</CardTitle><pre>{result.prompt}</pre></Card>
      <Card><CardTitle>Generation Params</CardTitle><pre>{JSON.stringify(result.generation_params, null, 2)}</pre></Card>
      <Card><CardTitle>DB1 Episodes</CardTitle>{result.retrieved_memory.db1_results.map((item) => <p key={item.id}><Badge>{item.record_type}</Badge> {item.user_input}</p>)}</Card>
      <Card><CardTitle>DB2 Semantic</CardTitle>{result.retrieved_memory.db2_results.map((item) => <p key={item.id}><Badge>{item.record_type}</Badge> {item.text}</p>)}</Card>
    </div>
  );
}
