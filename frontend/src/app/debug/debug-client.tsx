"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, errorMessage, type Attachment, type DebugChatResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/format";

type AttachmentType = "image" | "audio" | "video";

export function DebugClient() {
  const [message, setMessage] = useState("");
  const [attachmentType, setAttachmentType] = useState<AttachmentType>("image");
  const [attachmentUrl, setAttachmentUrl] = useState("");
  const [attachmentName, setAttachmentName] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [result, setResult] = useState<DebugChatResponse | null>(null);
  const mutation = useMutation({ mutationFn: api.debugChat, onSuccess: setResult });
  const canAddAttachment = attachmentUrl.trim().length > 0;

  function addAttachment() {
    if (!canAddAttachment) return;
    const url = attachmentUrl.trim();
    setAttachments((current) => [
      ...current,
      {
        type: attachmentType,
        url,
        name: attachmentName.trim() || url.split("/").pop() || undefined,
      },
    ]);
    setAttachmentUrl("");
    setAttachmentName("");
  }

  return (
    <div className="page">
      <header>
        <h1 className="page-title">Debug</h1>
        <p className="page-subtitle">Development-only view for hidden thought, prompt, loss, memory, and generation params.</p>
      </header>
      <form className="composer" onSubmit={(event) => { event.preventDefault(); if (message.trim()) mutation.mutate({ text: message, attachments, debug: true }); }}>
        <Textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Debug a message" />
        <div className="attachment-panel" aria-label="Debug attachments">
          <select className="ui-input" aria-label="Attachment type" value={attachmentType} onChange={(event) => setAttachmentType(event.target.value as AttachmentType)}>
            <option value="image">Image</option>
            <option value="audio">Audio</option>
            <option value="video">Video</option>
          </select>
          <Input value={attachmentUrl} onChange={(event) => setAttachmentUrl(event.target.value)} placeholder="Attachment URL" />
          <Input value={attachmentName} onChange={(event) => setAttachmentName(event.target.value)} placeholder="Optional attachment name" />
          <Button type="button" disabled={!canAddAttachment} onClick={addAttachment}>Add Attachment</Button>
        </div>
        {attachments.length ? (
          <div className="attachment-list" aria-label="Selected attachments">
            {attachments.map((attachment, index) => (
              <span key={`${attachment.type}-${attachment.url}-${index}`} className="attachment-chip">
                {attachment.type}: {attachment.name ?? attachment.url}
                <button type="button" onClick={() => setAttachments((current) => current.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
              </span>
            ))}
          </div>
        ) : null}
        <Button disabled={mutation.isPending || !message.trim()} type="submit">Run Debug Chat</Button>
      </form>
      {mutation.error ? <p className="error">{errorMessage(mutation.error)}</p> : null}
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
      <Card><CardTitle>Model</CardTitle><p>{result.model.model_id}</p><p>Adapter {result.model.adapter_id ?? "none"}</p><p>Fallback: {result.model.fallback_used ? "yes" : "no"}</p>{result.model.fallback_used ? <p className="muted">Fallback responses run without active adapters.</p> : null}</Card>
      <Card><CardTitle>Received Attachments</CardTitle>{result.attachments.length ? result.attachments.map((attachment, index) => <p key={`${attachment.type}-${attachment.url ?? index}`}><Badge>{attachment.type}</Badge> {attachment.name ?? attachment.url ?? "unnamed"}{attachment.content_type ? ` (${attachment.content_type})` : ""}</p>) : <p>none</p>}</Card>
      <Card className="wide"><CardTitle>Raw Prompt</CardTitle><pre>{result.prompt}</pre></Card>
      <Card><CardTitle>Generation Params</CardTitle><pre>{JSON.stringify(result.generation_params, null, 2)}</pre></Card>
      <Card><CardTitle>DB1 Episodes</CardTitle>{result.retrieved_memory.db1_results.length ? result.retrieved_memory.db1_results.map((item) => <p key={item.id}><Badge>{item.record_type}</Badge> {item.user_input}</p>) : <p className="muted">No DB1 episodes retrieved.</p>}</Card>
      <Card><CardTitle>DB2 Semantic</CardTitle>{result.retrieved_memory.db2_results.length ? result.retrieved_memory.db2_results.map((item) => <p key={item.id}><Badge>{item.record_type}</Badge> {item.text}</p>) : <p className="muted">No DB2 semantic memories retrieved.</p>}</Card>
    </div>
  );
}
