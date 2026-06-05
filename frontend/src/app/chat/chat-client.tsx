"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, type Attachment, type ChatResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { EmotionMeter } from "@/components/emotion-meter";

type AttachmentType = "image" | "audio" | "video";
type ChatTurn = { role: "user" | "assistant"; content: string; attachments?: Attachment[]; result?: ChatResponse };

export function ChatClient() {
  const [message, setMessage] = useState("");
  const [attachmentType, setAttachmentType] = useState<AttachmentType>("image");
  const [attachmentUrl, setAttachmentUrl] = useState("");
  const [attachmentName, setAttachmentName] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const mutation = useMutation({
    mutationFn: api.chat,
    onSuccess: (result, variables) => {
      setHistory((current) => [
        ...current,
        { role: "user", content: variables.text, attachments: variables.attachments },
        { role: "assistant", content: result.response, result },
      ]);
      setMessage("");
      setAttachments([]);
    },
  });

  const latest = [...history].reverse().find((turn) => turn.result)?.result;
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
      <header className="page-header">
        <div>
          <h1 className="page-title">Chat</h1>
          <p className="page-subtitle">Normal chat shows only user-facing answers and emotion state.</p>
        </div>
        {latest ? <Badge>{latest.model.model_id}</Badge> : null}
      </header>

      {latest ? (
        <Card>
          <CardTitle>Current Model</CardTitle>
          <div className="metadata-row"><span>{latest.model.model_id}</span><span>Adapter: {latest.model.adapter_id ?? "none"}</span></div>
          <EmotionMeter emotion={latest.emotion} />
        </Card>
      ) : null}

      <Card className="chat-history" aria-label="Chat history">
        {history.length === 0 ? <p className="muted">No messages yet.</p> : null}
        {history.map((turn, index) => (
          <div key={`${turn.role}-${index}`} className={`chat-bubble ${turn.role}`}>
            <strong>{turn.role === "user" ? "You" : "KAGYA"}</strong>
            <p>{turn.content}</p>
            {turn.attachments?.length ? <p className="muted">Attachments: {turn.attachments.map((attachment) => attachment.name ?? attachment.url ?? attachment.type).join(", ")}</p> : null}
          </div>
        ))}
      </Card>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          if (message.trim()) mutation.mutate({ text: message, attachments, debug: false });
        }}
      >
        <Textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Send a message to PROJECT-KAGYA" />
        <div className="attachment-panel" aria-label="Attachments">
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
        <Button disabled={mutation.isPending || !message.trim()} type="submit">{mutation.isPending ? "Sending" : "Send"}</Button>
      </form>

      {mutation.error ? <p className="error">{mutation.error.message}</p> : null}
    </div>
  );
}
