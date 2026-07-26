"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, errorMessage, streamChatJob, type Attachment, type ChatResponse, type FeedbackSignal, type OperationStatus } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { EmotionMeter } from "@/components/emotion-meter";

type AttachmentType = "image" | "audio" | "video";
type ChatTurn = { role: "user" | "assistant"; content: string; attachments?: Attachment[]; result?: ChatResponse };

const feedbackOptions: Array<{ value: FeedbackSignal; label: string }> = [
  { value: "good", label: "Good response" },
  { value: "bad", label: "Bad response" },
  { value: "factual_error", label: "Factual error" },
  { value: "style_problem", label: "Style problem" },
  { value: "unsafe_behavior", label: "Unsafe behavior" },
  { value: "remember", label: "Remember this" },
  { value: "do_not_remember", label: "Do not remember" },
  { value: "correction", label: "Correction" },
  { value: "expected_answer", label: "Expected answer" },
  { value: "exclude_from_training", label: "Exclude from training" },
];

function FeedbackControls({ result }: { result: ChatResponse }) {
  const [signal, setSignal] = useState<FeedbackSignal>("good");
  const [structuredText, setStructuredText] = useState("");
  const feedback = useMutation({
    mutationFn: () => api.feedback({
      idempotency_key: `${result.episode_id}:${signal}:${crypto.randomUUID()}`,
      target: {
        target_type: "response",
        target_id: result.episode_id,
        episode_id: result.episode_id,
        experience_id: result.experience_id,
        context_id: result.context_id,
      },
      signals: [signal],
      ...(signal === "correction" ? { correction: structuredText } : {}),
      ...(signal === "expected_answer" ? { expected_answer: structuredText } : {}),
    }),
    onSuccess: () => setStructuredText(""),
  });
  const requiresText = signal === "correction" || signal === "expected_answer";

  return (
    <div className="metadata-row" aria-label="Structured response feedback">
      <select aria-label="Feedback type" className="ui-input" value={signal} onChange={(event) => setSignal(event.target.value as FeedbackSignal)}>
        {feedbackOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
      {requiresText ? <Input aria-label={signal === "correction" ? "Correction" : "Expected answer"} value={structuredText} onChange={(event) => setStructuredText(event.target.value)} placeholder={signal === "correction" ? "Corrected response" : "Expected answer"} /> : null}
      <Button type="button" disabled={feedback.isPending || (requiresText && !structuredText.trim())} onClick={() => feedback.mutate()}>{feedback.isPending ? "Submitting" : "Submit feedback"}</Button>
      {feedback.isSuccess ? <Badge data-tone="accent">Feedback recorded</Badge> : null}
      {feedback.error ? <span className="error">{errorMessage(feedback.error)}</span> : null}
    </div>
  );
}

export function ChatClient() {
  const [message, setMessage] = useState("");
  const [attachmentType, setAttachmentType] = useState<AttachmentType>("image");
  const [attachmentUrl, setAttachmentUrl] = useState("");
  const [attachmentName, setAttachmentName] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [contextId, setContextId] = useState<string | undefined>();
  const [operation, setOperation] = useState<OperationStatus | null>(null);
  const [streamedText, setStreamedText] = useState("");
  const mutation = useMutation({
    mutationFn: (request: Parameters<typeof api.chat>[0]) => streamChatJob(request, {
      status: setOperation,
      token: (text) => setStreamedText((current) => current + text),
    }),
    onSuccess: (result, variables) => {
      setHistory((current) => [
        ...current,
        { role: "user", content: variables.text, attachments: variables.attachments },
        { role: "assistant", content: result.response, result },
      ]);
      setMessage("");
      setAttachments([]);
      setContextId(result.context_id);
      setOperation(null);
      setStreamedText("");
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
        {latest ? <div className="metadata-row"><Badge>{latest.model.model_id}</Badge>{latest.model.fallback_used ? <Badge data-tone="warning">Fallback model</Badge> : null}<Button type="button" onClick={() => { setContextId(undefined); setHistory([]); }}>New context</Button></div> : null}
      </header>

      {latest ? (
        <Card>
          <CardTitle>Current Model</CardTitle>
          <div className="metadata-row"><span>{latest.model.model_id}</span><span>Adapter: {latest.model.adapter_id ?? "none"}</span>{latest.model.fallback_used ? <Badge data-tone="warning">Fallback model</Badge> : <Badge data-tone="accent">Primary model</Badge>}</div>
          {latest.model.fallback_used ? <p className="muted">The primary model was unavailable for this response, so KAGYA used the configured fallback model without an adapter.</p> : null}
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
            {turn.result ? <FeedbackControls result={turn.result} /> : null}
          </div>
        ))}
        {mutation.isPending ? (
          <div className="chat-bubble assistant" aria-live="polite" aria-label="KAGYA is generating a response">
            <strong>KAGYA</strong>
            <p className="generating-indicator"><span className="spinner" aria-hidden="true" />{operation?.status === "queued" ? `Queued${operation.queue_position ? ` (${operation.queue_position})` : ""}` : operation?.status === "finalizing" ? "Finalizing response..." : "Generating response..."}</p>
            {streamedText ? <p>{streamedText}</p> : null}
            {operation ? <Button type="button" onClick={() => api.cancelChatJob(operation.operation_id)}>Cancel</Button> : null}
          </div>
        ) : null}
      </Card>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          if (message.trim()) mutation.mutate({ text: message, attachments, debug: false, context_id: contextId });
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

      {mutation.error ? <p className="error">{errorMessage(mutation.error)}</p> : null}
    </div>
  );
}
