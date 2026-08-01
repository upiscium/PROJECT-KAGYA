"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { api, errorMessage, type OutboxMessage } from "@/lib/api";
import Link from "next/link";
import { queryKeys } from "@/lib/query-keys";

export function OutboxClient() {
  const queryClient = useQueryClient();
  const [replies, setReplies] = useState<Record<string, string>>({});
  const messages = useQuery({
    queryKey: queryKeys.outbox,
    queryFn: api.outboxMessages,
    refetchInterval: 5000,
  });
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.outbox, refetchType: "all" });
    void queryClient.invalidateQueries({ queryKey: queryKeys.cockpit.outbox, refetchType: "all" });
  };
  const deliver = useMutation({ mutationFn: api.deliverOutbox, onSettled: refresh });
  const respond = useMutation({
    mutationFn: ({ message, kind }: { message: OutboxMessage; kind: "read" | "reply" }) =>
      api.respondToOutbox(message.message_id, kind, kind === "reply" ? replies[message.message_id] : undefined),
    onSettled: refresh,
  });
  const error = messages.error ?? deliver.error ?? respond.error;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Proactive Outbox</h1>
          <p className="page-subtitle">Local, governed messages with authoritative delivery and response correlation.</p>
        </div>
        <Button onClick={() => deliver.mutate()} disabled={deliver.isPending}>Deliver ready</Button>
      </header>
      {error ? <p className="error">{errorMessage(error)}</p> : null}
      <Card>
        <CardTitle>Message Journal</CardTitle>
        {messages.isLoading ? <p className="muted">Loading outbox...</p> : null}
        {messages.data?.messages.length === 0 ? <p className="muted">No proactive messages.</p> : null}
        <div className="stack">
          {messages.data?.messages.map((message) => (
            <article className="record" key={message.message_id}>
              <div className="metadata-row">
                <Badge data-tone={message.urgency === "critical" || message.urgency === "high" ? "danger" : "accent"}>{message.urgency}</Badge>
                <Badge data-tone={message.delivery_status === "failed" || message.delivery_status === "expired" || message.delivery_status === "cancelled" ? "danger" : message.delivery_status === "delivered" ? "success" : "neutral"}>{message.delivery_status}</Badge>
                <Badge data-tone={message.acknowledgment_status === "unacknowledged" ? "neutral" : "success"}>{message.acknowledgment_status}</Badge>
                <span>{message.kind.replaceAll("_", " ")}</span>
              </div>
              <h3>{message.title}</h3>
              {message.body_preview ? <p>{message.body_preview}</p> : null}
               <p className="muted">Created {new Date(message.created_at).toLocaleString()} · {message.channel} · {message.privacy_class}</p>
              {message.last_failure_code ? <p className="error">Last delivery failure: {message.last_failure_code}</p> : null}
              <p className="mono muted">{referenceSummary(message)}</p>
              {message.kind === "approval_request" ? <p><Link className="entity-link" href={`/cockpit#action-${encodeURIComponent(message.references.action_id ?? message.message_id)}`}>Open in Cockpit</Link></p> : null}
              {message.kind !== "approval_request" && message.delivery_status === "delivered" && message.acknowledgment_status === "unacknowledged" ? (
                <div className="composer">
                  {message.kind === "question" || message.kind === "renegotiation" ? (
                    <Textarea aria-label={`Reply to ${message.title}`} value={replies[message.message_id] ?? ""} onChange={(event) => setReplies((current) => ({ ...current, [message.message_id]: event.target.value }))} placeholder="Reply with information that should be correlated to the originating state" />
                  ) : null}
                  <div className="action-row">
                    <Button onClick={() => respond.mutate({ message, kind: "read" })}>Mark read</Button>
                    {message.kind === "question" || message.kind === "renegotiation" ? <Button disabled={!replies[message.message_id]?.trim()} onClick={() => respond.mutate({ message, kind: "reply" })}>Reply</Button> : null}
                  </div>
                </div>
              ) : null}
            </article>
          ))}
        </div>
      </Card>
    </div>
  );
}

function referenceSummary(message: OutboxMessage): string {
  const references = Object.entries(message.references).filter(([, value]) => value !== null);
  return references.length ? references.map(([kind, value]) => `${kind}: ${value}`).join(" · ") : `message: ${message.message_id}`;
}
