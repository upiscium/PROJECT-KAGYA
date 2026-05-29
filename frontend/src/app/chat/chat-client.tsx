"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, type ChatResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { EmotionMeter } from "@/components/emotion-meter";

type ChatTurn = { role: "user" | "assistant"; content: string; result?: ChatResponse };

export function ChatClient() {
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const mutation = useMutation({
    mutationFn: api.chat,
    onSuccess: (result, variables) => {
      setHistory((current) => [
        ...current,
        { role: "user", content: variables.message },
        { role: "assistant", content: result.response, result },
      ]);
      setMessage("");
    },
  });

  const latest = [...history].reverse().find((turn) => turn.result)?.result;

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
          </div>
        ))}
      </Card>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          if (message.trim()) mutation.mutate({ message, attachments: [], debug: false });
        }}
      >
        <Textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Send a message to PROJECT-KAGYA" />
        <Button disabled={mutation.isPending || !message.trim()} type="submit">{mutation.isPending ? "Sending" : "Send"}</Button>
      </form>

      {mutation.error ? <p className="error">{mutation.error.message}</p> : null}
    </div>
  );
}
