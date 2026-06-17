"use client";

import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useState } from "react";
import { api, errorMessage } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export function MemoryClient() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const search = useQuery({
    queryKey: ["memory", submitted],
    queryFn: () => api.memorySearch(submitted),
    enabled: submitted.length > 0,
  });

  return (
    <div className="page">
      <header><h1 className="page-title">Memory</h1><p className="page-subtitle">Search DB1 episodes and DB2 semantic memories.</p></header>
      <form className="composer" onSubmit={(event) => { event.preventDefault(); setSubmitted(query.trim()); }}>
        <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search memory" />
        <Button disabled={!query.trim()} type="submit">Search</Button>
      </form>
      {search.error ? <p className="error">{errorMessage(search.error)}</p> : null}
      <div className="grid">
        <Card><CardTitle>DB1 Episodes</CardTitle>{memoryResults(search.data?.db1_results, submitted, "No DB1 episodes matched this query.", (item) => <article key={item.id} className="record"><Badge>{item.record_type}</Badge><h3>{item.user_input}</h3><p>{item.response}</p></article>)}</Card>
        <Card><CardTitle>DB2 Semantic</CardTitle>{memoryResults(search.data?.db2_results, submitted, "No DB2 semantic memories matched this query.", (item) => <article key={item.id} className="record"><Badge>{item.record_type}</Badge><p>{item.text}</p></article>)}</Card>
      </div>
    </div>
  );
}

function memoryResults<T>(items: T[] | undefined, submitted: string, emptyMessage: string, render: (item: T) => ReactNode) {
  if (!submitted) return <p className="muted">Enter a query to search memory.</p>;
  if (!items) return <p className="muted">Searching memory...</p>;
  if (items.length === 0) return <p className="muted">{emptyMessage}</p>;
  return items.map(render);
}
