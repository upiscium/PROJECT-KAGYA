"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/lib/api";
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
      {search.error ? <p className="error">{search.error.message}</p> : null}
      <div className="grid">
        <Card><CardTitle>DB1 Episodes</CardTitle>{search.data?.db1_results.map((item) => <article key={item.id} className="record"><Badge>{item.record_type}</Badge><h3>{item.user_input}</h3><p>{item.response}</p></article>) ?? <p className="muted">No query yet.</p>}</Card>
        <Card><CardTitle>DB2 Semantic</CardTitle>{search.data?.db2_results.map((item) => <article key={item.id} className="record"><Badge>{item.record_type}</Badge><p>{item.text}</p></article>) ?? <p className="muted">No query yet.</p>}</Card>
      </div>
    </div>
  );
}
