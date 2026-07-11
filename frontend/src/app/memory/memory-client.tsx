"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useState } from "react";
import {
  api,
  errorMessage,
  type EpisodeMemory,
  type SemanticMemory,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export function MemoryClient() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [tagInputs, setTagInputs] = useState<Record<string, string>>({});
  const queryClient = useQueryClient();
  const search = useQuery({
    queryKey: ["memory", submitted],
    queryFn: () => api.memorySearch(submitted),
    enabled: submitted.length > 0,
  });
  const refreshMemory = () =>
    queryClient.invalidateQueries({ queryKey: ["memory", submitted] });
  const archiveEpisode = useMutation({
    mutationFn: api.archiveEpisodeMemory,
    onSuccess: refreshMemory,
  });
  const archiveSemantic = useMutation({
    mutationFn: api.archiveSemanticMemory,
    onSuccess: refreshMemory,
  });
  const updateEpisodeTags = useMutation({
    mutationFn: ({ id, tags }: { id: string; tags: string[] }) =>
      api.updateEpisodeMemoryMetadata(id, { tags }),
    onSuccess: refreshMemory,
  });
  const updateSemanticTags = useMutation({
    mutationFn: ({ id, tags }: { id: string; tags: string[] }) =>
      api.updateSemanticMemoryMetadata(id, { tags }),
    onSuccess: refreshMemory,
  });

  const addTag = (item: EpisodeMemory | SemanticMemory, kind: "episode" | "semantic") => {
    const value = (tagInputs[item.id] ?? "").trim();
    if (!value) return;
    const tags = [...(item.tags ?? []).filter((tag) => tag !== value), value];
    setTagInputs((current) => ({ ...current, [item.id]: "" }));
    if (kind === "episode") updateEpisodeTags.mutate({ id: item.id, tags });
    else updateSemanticTags.mutate({ id: item.id, tags });
  };

  return (
    <div className="page">
      <header><h1 className="page-title">Memory</h1><p className="page-subtitle">Search DB1 episodes and DB2 semantic memories.</p></header>
      <form className="composer" onSubmit={(event) => { event.preventDefault(); setSubmitted(query.trim()); }}>
        <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search memory" />
        <Button disabled={!query.trim()} type="submit">Search</Button>
      </form>
      {search.error ? <p className="error">{errorMessage(search.error)}</p> : null}
      {archiveEpisode.error || archiveSemantic.error || updateEpisodeTags.error || updateSemanticTags.error ? (
        <p className="error">
          {errorMessage(
            archiveEpisode.error ?? archiveSemantic.error ?? updateEpisodeTags.error ?? updateSemanticTags.error,
          )}
        </p>
      ) : null}
      <div className="grid">
        <Card>
          <CardTitle>DB1 Episodes</CardTitle>
          {memoryResults(search.data?.db1_results, submitted, "No DB1 episodes matched this query.", (item) => (
            <article key={item.id} className="record">
              <RecordHeader item={item} />
              <h3>{item.user_input}</h3>
              <p>{item.response}</p>
              <MemoryControls
                item={item}
                kind="episode"
                value={tagInputs[item.id] ?? ""}
                onInput={(value) => setTagInputs((current) => ({ ...current, [item.id]: value }))}
                onAddTag={() => addTag(item, "episode")}
                onArchive={() => archiveEpisode.mutate(item.id)}
              />
            </article>
          ))}
        </Card>
        <Card>
          <CardTitle>DB2 Semantic</CardTitle>
          {memoryResults(search.data?.db2_results, submitted, "No DB2 semantic memories matched this query.", (item) => (
            <article key={item.id} className="record">
              <RecordHeader item={item} />
              <p>{item.text}</p>
              <MemoryControls
                item={item}
                kind="semantic"
                value={tagInputs[item.id] ?? ""}
                onInput={(value) => setTagInputs((current) => ({ ...current, [item.id]: value }))}
                onAddTag={() => addTag(item, "semantic")}
                onArchive={() => archiveSemantic.mutate(item.id)}
              />
            </article>
          ))}
        </Card>
      </div>
    </div>
  );
}

function RecordHeader({ item }: { item: EpisodeMemory | SemanticMemory }) {
  return (
    <div>
      <Badge>{item.record_type}</Badge>
      {item.archived ? <Badge>archived</Badge> : null}
      {(item.tags ?? []).map((tag) => <Badge key={tag}>{tag}</Badge>)}
    </div>
  );
}

function MemoryControls({
  item,
  kind,
  value,
  onInput,
  onAddTag,
  onArchive,
}: {
  item: EpisodeMemory | SemanticMemory;
  kind: "episode" | "semantic";
  value: string;
  onInput: (value: string) => void;
  onAddTag: () => void;
  onArchive: () => void;
}) {
  return (
    <div className="composer" aria-label={`${kind} controls for ${item.id}`}>
      <Input value={value} onChange={(event) => onInput(event.target.value)} placeholder="Add tag" />
      <Button type="button" disabled={!value.trim()} onClick={onAddTag}>Add tag</Button>
      <Button type="button" disabled={item.archived} onClick={onArchive}>Archive</Button>
    </div>
  );
}

function memoryResults<T>(items: T[] | undefined, submitted: string, emptyMessage: string, render: (item: T) => ReactNode) {
  if (!submitted) return <p className="muted">Enter a query to search memory.</p>;
  if (!items) return <p className="muted">Searching memory...</p>;
  if (items.length === 0) return <p className="muted">{emptyMessage}</p>;
  return items.map(render);
}
