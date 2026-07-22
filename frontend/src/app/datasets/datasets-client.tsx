"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api, errorMessage } from "@/lib/api";

export function DatasetsClient() {
  const [selectedRevision, setSelectedRevision] = useState<string | null>(null);
  const revisions = useQuery({ queryKey: ["dataset-revisions"], queryFn: api.datasetRevisions });
  const items = revisions.data?.datasets ?? [];
  const selected = selectedRevision ?? items.at(-1)?.revision ?? null;
  const detail = useQuery({
    queryKey: ["dataset-revision", selected],
    queryFn: () => api.datasetRevision(selected as string),
    enabled: selected !== null,
  });
  const previous = selected ? items.find((item) => item.revision === selected)?.parent_revision : null;
  const diff = useQuery({
    queryKey: ["dataset-diff", previous, selected],
    queryFn: () => api.datasetRevisionDiff(previous as string, selected as string),
    enabled: previous !== null && selected !== null,
  });

  return (
    <div className="page">
      <header><h1 className="page-title">Dataset Governance</h1><p className="page-subtitle">Immutable revisions, source lineage, consent, privacy, quarantine, and fixed split assignments.</p></header>
      {revisions.error ? <p className="error">{errorMessage(revisions.error)}</p> : null}
      {detail.error ? <p className="error">{errorMessage(detail.error)}</p> : null}
      <div className="grid">
        <Card><CardTitle>Revisions</CardTitle>{items.length === 0 ? <p className="muted">No governed dataset revisions.</p> : <div className="stack">{items.toReversed().map((item) => <Button key={item.revision} onClick={() => setSelectedRevision(item.revision)}><span className="mono">{item.revision.slice(0, 12)}</span> · {item.record_count} records</Button>)}</div>}</Card>
        <Card><CardTitle>Revision Manifest</CardTitle>{detail.data ? <><p className="mono">{detail.data.manifest.revision}</p><p>Manifest checksum: <span className="mono">{detail.data.manifest.manifest_hash}</span></p><p>Splits: {formatCounts(detail.data.manifest.split_counts)}</p><p>Disposition: {formatCounts(detail.data.manifest.disposition_counts)}</p>{detail.data.manifest.quality_findings.map((finding) => <p className="error" key={finding}>{finding}</p>)}</> : <p className="muted">Select a revision.</p>}</Card>
        <Card><CardTitle>Version Diff</CardTitle>{diff.data ? <><p>Added {diff.data.added_record_ids.length} · removed {diff.data.removed_record_ids.length} · changed {diff.data.changed_record_ids.length}</p><p className="mono">{diff.data.from_revision.slice(0, 12)} → {diff.data.to_revision.slice(0, 12)}</p></> : <p className="muted">The first revision has no parent diff.</p>}</Card>
        <Card className="wide"><CardTitle>Governed Records</CardTitle>{detail.data?.records.map((record) => <article className="record" key={record.record_id}><div className="metadata-row"><Badge data-tone={record.disposition === "included" ? "success" : "danger"}>{record.disposition}</Badge>{record.split ? <Badge data-tone="accent">{record.split}</Badge> : null}<span>{record.privacy}</span><span>{record.consent}</span></div><h3>{record.provenance.source_kind}: {record.provenance.source_id}</h3><p>{record.inclusion_reason}</p><p className="muted">Events {record.provenance.source_event_ids.join(", ") || "none"} · memories {record.provenance.source_memory_ids.join(", ") || "none"} · decisions {record.provenance.source_decision_ids.join(", ") || "none"} · feedback {record.provenance.source_feedback_ids.join(", ") || "none"}</p>{record.quarantine_reasons.length ? <p className="error">Quarantine: {record.quarantine_reasons.join(", ")}</p> : null}{record.exclusion_reasons.length ? <p className="muted">Excluded: {record.exclusion_reasons.join(", ")}</p> : null}</article>)}</Card>
      </div>
    </div>
  );
}

function formatCounts(counts: Record<string, number>): string {
  return Object.entries(counts).map(([name, count]) => `${name} ${count}`).join(" · ") || "none";
}
