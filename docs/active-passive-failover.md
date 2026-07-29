# Active-Passive Failover Runbook

KAGYA failover uses an etcd v3 lease only to select a writer. Authoritative
visibility is the etcd transaction that creates an immutable state bundle
and advances its watermark while comparing the active lease key, fencing epoch,
tenure value, and previous watermark revision. A local snapshot or WAL tail that
was not watermark-published is never promotion input.

## Preconditions

- Operate an odd-sized etcd cluster with quorum, durable storage, authentication,
  TLS, backups, and a request-size limit at least 40% above
  `failover.etcd.max_bundle_bytes` for transaction JSON/base64 overhead.
- Give each subject node a unique `deployment.node.id`, the same failover key
  prefix, exact immutable model/processor/fallback revisions, adapter artifacts,
  and encryption key environment. Replication copies encrypted snapshot, WAL,
  Journal, and Chat request ciphertext plus adapter registry/history; it never
  copies encryption keys or model/adapter artifact bodies.
- Configure both nodes for the same durable Chroma HTTP tenant/database. KAGYA
  writes event memory as `pending`, publishes the etcd generation, and only then
  marks that event committed. A stale node cannot expose memory without first
  winning the watermark CAS. Production requires TLS and an auth token env.
- Migrate/scrub legacy local Chroma data before enabling failover. Failover
  startup is read-only before reconciliation and will not run legacy backfills
  against the shared service.
- Production endpoints must use HTTPS and `auth_token_env`. The environment
  value is sent as the etcd `Authorization` header and must not be logged.
- Keep etcd data and bundle keys. Deleting the prefix destroys fencing history
  and requires a deliberate new-cluster bootstrap from one audited source.
- KAGYA deletes the superseded immutable bundle after advancing the watermark.
  Configure normal etcd history compaction/defragmentation so deleted revisions
  do not eventually exhaust the cluster quota.
- Route mutation traffic only to a node whose readiness reports `active`.

## Manual Failover

1. Stop or isolate the old active and verify its lease key has expired. Do not
   promote while etcd quorum is unavailable.
2. Configure exactly one standby with `subject_role: active`, leave
   `automatic_promotion: false`, and start it.
3. Startup must acquire a higher fencing epoch, download the current immutable
   bundle, and pass Journal/WAL/snapshot sequence and hash continuity plus exact
   model, processor, fallback, and active-adapter identity checks.
4. Confirm `/health/ready` reports `subject_role: active`, a higher
   `fencing_token`, and all runtime checks ready before moving traffic.
5. Return the old node only as `subject_role: standby`. Never copy its local
   state over the promoted node.

For the first audited generation only, configure one stopped source as
`subject_role: active` with `bootstrap_from_local_state: true`. Disable that
flag immediately after its first watermark is published. Standbys and automatic
promotion can never bootstrap a missing watermark.

## Automatic Failover

Set `subject_role: standby` and `automatic_promotion: true` on eligible nodes.
All contenders use one etcd transaction; one acquires the lease and the others
remain standby. Use traffic routing based on active readiness. Automatic mode
does not bypass promotion preflight. A node that loses active authority remains
fenced; restart it as a standby before it campaigns again.

## Partition And Recovery

Any keepalive, linearizable lease verification, or publication failure fences
the process locally and aborts producers. Requests receive the stable
`not_authoritative` response. Availability is intentionally lost until etcd
quorum returns. In-flight Journal records are classified by the existing
recovery rules (`accepted_not_started`, `uncommitted_after_crash`, or
`committed_before_crash`); no handler payload or external side effect is
opaquely replayed.

Accepted Chat work is acknowledged only after its encrypted request spool and
`accepted` Journal evidence share one published watermark. A crash after
`started` consumes that processing sequence as a no-op uncommitted transition;
the promoted runtime continues at the next sequence without replaying the
opaque handler.

Online backup restore is rejected while failover is enabled because publishing
an older processing sequence would violate the monotonic watermark. Disable
failover and use a new audited etcd prefix for disaster recovery, or use the
normal point-in-time restore event when continuity must be preserved.

The current etcd-backed bundle transport is intentionally bounded by
`max_bundle_bytes`. If retained Journal/WAL ciphertext exceeds the etcd request
limit, or if Chat/adapter registry state makes the bundle too large,
publication fails closed and no mutation success is acknowledged. A future
large-state deployment should add an immutable replicated object store whose
content hash is CAS-bound by the same watermark; increasing the limit without
matching etcd server configuration is not safe.
