# Encrypted State And Restore Runbook

## Security Boundary

When live encryption is enabled, live snapshot, private WAL, operator-safe Journal,
and active Chat request spool encryption uses AES-256-GCM from `cryptography`.
Every record has a random 96-bit nonce and authenticated
format, version, purpose, context, key ID, and sequence/record metadata. HKDF-SHA256
derives separate context keys from each 32-byte root key. Live state, backup, and
adapter artifact key rings are independent.

Chat requests use the live key ring with the separate `chat-request-spool`
purpose. Operation and event IDs are authenticated metadata. Terminal Chat records
and idempotency tombstones do not retain request ciphertext, and no adjacent
`chat_jobs.json.key` is created.

Root keys are strict base64 environment values whose names are configured under
`at_rest`. Do not put key values in YAML, files, command arguments, logs, manifests,
or tickets. A ring has one write key ID and only explicitly configured old read key
IDs. Removing an old ID makes that generation unreadable by design.

Chroma and plaintext restore staging do not provide transparent application-level
encryption through this integration. Production therefore requires separate
`memory_encrypted_filesystem_attested: true` and
`backup.encrypted_filesystem_attested: true` attestations. Place both
`memory.persist_directory` and `backup.restore_staging_directory` on encrypted
filesystems or volumes and verify their boot/unlock/access controls.

## Configure

Generate independent keys without writing them to the repository:

```bash
export KAGYA_LIVE_STATE_KEY="$(openssl rand -base64 32)"
export KAGYA_BACKUP_KEY="$(openssl rand -base64 32)"
export KAGYA_ADAPTER_ARTIFACT_KEY="$(openssl rand -base64 32)"
```

For production, set `project.environment: production`, `at_rest.live.enabled:
true`, and both filesystem attestations. Production first boot is intentionally
sealed: missing snapshot, WAL, Journal, or generation marker is data loss, not an
empty state. Initialize a new empty deployment exactly once while the API is down:

```bash
uv run kagya-backup --config /path/to/config.yaml state-encryption-init
```

Development may explicitly retain `at_rest.live.enabled: false` and keeps its
existing first-boot behavior.

If plaintext development state already exists, stop the API and run the explicit
one-shot migration. Startup never silently migrates encrypted production state:

```bash
just live-encryption-migrate /path/to/config.yaml
```

After upgrading a deployment that has the retired adjacent-key Chat spool, keep
the API stopped and migrate it atomically before startup:

```bash
uv run kagya-backup --config /path/to/config.yaml migrate-chat-spool
```

The migration authenticates every legacy request before replacing the mixed full
record/tombstone registry, then removes the adjacent key. Runtime startup rejects
legacy, mixed, tampered, moved, or unavailable-key request ciphertext.

## Back Up And Verify

The `.kgb` format streams each source file, including the Chat registry but never
its retired adjacent key, into independently encrypted chunks and
never creates a plaintext tar/archive. Its encrypted internal manifest records
relative names, sizes, SHA-256 digests, schema/source/model/adapter revisions,
backup ID/time, and incremental base bindings. Adapter chunks receive an inner,
purpose-separated encryption layer. The public status sidecar contains only format
version, backup ID/time, encrypted size/hash, and key IDs.

```bash
just backup-create /path/to/config.yaml
uv run kagya-backup --config /path/to/config.yaml list --limit 20
just backup-verify <backup-id> /path/to/config.yaml
uv run kagya-backup --config /path/to/config.yaml preview <backup-id>
```

Use `scheduled` from a systemd timer or cron. It respects
`schedule_interval_seconds`, creates incremental backups between configured full
intervals, and enforces `retention_count`:

```bash
just backup-scheduled /path/to/config.yaml
```

Retention counts full generations/chains. Every retained incremental keeps its
transitive bases. A superseded full backup and all dependent incrementals are
deleted together only after a newer verified full chain exists; orphaned
incrementals are never listed as restorable.

Deletion is best effort. Unlinking expired bundles does not guarantee physical
erasure on SSDs, flash translation layers, snapshots, journaled filesystems, RAID,
or copy-on-write storage. Use encrypted-volume key destruction and storage-provider
lifecycle controls when cryptographic erasure is required.

## Restore

Preview supplies the encrypted manifest hash required for destructive commit.
Stop the API for the preferred offline workflow:

```bash
uv run kagya-backup --config /path/to/config.yaml verify <backup-id>
uv run kagya-backup --config /path/to/config.yaml preview <backup-id>
uv run kagya-backup --config /path/to/config.yaml restore <backup-id> \
  --manifest-hash <expected-manifest-sha256>
```

The admin API exposes equivalent bounded list, create, verify, preview, rotate, and
commit operations under `/api/state/backups`. Commit requires the existing full
admin and recent re-authentication boundary, an expected backup ID, and expected
manifest hash. It drains the single `AgentRuntime`, swaps only after isolated
validation, and rebuilds a fresh runtime graph before serving restored state.

Restore decrypts into a mode-`0700` child of the dedicated
`backup.restore_staging_directory`, with files mode `0600`. This directory must not
default beneath backup storage and requires its own production encrypted-filesystem
attestation. Restore validates AEAD, manifest and chunk checksums,
incremental base hashes, snapshot/WAL/Journal continuity, and source/model/adapter
revisions before swap. The prior generation remains in `previous-generation` for
rollback. Failed pre-swap verification changes nothing; an interrupted swap rolls
back completed replacements.

The isolated directory contains plaintext while restore is running. It is deleted
on success and ordinary failure, but plaintext may survive process/power loss,
filesystem snapshots, swap, forensic recovery, or privileged host compromise.
Restrict host/root access, disable unencrypted swap, and inspect abandoned
`kagya-restore-*` directories after a crash. Ordinary success and failure clean the
staging child. The design does not claim secure deletion on CoW filesystems or SSDs.

The durable `.restore-in-progress` marker is removed only after the restored graph
is rebuilt and verified, or after rollback and prior-generation verification both
succeed. Any rollback replace, fsync, or rebuild failure preserves bounded phase
metadata and blocks API startup. Do not delete the marker manually. With the API
stopped, inspect and retry recovery:

```bash
uv run kagya-backup --config /path/to/config.yaml recovery-status
uv run kagya-backup --config /path/to/config.yaml recovery-retry
```

## Rotate

Add the old key ID/environment name to `allowed_old_key_envs`, configure a new
current key ID/environment name, then re-encrypt and verify:

```bash
uv run kagya-backup --config /path/to/config.yaml rotate <backup-id>
just live-encryption-rotate /path/to/config.yaml
```

After every required live file and backup has been re-encrypted and verified,
remove the old allowed ID and retire its secret. A missing or omitted old key fails
closed; there is no plaintext fallback.

`rotate-live` also rewrites every active Chat request under the current live key
generation while preserving request-free terminal records and tombstones.
