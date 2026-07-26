# Encrypted State And Restore Runbook

## Security Boundary

Live snapshot, private WAL, and operator-safe Journal encryption uses AES-256-GCM
from `cryptography`. Every record has a random 96-bit nonce and authenticated
format, version, purpose, context, key ID, and sequence/record metadata. HKDF-SHA256
derives separate context keys from each 32-byte root key. Live state, backup, and
adapter artifact key rings are independent.

Root keys are strict base64 environment values whose names are configured under
`at_rest`. Do not put key values in YAML, files, command arguments, logs, manifests,
or tickets. A ring has one write key ID and only explicitly configured old read key
IDs. Removing an old ID makes that generation unreadable by design.

Chroma does not provide transparent application-level encryption through this
integration. Production therefore requires `memory_encrypted_filesystem_attested:
true`, meaning the operator has placed `memory.persist_directory` on an encrypted
filesystem or volume and verified its boot/unlock/access controls. This is an
attestation, not application encryption.

## Configure

Generate independent keys without writing them to the repository:

```bash
export KAGYA_LIVE_STATE_KEY="$(openssl rand -base64 32)"
export KAGYA_BACKUP_KEY="$(openssl rand -base64 32)"
export KAGYA_ADAPTER_ARTIFACT_KEY="$(openssl rand -base64 32)"
```

For production, set `project.environment: production`, `at_rest.live.enabled:
true`, and the Chroma filesystem attestation. Development may explicitly retain
`at_rest.live.enabled: false`.

If plaintext development state already exists, stop the API and run the explicit
one-shot migration. Startup never silently migrates encrypted production state:

```bash
just live-encryption-migrate /path/to/config.yaml
```

## Back Up And Verify

The `.kgb` format streams each source file into independently encrypted chunks and
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

Restore decrypts into a mode-`0700` directory beneath the configured backup
directory, with files mode `0600`. It validates AEAD, manifest and chunk checksums,
incremental base hashes, snapshot/WAL/Journal continuity, and source/model/adapter
revisions before swap. The prior generation remains in `previous-generation` for
rollback. Failed pre-swap verification changes nothing; an interrupted swap rolls
back completed replacements.

The isolated directory contains plaintext while restore is running. It is deleted
on success and ordinary failure, but plaintext may survive process/power loss,
filesystem snapshots, swap, forensic recovery, or privileged host compromise.
Place `at_rest.backup.directory` on an encrypted filesystem, restrict host/root
access, disable unencrypted swap, and clean abandoned `kagya-restore-*` directories
after a crash. The design does not claim secure deletion on CoW filesystems or SSDs.

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
