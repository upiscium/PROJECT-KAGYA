"""Purpose-separated live authoritative persistence codecs."""

from dataclasses import dataclass

from kagya.config.schema import Settings
from kagya.security.crypto import EncryptedCodec, load_key_ring


@dataclass(frozen=True)
class LiveCodecs:
    snapshot: EncryptedCodec
    wal: EncryptedCodec
    journal: EncryptedCodec
    chat_request_spool: EncryptedCodec


def build_live_codecs(settings: Settings) -> LiveCodecs:
    enabled = settings.at_rest.live.enabled
    ring = load_key_ring(settings.at_rest.live.keys) if enabled else None
    return LiveCodecs(
        snapshot=EncryptedCodec(
            enabled=enabled,
            purpose="live-state",
            context="agent-snapshot",
            key_ring=ring,
        ),
        wal=EncryptedCodec(
            enabled=enabled,
            purpose="live-state",
            context="private-wal",
            key_ring=ring,
        ),
        journal=EncryptedCodec(
            enabled=enabled,
            purpose="live-state",
            context="operator-journal",
            key_ring=ring,
        ),
        chat_request_spool=EncryptedCodec(
            enabled=enabled,
            purpose="chat-request-spool",
            context="request-record",
            key_ring=ring,
        ),
    )
