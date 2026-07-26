"""Versioned AES-256-GCM envelopes backed by environment-only key rings."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
import json
import os
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from kagya.config.schema import KeyRingSettings


_FORMAT = "kagya-aead"
_VERSION = 1
_ALGORITHM = "AES-256-GCM"
_NONCE_BYTES = 12
_KEY_BYTES = 32


class EncryptionError(RuntimeError):
    """Raised when encrypted state cannot be authenticated or decoded."""


@dataclass(frozen=True)
class KeyRing:
    current_key_id: str
    keys: Mapping[str, bytes]

    @property
    def current_key(self) -> bytes:
        return self.keys[self.current_key_id]


def load_key_ring(settings: KeyRingSettings) -> KeyRing:
    """Load an explicit key generation ring without retaining env names in output."""

    configured = {
        settings.current_key_id: settings.current_key_env,
        **settings.allowed_old_key_envs,
    }
    keys: dict[str, bytes] = {}
    for key_id, env_name in configured.items():
        encoded = os.getenv(env_name)
        if encoded is None:
            raise EncryptionError(f"required encryption key {key_id!r} is unavailable")
        try:
            key = b64decode(encoded, validate=True)
        except ValueError as exc:
            raise EncryptionError(
                f"encryption key {key_id!r} is not strict base64"
            ) from exc
        if len(key) != _KEY_BYTES:
            raise EncryptionError(
                f"encryption key {key_id!r} must decode to exactly 32 bytes"
            )
        keys[key_id] = key
    return KeyRing(settings.current_key_id, keys)


class EncryptedCodec:
    """Encode one independently authenticated record for a single purpose/context."""

    def __init__(
        self,
        *,
        enabled: bool,
        purpose: str,
        context: str,
        key_ring: KeyRing | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9._-]+", purpose):
            raise ValueError("encryption purpose is invalid")
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", context):
            raise ValueError("encryption context is invalid")
        if enabled and key_ring is None:
            raise EncryptionError("enabled encryption requires a key ring")
        self.enabled = enabled
        self.purpose = purpose
        self.context = context
        self.key_ring = key_ring

    def encode(
        self, plaintext: bytes, *, metadata: Mapping[str, Any] | None = None
    ) -> bytes:
        if not self.enabled:
            return plaintext
        assert self.key_ring is not None
        nonce = os.urandom(_NONCE_BYTES)
        header = {
            "format": _FORMAT,
            "version": _VERSION,
            "algorithm": _ALGORITHM,
            "key_id": self.key_ring.current_key_id,
            "purpose": self.purpose,
            "context": self.context,
            "nonce": b64encode(nonce).decode("ascii"),
            "metadata": dict(metadata or {}),
        }
        aad = _canonical(header)
        key = _derive_key(
            self.key_ring.current_key, self.purpose, self.context
        )
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
        return _canonical(
            {**header, "ciphertext": b64encode(ciphertext).decode("ascii")}
        )

    def decode(
        self,
        encoded: bytes,
        *,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> bytes:
        looks_encrypted = encoded.lstrip().startswith(b'{"algorithm":"AES-256-GCM"')
        if not self.enabled:
            if looks_encrypted or b'"format":"kagya-aead"' in encoded[:256]:
                raise EncryptionError("encrypted content is not allowed by this codec")
            return encoded
        assert self.key_ring is not None
        try:
            envelope = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EncryptionError("encrypted envelope is malformed or plaintext") from exc
        if not isinstance(envelope, dict):
            raise EncryptionError("encrypted envelope root must be an object")
        required = {
            "format",
            "version",
            "algorithm",
            "key_id",
            "purpose",
            "context",
            "nonce",
            "metadata",
            "ciphertext",
        }
        if set(envelope) != required:
            if envelope.get("format") != _FORMAT:
                raise EncryptionError("encrypted envelope is malformed or plaintext")
            raise EncryptionError("encrypted envelope schema is invalid")
        if (
            envelope["format"] != _FORMAT
            or envelope["version"] != _VERSION
            or envelope["algorithm"] != _ALGORITHM
            or envelope["purpose"] != self.purpose
            or envelope["context"] != self.context
        ):
            raise EncryptionError("encrypted envelope header is invalid")
        if not isinstance(envelope["metadata"], dict):
            raise EncryptionError("encrypted envelope metadata is invalid")
        if expected_metadata is not None and envelope["metadata"] != dict(
            expected_metadata
        ):
            raise EncryptionError("encrypted envelope sequence metadata is invalid")
        key_id = envelope["key_id"]
        if not isinstance(key_id, str) or key_id not in self.key_ring.keys:
            raise EncryptionError("encrypted envelope key generation is not allowed")
        try:
            nonce = b64decode(envelope["nonce"], validate=True)
            ciphertext = b64decode(envelope["ciphertext"], validate=True)
        except (TypeError, ValueError) as exc:
            raise EncryptionError("encrypted envelope base64 is invalid") from exc
        if len(nonce) != _NONCE_BYTES or len(ciphertext) < 16:
            raise EncryptionError("encrypted envelope nonce or ciphertext is invalid")
        header = {key: value for key, value in envelope.items() if key != "ciphertext"}
        key = _derive_key(self.key_ring.keys[key_id], self.purpose, self.context)
        try:
            return AESGCM(key).decrypt(nonce, ciphertext, _canonical(header))
        except InvalidTag as exc:
            raise EncryptionError("encrypted envelope authentication failed") from exc


def _derive_key(root_key: bytes, purpose: str, context: str) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES,
        salt=b"PROJECT-KAGYA at-rest envelope v1",
        info=f"{purpose}\0{context}".encode("ascii"),
    ).derive(root_key)


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as exc:
        raise EncryptionError("envelope metadata is not canonical JSON") from exc
