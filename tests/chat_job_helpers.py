from typing import Any

from kagya.chat_jobs import ChatJobRegistry as _ChatJobRegistry
from kagya.security import EncryptedCodec
from kagya.security.crypto import KeyRing


def request_codec(key_id: str = "test", key: bytes = bytes(range(32))) -> EncryptedCodec:
    return EncryptedCodec(
        enabled=True,
        purpose="chat-request-spool",
        context="request-record",
        key_ring=KeyRing(key_id, {key_id: key}),
    )


class ChatJobRegistry(_ChatJobRegistry):
    def __init__(
        self,
        *args: Any,
        request_codec: EncryptedCodec | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            request_codec=request_codec or globals()["request_codec"](),
            **kwargs,
        )
