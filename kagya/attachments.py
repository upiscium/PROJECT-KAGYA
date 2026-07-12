"""Safe attachment validation for first multimodal milestone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image


SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_ATTACHMENTS = 1


@dataclass(frozen=True)
class ProcessedImageAttachment:
    name: str
    content_type: str
    path: Path
    image: Image.Image


def validate_image_attachments(
    attachments: list[dict[str, Any]],
) -> list[ProcessedImageAttachment]:
    """Validate local image attachments and load them for capable providers."""

    image_attachments = [item for item in attachments if item.get("type") == "image"]
    unsupported = [item for item in attachments if item.get("type") != "image"]
    if unsupported:
        raise ValueError("Only image attachments are supported by the first multimodal milestone")
    if len(image_attachments) > MAX_IMAGE_ATTACHMENTS:
        raise ValueError("Only one image attachment is supported")

    processed: list[ProcessedImageAttachment] = []
    for item in image_attachments:
        content_type = str(item.get("content_type") or "")
        if content_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError("Unsupported image content type")
        path = _file_url_to_path(str(item.get("url") or ""))
        if not path.is_file():
            raise ValueError("Image attachment file does not exist")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Image attachment exceeds size limit")
        try:
            image = Image.open(path)
            image.load()
        except Exception as exc:
            raise ValueError("Image attachment could not be decoded") from exc
        processed.append(
            ProcessedImageAttachment(
                name=str(item.get("name") or path.name),
                content_type=content_type,
                path=path,
                image=image,
            )
        )
    return processed


def _file_url_to_path(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("Only local file:// image attachments are supported")
    if not parsed.path:
        raise ValueError("Image attachment URL is missing a file path")
    return Path(unquote(parsed.path)).expanduser().resolve()
