"""Deployment topology validation outside schema parsing."""

import socket

from kagya.config.schema import Settings


def validate_deployment_hostname(
    settings: Settings, *, actual_hostname: str | None = None
) -> None:
    """Reject startup on an unexpected host when enforcement is enabled."""

    node = settings.deployment.node
    if not node.enforce_hostname_match:
        return
    current = actual_hostname or socket.gethostname()
    if current != node.expected_hostname:
        raise RuntimeError(
            f"Node {node.id} expected hostname {node.expected_hostname!r}, "
            f"but startup host is {current!r}"
        )
