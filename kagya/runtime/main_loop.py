"""Compatibility facade for the staged main-loop coordinators."""

from kagya.runtime.coordinated_main_loop import ChatResult, _MainLoopImplementation


class KagyaMainLoop(_MainLoopImplementation):
    """Authoritative coordinator and stable public runtime facade."""


__all__ = ["ChatResult", "KagyaMainLoop"]
