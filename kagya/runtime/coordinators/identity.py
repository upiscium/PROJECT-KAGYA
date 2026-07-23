"""Identity and narrative coordination boundary."""

from typing import Callable

from kagya.identity import NarrativeSelf, SelfModel


class IdentityNarrativeCoordinator:
    def __init__(
        self,
        self_model: SelfModel,
        narrative: NarrativeSelf,
        *,
        persist_self_model: Callable[[], None],
        persist_narrative: Callable[[], None],
    ) -> None:
        self._self_model = self_model
        self._narrative = narrative
        self._persist_self_model = persist_self_model
        self._persist_narrative = persist_narrative

    def persist(self) -> None:
        self._persist_self_model()
        self._persist_narrative()
