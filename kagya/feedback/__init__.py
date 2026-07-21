"""Structured explicit feedback domain."""

from kagya.feedback.records import (
    FeedbackPropagation,
    FeedbackProvenance,
    FeedbackRecord,
    FeedbackRevision,
    FeedbackSignal,
    FeedbackStatus,
    FeedbackStore,
    FeedbackTarget,
    FeedbackTargetType,
    TrainingDisposition,
    ValueEvidenceProposal,
    feedback_fingerprint,
    normalize_signals,
)

__all__ = [
    "FeedbackPropagation",
    "FeedbackProvenance",
    "FeedbackRecord",
    "FeedbackRevision",
    "FeedbackSignal",
    "FeedbackStatus",
    "FeedbackStore",
    "FeedbackTarget",
    "FeedbackTargetType",
    "TrainingDisposition",
    "ValueEvidenceProposal",
    "feedback_fingerprint",
    "normalize_signals",
]
