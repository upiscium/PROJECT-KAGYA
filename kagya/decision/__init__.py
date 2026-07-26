"""Structured action candidates and causal decision records."""

from kagya.decision.records import (
    ActionCandidate,
    ActionType,
    ActualOutcome,
    CandidateEvaluation,
    DecisionDatasetGenerator,
    DecisionDatasetRecord,
    DecisionRecord,
    DecisionStatus,
    DecisionStore,
    PredictedOutcome,
    parse_candidate_output,
    schema_candidate_prompt,
)
from kagya.decision.explanations import (
    DecisionExplanationStore,
    ExplanationDisposition,
    NaturalExplanationOutput,
    PublicDecisionExplanation,
    RendererState,
    build_explanation,
    explanation_input_digest,
    render_natural,
)

__all__ = [
    "ActionCandidate",
    "ActionType",
    "ActualOutcome",
    "CandidateEvaluation",
    "DecisionDatasetGenerator",
    "DecisionDatasetRecord",
    "DecisionRecord",
    "DecisionStatus",
    "DecisionStore",
    "PredictedOutcome",
    "parse_candidate_output",
    "schema_candidate_prompt",
    "DecisionExplanationStore",
    "ExplanationDisposition",
    "NaturalExplanationOutput",
    "PublicDecisionExplanation",
    "RendererState",
    "build_explanation",
    "explanation_input_digest",
    "render_natural",
]
