"""Structure of a citation-grounded clinical report: every narrative claim
carries the evidence-bundle keys (see application/evidence_bundle.py) it is
based on, instead of being a free-text blob."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ClaimOutput(BaseModel):
    """One statement plus the evidence-bundle keys it is grounded in."""

    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_keys: list[str]


class GroundedClinicalReportOutput(BaseModel):
    """What the model must return for a citation-grounded report."""

    model_config = ConfigDict(extra="forbid")

    findings: list[ClaimOutput]
    visual_observations: list[ClaimOutput]
    impression: list[ClaimOutput]
    differential_considerations: list[ClaimOutput]
    suggested_next_steps: list[str]
    ai_agreement: Literal["agrees", "disagrees", "uncertain"]
