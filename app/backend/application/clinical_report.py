"""Structure and fixed wording of the AI-drafted clinical report."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

DISCLAIMER = (
    "AI-generated draft for decision support only. It is not a diagnosis and "
    "must be reviewed and confirmed by a qualified radiologist."
)


class ClinicalReportOutput(BaseModel):
    """What the model must return, validated before anything is stored."""

    model_config = ConfigDict(extra="forbid")

    findings: str
    visual_observations: str
    impression: str
    differential_considerations: str
    suggested_next_steps: list[str]
    ai_agreement: Literal["agrees", "disagrees", "uncertain"]
