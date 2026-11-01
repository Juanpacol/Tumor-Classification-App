"""Use case: the deterministic rubric that checks each claim against the
evidence bundle. A claim is verified only when every evidence key it cites
actually resolves to real evidence - formatting a citation is not the same
as the citation supporting the claim."""

from application.evidence_bundle import EvidenceBundle
from application.grounded_report import ClaimOutput
from domain.models import Citation


class VerifyCitations:
    def execute(
        self, claims: list[ClaimOutput], bundle: EvidenceBundle
    ) -> list[Citation]:
        return [self._verify(claim, bundle) for claim in claims]

    def _verify(self, claim: ClaimOutput, bundle: EvidenceBundle) -> Citation:
        resolved = [bundle.get(key) for key in claim.evidence_keys]
        verified = bool(claim.evidence_keys) and all(
            ref is not None for ref in resolved
        )
        return Citation(
            claim=claim.text,
            evidence=[ref for ref in resolved if ref is not None],
            verified=verified,
        )
