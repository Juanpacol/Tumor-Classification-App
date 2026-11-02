"""Use case: the deterministic rubric that checks each claim against the
evidence bundle. A claim is verified only when every evidence key it cites
actually resolves to real evidence and the claim does not contradict it -
formatting a citation is not the same as the citation supporting the claim."""

import re

from application.evidence_bundle import EvidenceBundle
from application.grounded_report import ClaimOutput
from domain.models import Citation, EvidenceRef

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%)?")
_PROBABILITY_KEY = "prediction.probabilities."
_LABEL_KEY = "prediction.label"


class VerifyCitations:
    def execute(
        self, claims: list[ClaimOutput], bundle: EvidenceBundle
    ) -> list[Citation]:
        classes = [
            ref.key.removeprefix(_PROBABILITY_KEY)
            for ref in bundle.refs
            if ref.key.startswith(_PROBABILITY_KEY)
        ]
        return [self._verify(claim, bundle, classes) for claim in claims]

    def _verify(
        self, claim: ClaimOutput, bundle: EvidenceBundle, classes: list[str]
    ) -> Citation:
        resolved = [bundle.get(key) for key in claim.evidence_keys]
        evidence = [ref for ref in resolved if ref is not None]
        verified = (
            bool(claim.evidence_keys)
            and len(evidence) == len(resolved)
            and _numbers_match(claim.text, evidence)
            and _classes_match(claim.text, evidence, classes)
        )
        return Citation(claim=claim.text, evidence=evidence, verified=verified)


def _numbers_match(text: str, evidence: list[EvidenceRef]) -> bool:
    """Every number the claim states must be one of the cited values (a
    probability may be written as a percentage)."""
    cited = [float(n) for ref in evidence for n, _ in _NUMBER.findall(ref.description)]
    if not cited:
        return True
    for number, percent in _NUMBER.findall(text):
        value = float(number)
        if not any(
            abs(value - c) < 0.01
            or ((percent or value > 1) and abs(value - c * 100) < 1)
            for c in cited
        ):
            return False
    return True


def _classes_match(text: str, evidence: list[EvidenceRef], classes: list[str]) -> bool:
    """A claim citing the label must name it, and may name another class only
    when it cites that class's probability."""
    keys = {ref.key for ref in evidence}
    allowed = {
        k.removeprefix(_PROBABILITY_KEY) for k in keys if k.startswith(_PROBABILITY_KEY)
    }
    label = next((ref.description for ref in evidence if ref.key == _LABEL_KEY), None)
    if label is not None:
        if not _mentions(text, label):
            return False
        allowed.add(label)
    if not allowed:
        return True
    return all(c in allowed or not _mentions(text, c) for c in classes)


def _mentions(text: str, label: str) -> bool:
    words = r"[\s_-]+".join(map(re.escape, re.split(r"[_\s]+", label)))
    return re.search(rf"\b{words}\b", text, re.IGNORECASE) is not None
