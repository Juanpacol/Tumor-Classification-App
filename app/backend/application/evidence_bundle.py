"""Use case: turn a prediction's own output into the verifiable facts a
report's claims can be checked against (the citation verifier's evidence)."""

from dataclasses import dataclass

from domain.models import EvidenceRef, Prediction


@dataclass
class EvidenceBundle:
    refs: list[EvidenceRef]

    def get(self, key: str) -> EvidenceRef | None:
        return next((ref for ref in self.refs if ref.key == key), None)


class BuildEvidenceBundle:
    def execute(self, prediction: Prediction) -> EvidenceBundle:
        refs = [
            EvidenceRef(key="prediction.label", description=prediction.label),
            EvidenceRef(
                key="prediction.confidence",
                description=f"{prediction.confidence:.3f}",
            ),
            EvidenceRef(
                key="prediction.needs_expert_triage",
                description="yes" if prediction.needs_expert_triage else "no",
            ),
        ]
        for label, value in prediction.probabilities.items():
            refs.append(
                EvidenceRef(
                    key=f"prediction.probabilities.{label}",
                    description=f"{value:.3f}",
                )
            )
        for feature in prediction.top_features:
            name = feature.get("name")
            if name is None:
                continue
            refs.append(
                EvidenceRef(
                    key=f"prediction.top_features.{name}",
                    description=(
                        f"value {feature.get('value')}, "
                        f"z-score {feature.get('z_score', 'n/a')}"
                    ),
                )
            )
        return EvidenceBundle(refs=refs)
