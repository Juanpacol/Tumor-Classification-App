from domain.models import Prediction


class FakeClassifier:
    def classify(self, image_path: str) -> Prediction:
        return Prediction(
            study_id="fake",
            label="no_tumor",
            probabilities={
                "glioma": 0.05,
                "meningioma": 0.05,
                "pituitary": 0.05,
                "no_tumor": 0.85,
            },
            confidence=0.85,
            needs_expert_triage=False,
            quality={},
            top_features=[],
            model_version="fake-0.0.0",
        )

    def is_ready(self) -> bool:
        return True
