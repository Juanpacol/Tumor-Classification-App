from application.evidence_bundle import BuildEvidenceBundle
from tests.factories import make_prediction


def test_bundle_carries_the_predicted_label_and_confidence():
    prediction = make_prediction("s1", label="glioma", confidence=0.93)

    bundle = BuildEvidenceBundle().execute(prediction)

    assert bundle.get("prediction.label").description == "glioma"
    assert bundle.get("prediction.confidence").description == "0.930"


def test_bundle_carries_expert_triage_as_yes_or_no():
    prediction = make_prediction("s1", needs_expert_triage=False)

    bundle = BuildEvidenceBundle().execute(prediction)

    assert bundle.get("prediction.needs_expert_triage").description == "no"


def test_bundle_carries_every_class_probability():
    prediction = make_prediction("s1", probabilities={"glioma": 0.7, "meningioma": 0.3})

    bundle = BuildEvidenceBundle().execute(prediction)

    assert bundle.get("prediction.probabilities.glioma").description == "0.700"
    assert bundle.get("prediction.probabilities.meningioma").description == "0.300"


def test_bundle_carries_the_top_features_and_falls_back_on_missing_z_score():
    prediction = make_prediction(
        "s1", top_features=[{"name": "glcm_contrast", "value": 0.42}]
    )

    bundle = BuildEvidenceBundle().execute(prediction)

    ref = bundle.get("prediction.top_features.glcm_contrast")
    assert ref.description == "value 0.42, z-score n/a"


def test_bundle_has_no_entry_for_an_unknown_key():
    bundle = BuildEvidenceBundle().execute(make_prediction("s1"))

    assert bundle.get("prediction.not_a_real_key") is None
