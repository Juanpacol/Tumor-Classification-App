from application.citation_verifier import VerifyCitations
from application.evidence_bundle import BuildEvidenceBundle
from application.grounded_report import ClaimOutput
from tests.factories import make_prediction


def _bundle():
    return BuildEvidenceBundle().execute(make_prediction("s1"))


def test_a_claim_grounded_in_real_evidence_is_verified():
    claim = ClaimOutput(
        text="The predicted label is glioma.", evidence_keys=["prediction.label"]
    )

    [citation] = VerifyCitations().execute([claim], _bundle())

    assert citation.verified is True
    assert citation.evidence[0].key == "prediction.label"


def test_a_claim_citing_a_key_not_in_the_bundle_is_unverified():
    claim = ClaimOutput(
        text="The tumor is 3cm wide.", evidence_keys=["prediction.tumor_size"]
    )

    [citation] = VerifyCitations().execute([claim], _bundle())

    assert citation.verified is False
    assert citation.evidence == []


def test_a_claim_with_no_evidence_keys_is_unverified():
    claim = ClaimOutput(text="This looks concerning.", evidence_keys=[])

    [citation] = VerifyCitations().execute([claim], _bundle())

    assert citation.verified is False


def test_a_claim_is_unverified_if_any_one_of_its_keys_is_missing():
    claim = ClaimOutput(
        text="High confidence glioma.",
        evidence_keys=["prediction.label", "prediction.not_real"],
    )

    [citation] = VerifyCitations().execute([claim], _bundle())

    assert citation.verified is False
    assert [ref.key for ref in citation.evidence] == ["prediction.label"]


def test_claims_are_verified_independently_of_each_other():
    claims = [
        ClaimOutput(text="Grounded in glioma.", evidence_keys=["prediction.label"]),
        ClaimOutput(text="Not grounded.", evidence_keys=["prediction.not_real"]),
    ]

    grounded, ungrounded = VerifyCitations().execute(claims, _bundle())

    assert grounded.verified is True
    assert ungrounded.verified is False


def _verify(text: str, *keys: str) -> bool:
    [citation] = VerifyCitations().execute(
        [ClaimOutput(text=text, evidence_keys=list(keys))], _bundle()
    )
    return citation.verified


def test_a_number_matching_the_cited_value_is_verified():
    assert _verify("Confidence is 0.70.", "prediction.confidence")
    assert _verify("The model is 70% confident.", "prediction.confidence")


def test_a_number_contradicting_the_cited_value_is_unverified():
    assert not _verify("The classifier is 99% confident.", "prediction.confidence")
    assert not _verify("Confidence is 0.95.", "prediction.confidence")


def test_a_claim_naming_another_class_than_the_cited_label_is_unverified():
    assert not _verify("The predicted label is pituitary.", "prediction.label")
    assert not _verify(
        "The classifier is 70% confident it is meningioma.",
        "prediction.label",
        "prediction.confidence",
    )


def test_a_claim_citing_the_label_must_name_it():
    assert not _verify("There is midline shift.", "prediction.label")


def test_another_class_may_be_named_when_its_probability_is_cited():
    assert _verify(
        "Glioma is favored; meningioma is less likely at 10%.",
        "prediction.label",
        "prediction.probabilities.meningioma",
    )


def test_a_class_written_with_spaces_matches_its_key():
    assert _verify(
        "No tumor has a probability of 0.10.", "prediction.probabilities.no_tumor"
    )
    assert not _verify("It is likely no tumor.", "prediction.probabilities.glioma")
