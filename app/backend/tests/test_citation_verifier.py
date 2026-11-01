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
        ClaimOutput(text="Grounded.", evidence_keys=["prediction.label"]),
        ClaimOutput(text="Not grounded.", evidence_keys=["prediction.not_real"]),
    ]

    grounded, ungrounded = VerifyCitations().execute(claims, _bundle())

    assert grounded.verified is True
    assert ungrounded.verified is False
