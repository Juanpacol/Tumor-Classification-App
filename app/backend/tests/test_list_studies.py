from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from application.list_studies import ListStudies
from tests.factories import make_prediction, make_report, make_study


async def test_lists_studies_with_their_prediction_and_report():
    repo = InMemoryStudyRepository()
    study = make_study()
    await repo.save(study)
    await repo.save_prediction(make_prediction(study.id))
    await repo.save_report(make_report(study.id))

    page = await ListStudies(repo).execute(page=1, page_size=20)

    assert page.total == 1
    assert page.items[0].study == study
    assert page.items[0].prediction is not None
    assert page.items[0].report is not None


async def test_study_without_results_has_none_prediction_and_report():
    repo = InMemoryStudyRepository()
    await repo.save(make_study())

    page = await ListStudies(repo).execute()

    assert page.items[0].prediction is None
    assert page.items[0].report is None
