import pytest

from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from application.get_study import GetStudy
from domain.errors import DomainError
from tests.factories import make_study


async def test_returns_the_study():
    repo = InMemoryStudyRepository()
    study = make_study()
    await repo.save(study)

    detail = await GetStudy(repo).execute(study.id)

    assert detail.study == study


async def test_raises_404_for_an_unknown_study():
    with pytest.raises(DomainError) as excinfo:
        await GetStudy(InMemoryStudyRepository()).execute("missing")

    assert excinfo.value.status_code == 404
