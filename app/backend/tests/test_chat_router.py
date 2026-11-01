import json

from adapters.outbound.llm.fake_llm import FakeLLM
from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from domain.errors import DomainError
from domain.models import StudyStatus
from tests.factories import make_message, make_prediction, make_study
from tests.test_studies_router import client_with
from tests.test_study_chat import StreamingLLM


async def completed_study(
    repository: InMemoryStudyRepository, storage: InMemoryStorage | None = None
):
    study = make_study(status=StudyStatus.COMPLETED)
    await repository.save(study)
    await repository.save_prediction(make_prediction(study.id))
    if storage is not None:
        storage.save(b"image", study.file_key)
    return study


def events(body: str) -> list[dict]:
    return [
        json.loads(block.removeprefix("data: "))
        for block in body.split("\n\n")
        if block
    ]


async def test_list_chat_returns_the_messages_oldest_first() -> None:
    repository = InMemoryStudyRepository()
    study = await completed_study(repository)
    await repository.add_message(make_message(study.id, minutes=1, content="second"))
    await repository.add_message(make_message(study.id, minutes=0, content="first"))

    with client_with(repository=repository) as client:
        response = client.get(f"/api/studies/{study.id}/chat")

    assert [m["content"] for m in response.json()] == ["first", "second"]


def test_list_chat_of_an_unknown_study_is_404() -> None:
    with client_with() as client:
        assert client.get("/api/studies/missing/chat").status_code == 404


async def test_post_chat_streams_tokens_then_done() -> None:
    repository = InMemoryStudyRepository()
    storage = InMemoryStorage()
    study = await completed_study(repository, storage)

    with client_with(repository=repository, storage=storage, llm=FakeLLM()) as client:
        response = client.post(f"/api/studies/{study.id}/chat", json={"message": "Hi"})

    assert response.headers["content-type"].startswith("text/event-stream")
    sent = events(response.text)
    assert [e["type"] for e in sent] == ["token", "token", "token", "done"]
    assert sent[-1]["message"]["content"] == "fake response"
    assert sent[-1]["message"]["role"] == "assistant"


async def test_post_chat_rejects_an_empty_message() -> None:
    repository = InMemoryStudyRepository()
    study = await completed_study(repository)

    with client_with(repository=repository) as client:
        response = client.post(f"/api/studies/{study.id}/chat", json={"message": ""})

    assert response.status_code == 422


async def test_post_chat_with_the_llm_unavailable_is_503() -> None:
    repository = InMemoryStudyRepository()
    storage = InMemoryStorage()
    study = await completed_study(repository, storage)
    llm = StreamingLLM(fail=DomainError("LLM_UNAVAILABLE", "No key.", status_code=503))

    with client_with(repository=repository, storage=storage, llm=llm) as client:
        response = client.post(f"/api/studies/{study.id}/chat", json={"message": "Hi"})

    assert (response.status_code, response.json()["code"]) == (503, "LLM_UNAVAILABLE")


async def test_post_chat_reports_a_failure_after_the_stream_started() -> None:
    repository = InMemoryStudyRepository()
    storage = InMemoryStorage()
    study = await completed_study(repository, storage)

    class DropsMidAnswer(StreamingLLM):
        async def stream(self, messages, image=None):
            yield "partial"
            raise DomainError("LLM_UNAVAILABLE", "Lost.", status_code=503)

    with client_with(
        repository=repository, storage=storage, llm=DropsMidAnswer()
    ) as client:
        response = client.post(f"/api/studies/{study.id}/chat", json={"message": "Hi"})

    sent = events(response.text)
    assert [e["type"] for e in sent] == ["token", "error"]
    assert sent[-1] == {"type": "error", "code": "LLM_UNAVAILABLE", "message": "Lost."}
    assert await repository.list_messages(study.id) == []


def test_post_chat_of_an_unknown_study_is_404() -> None:
    with client_with() as client:
        response = client.post("/api/studies/missing/chat", json={"message": "Hi"})

    assert response.status_code == 404
