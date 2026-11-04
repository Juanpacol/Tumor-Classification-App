from collections.abc import AsyncIterator
from typing import Any

import pytest

from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from application.chat_prompt import HISTORY_LIMIT
from application.study_chat import StudyChat
from domain.errors import DomainError
from domain.models import ChatMessage, StudyStatus
from tests.factories import make_message, make_prediction, make_report, make_study


class StreamingLLM:
    def __init__(self, *chunks: str, fail: Exception | None = None) -> None:
        self.chunks = chunks
        self.fail = fail
        self.calls: list[list[dict[str, Any]]] = []
        self.images: list[bytes | None] = []

    async def stream(
        self, messages: list[dict[str, Any]], image: bytes | None = None
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        self.images.append(image)
        if self.fail:
            raise self.fail
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


async def study_in(repository: InMemoryStudyRepository, storage: InMemoryStorage):
    study = make_study(status=StudyStatus.COMPLETED)
    await repository.save(study)
    await repository.save_prediction(make_prediction(study.id))
    storage.save(b"image", study.file_key)
    return study


async def answer(chat: StudyChat, study_id: str, question: str = "Why?"):
    return [item async for item in chat.ask(study_id, question)]


async def test_the_answer_streams_chunks_then_the_stored_message() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)
    chat = StudyChat(repository, storage, StreamingLLM("It is ", "uncertain."))

    items = await answer(chat, study.id)

    assert items[:2] == ["It is ", "uncertain."]
    reply = items[2]
    assert isinstance(reply, ChatMessage)
    assert (reply.role, reply.content) == ("assistant", "It is uncertain.")


async def test_the_exchange_is_stored_in_order() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)

    await answer(StudyChat(repository, storage, StreamingLLM("ok")), study.id, "Why?")

    stored = await StudyChat(repository, storage, StreamingLLM()).history(study.id)
    assert [(m.role, m.content) for m in stored] == [
        ("user", "Why?"),
        ("assistant", "ok"),
    ]


async def test_a_failed_answer_stores_nothing() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)
    error = DomainError("LLM_UNAVAILABLE", "No key.", status_code=503)

    with pytest.raises(DomainError):
        await answer(StudyChat(repository, storage, StreamingLLM(fail=error)), study.id)

    assert await repository.list_messages(study.id) == []


async def test_the_prompt_carries_the_evidence_the_report_and_the_question() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)
    await repository.save_report(make_report(study.id))
    llm = StreamingLLM("ok")

    await answer(StudyChat(repository, storage, llm), study.id, "Is it a glioma?")

    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert "prediction.label: glioma" in messages[1]["content"]
    assert "Mass in left frontal lobe" in messages[1]["content"]
    assert messages[-1] == {"role": "user", "content": "Is it a glioma?"}


async def test_only_the_last_turns_of_the_history_are_sent() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)
    for minute in range(HISTORY_LIMIT + 5):
        await repository.add_message(make_message(study.id, minutes=minute))
    llm = StreamingLLM("ok")

    await answer(StudyChat(repository, storage, llm), study.id)

    assert len(llm.calls[0]) == 2 + HISTORY_LIMIT + 1


async def test_an_unknown_study_is_404() -> None:
    chat = StudyChat(InMemoryStudyRepository(), InMemoryStorage(), StreamingLLM())

    with pytest.raises(DomainError) as error:
        await chat.history("missing")

    assert error.value.status_code == 404


async def test_a_study_without_a_prediction_is_409() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = make_study(status=StudyStatus.COMPLETED)
    await repository.save(study)

    with pytest.raises(DomainError) as error:
        await answer(StudyChat(repository, storage, StreamingLLM()), study.id)

    assert (error.value.status_code, error.value.code) == (409, "PREDICTION_NOT_FOUND")


async def test_the_study_image_is_sent_to_the_model() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)
    llm = StreamingLLM("ok")

    await answer(StudyChat(repository, storage, llm), study.id)

    assert llm.images == [b"image"]


async def test_the_prompt_frames_image_readings_as_unverified_observations() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)
    llm = StreamingLLM("ok")

    await answer(StudyChat(repository, storage, llm), study.id)

    system = llm.calls[0][0]["content"]
    assert "not a measurement" in system
    assert "never claim to locate a lesion with certainty" in system


async def test_the_prompt_asks_for_an_interpreted_answer_in_the_user_language() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await study_in(repository, storage)
    llm = StreamingLLM("ok")

    await answer(StudyChat(repository, storage, llm), study.id)

    system = llm.calls[0][0]["content"]
    for rule in (
        "match the language",
        "the differential",
        "runner-up labels",
        "do not repeat disclaimers",
    ):
        assert rule in system.lower()


async def test_a_missing_image_is_409() -> None:
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = make_study(status=StudyStatus.COMPLETED)
    await repository.save(study)
    await repository.save_prediction(make_prediction(study.id))

    with pytest.raises(DomainError) as error:
        await answer(StudyChat(repository, storage, StreamingLLM()), study.id)

    assert (error.value.status_code, error.value.code) == (409, "IMAGE_NOT_FOUND")
