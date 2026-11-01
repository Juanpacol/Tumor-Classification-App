"""Use case: a radiologist chats with the assistant about one study."""

import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

from application.chat_prompt import build_messages
from application.evidence_bundle import BuildEvidenceBundle
from domain.errors import DomainError
from domain.models import ChatMessage, Prediction, ReportStatus, Study, StudyStatus
from ports.llm_port import LLMPort
from ports.storage_port import StoragePort
from ports.study_repository_port import StudyRepositoryPort


class StudyChat:
    """Stored conversation of a study and streamed answers to new questions."""

    def __init__(
        self,
        repository: StudyRepositoryPort,
        storage: StoragePort,
        llm: LLMPort,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._llm = llm
        self._clock = clock

    async def history(self, study_id: str) -> list[ChatMessage]:
        """Messages of the study in order. Raises 404 for an unknown study."""
        await self._require_study(study_id)
        return await self._repository.list_messages(study_id)

    async def ask(
        self, study_id: str, question: str
    ) -> AsyncIterator[str | ChatMessage]:
        """Yields the answer's text chunks, then the stored assistant message.

        Raises ``DomainError`` 404 for an unknown study and 409 for one without a
        prediction or image. LLM errors surface on the first chunk. The exchange is stored
        only when the answer completes.
        """
        study = await self._require_study(study_id)
        prediction = await self._require_prediction(study_id)
        report = await self._repository.get_report(study_id)
        content = (
            report.content
            if report and report.status is ReportStatus.COMPLETED
            else None
        )
        history = await self._repository.list_messages(study_id)
        image = self._load_image(study)
        messages = build_messages(
            study,
            prediction,
            BuildEvidenceBundle().execute(prediction),
            content,
            history,
            question,
        )
        answer: list[str] = []
        async for chunk in self._llm.stream(messages, image=image):
            answer.append(chunk)
            yield chunk
        await self._repository.add_message(self._message(study_id, "user", question))
        reply = self._message(study_id, "assistant", "".join(answer))
        await self._repository.add_message(reply)
        yield reply

    def _load_image(self, study: Study) -> bytes:
        try:
            return self._storage.load(study.file_key)
        except KeyError:
            raise DomainError(
                "IMAGE_NOT_FOUND", "The study image is missing.", status_code=409
            ) from None

    def _message(self, study_id: str, role: str, content: str) -> ChatMessage:
        return ChatMessage(
            id=str(uuid.uuid4()),
            study_id=study_id,
            role=role,
            content=content,
            created_at=self._clock(),
        )

    async def _require_study(self, study_id: str) -> Study:
        study = await self._repository.get(study_id)
        if study is None:
            raise DomainError("STUDY_NOT_FOUND", "Study not found.", status_code=404)
        if study.status is not StudyStatus.COMPLETED:
            raise DomainError(
                "STUDY_NOT_COMPLETED",
                f"A chat needs a completed study, this one is {study.status.value}.",
                status_code=409,
            )
        return study

    async def _require_prediction(self, study_id: str) -> Prediction:
        prediction = await self._repository.get_prediction(study_id)
        if prediction is None:
            raise DomainError(
                "PREDICTION_NOT_FOUND",
                "The study has no prediction to chat about.",
                status_code=409,
            )
        return prediction
