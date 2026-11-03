import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from application.study_chat import StudyChat
from domain.errors import DomainError
from domain.models import ChatMessage


class ChatMessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatEventResponse(BaseModel):
    """One server-sent event of a streamed answer.

    `token` carries `text`, `done` carries the stored `message` and `error`
    carries a `code` and a `message` string.
    """

    type: Literal["token", "done", "error"]
    text: str | None = None
    message: ChatMessageResponse | str | None = None
    code: str | None = None


def _message_response(message: ChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        content=message.content,
        created_at=message.created_at.isoformat(),
    )


def _event(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _events(
    first: str | ChatMessage, rest: AsyncIterator[str | ChatMessage]
) -> AsyncIterator[str]:
    try:
        yield _item_event(first)
        async for item in rest:
            yield _item_event(item)
    except DomainError as error:
        yield _event({"type": "error", "code": error.code, "message": error.message})


def _item_event(item: str | ChatMessage) -> str:
    if isinstance(item, str):
        return _event({"type": "token", "text": item})
    return _event({"type": "done", "message": _message_response(item).model_dump()})


def create_chat_router(study_chat: StudyChat) -> APIRouter:
    router = APIRouter(prefix="/api/studies", tags=["Chat"])

    @router.get(
        "/{study_id}/chat",
        response_model=list[ChatMessageResponse],
        summary="List a study's chat messages",
        description="Returns the conversation about a study, oldest message first.",
        responses={
            404: {"description": "The study does not exist."},
            409: {"description": "The study is not completed."},
        },
    )
    async def list_endpoint(study_id: str) -> list[ChatMessageResponse]:
        return [_message_response(m) for m in await study_chat.history(study_id)]

    @router.post(
        "/{study_id}/chat",
        response_model=ChatEventResponse,
        summary="Ask the assistant about a study",
        description=(
            "Streams the answer as server-sent events: `token` events with text "
            "chunks, then one `done` event with the stored message. A failure "
            "after the stream started arrives as an `error` event. The exchange is "
            "stored only when the answer completes."
        ),
        responses={
            200: {"content": {"text/event-stream": {}}},
            404: {"description": "The study does not exist."},
            409: {"description": "The study is not completed or has no prediction."},
            503: {"description": "The assistant is not available."},
        },
    )
    async def ask_endpoint(study_id: str, request: ChatRequest) -> StreamingResponse:
        stream = study_chat.ask(study_id, request.message)
        first = await anext(stream)
        return StreamingResponse(_events(first, stream), media_type="text/event-stream")

    return router
