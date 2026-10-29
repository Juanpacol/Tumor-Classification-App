"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { ApiError, type ApiClient } from "@/services/api";
import {
  chatReducer,
  initialChatState,
  isSendable,
  type ChatAction,
} from "./chatReducer";

const INTERRUPTED =
  "The connection was interrupted before the answer finished.";
const FAILED = "The assistant could not answer.";

function describe(error: unknown): { message: string; unavailable: boolean } {
  if (error instanceof ApiError) {
    return { message: error.message, unavailable: error.status === 503 };
  }
  return { message: FAILED, unavailable: false };
}

/**
 * The conversation of a study. The answer is read as a stream; leaving the
 * page or starting another stream aborts the one in progress, and nothing
 * from an aborted stream reaches the state.
 */
export function useChat(client: ApiClient, studyId: string) {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const controller = useRef<AbortController | null>(null);
  const counter = useRef(0);
  const busy = useRef(false);

  useEffect(() => {
    let current = true;
    client
      .listChatMessages(studyId)
      .then(
        (messages) => current && dispatch({ type: "historyLoaded", messages }),
      )
      .catch((error: unknown) => {
        if (!current) return;
        const { message, unavailable } = describe(error);
        dispatch({ type: "historyFailed", message, unavailable });
      });
    return () => {
      current = false;
    };
  }, [client, studyId]);

  useEffect(() => () => controller.current?.abort(), []);

  const stream = useCallback(
    async (sendId: number, text: string) => {
      const abort = new AbortController();
      controller.current = abort;
      const emit = (action: ChatAction) => {
        if (!abort.signal.aborted) dispatch(action);
      };
      try {
        // A done or error event ends the answer; leaving the loop closes the
        // stream even if the server keeps the connection open.
        let finished = false;
        for await (const event of client.streamChat(
          studyId,
          text,
          abort.signal,
        )) {
          if (event.type === "token") {
            emit({ type: "token", sendId, text: event.text });
          } else if (event.type === "done") {
            finished = true;
            emit({ type: "done", sendId, message: event.message });
            break;
          } else {
            finished = true;
            emit({
              type: "fail",
              sendId,
              message: event.message,
              unavailable: false,
            });
            break;
          }
        }
        if (!finished) {
          emit({
            type: "fail",
            sendId,
            message: INTERRUPTED,
            unavailable: false,
          });
        }
      } catch (error) {
        emit({ type: "fail", sendId, ...describe(error) });
      } finally {
        if (controller.current === abort) busy.current = false;
      }
    },
    [client, studyId],
  );

  const send = useCallback(
    (text: string) => {
      if (busy.current || state.phase !== "ready" || !isSendable(text)) return;
      busy.current = true;
      counter.current += 1;
      dispatch({ type: "send", sendId: counter.current, text });
      void stream(counter.current, text.trim());
    },
    [state.phase, stream],
  );

  const retry = useCallback(
    (itemId: string) => {
      const failed = state.items.find((item) => item.id === itemId);
      if (busy.current || failed?.status !== "failed" || !failed.prompt) {
        return;
      }
      busy.current = true;
      counter.current += 1;
      dispatch({ type: "retry", sendId: counter.current, itemId });
      void stream(counter.current, failed.prompt);
    },
    [state.items, stream],
  );

  return { state, send, retry };
}
