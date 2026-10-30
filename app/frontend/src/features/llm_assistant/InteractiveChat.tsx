"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { Alert, Button } from "@/features/portal/components";
import type { ApiClient, StudyDetail } from "@/services/api";
import { isSendable, MAX_MESSAGE_LENGTH, type ChatItem } from "./chatReducer";
import { useChat } from "./useChat";
import styles from "./InteractiveChat.module.css";

const NEAR_BOTTOM_PX = 80;

/** Chat about a study with the AI assistant; hidden until the study is completed. */
export function InteractiveChat({
  client,
  study,
}: {
  client: ApiClient;
  study: StudyDetail;
}) {
  if (study.status !== "completed") return null;
  return <Chat client={client} studyId={study.id} />;
}

function Chat({ client, studyId }: { client: ApiClient; studyId: string }) {
  const { state, send, retry } = useChat(client, studyId);
  const [draft, setDraft] = useState("");
  const log = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const streaming = state.activeSend !== null;
  const disabled = state.phase === "disabled";

  useEffect(() => {
    const element = log.current;
    if (element && stick.current) element.scrollTop = element.scrollHeight;
  }, [state.items]);

  function trackScroll() {
    const element = log.current;
    if (!element) return;
    stick.current =
      element.scrollHeight - element.scrollTop - element.clientHeight <
      NEAR_BOTTOM_PX;
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!isSendable(draft) || streaming || state.phase !== "ready") return;
    stick.current = true;
    send(draft);
    setDraft("");
  }

  return (
    <section aria-label="Chat">
      {state.phase === "loading" && (
        <p className={styles.message} role="status">
          Loading the conversation…
        </p>
      )}
      {state.notice && (
        <Alert
          tone={disabled ? "neutral" : "warning"}
          title={
            disabled
              ? "The chat is not available right now"
              : "The conversation could not be loaded"
          }
        >
          {state.notice}
        </Alert>
      )}
      <div
        ref={log}
        role="log"
        aria-live="polite"
        aria-label="Conversation"
        className={styles.log}
        onScroll={trackScroll}
      >
        {state.items.map((item) => (
          <Message key={item.id} item={item} onRetry={retry} />
        ))}
      </div>
      <form className={styles.form} onSubmit={submit}>
        <label htmlFor="chat-input" className={styles.label}>
          Ask about this study
        </label>
        <textarea
          id="chat-input"
          className={styles.input}
          value={draft}
          maxLength={MAX_MESSAGE_LENGTH}
          rows={3}
          disabled={disabled || state.phase === "loading"}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <div className={styles.footer}>
          <span className={styles.counter}>
            {draft.length}/{MAX_MESSAGE_LENGTH}
          </span>
          <Button
            type="submit"
            disabled={
              !isSendable(draft) || streaming || state.phase !== "ready"
            }
          >
            Send
          </Button>
        </div>
      </form>
      <p className={styles.note}>
        Answers come from an AI assistant and are not a diagnosis.
      </p>
    </section>
  );
}

function Message({
  item,
  onRetry,
}: {
  item: ChatItem;
  onRetry: (id: string) => void;
}) {
  return (
    <div className={`${styles.message_} ${styles[item.role]}`}>
      <span className={styles.author}>
        {item.role === "user" ? "You" : "Assistant"}
      </span>
      {item.status === "failed" ? (
        <div role="alert" className={styles.failed}>
          <p>{item.error}</p>
          <Button variant="quiet" onClick={() => onRetry(item.id)}>
            Retry
          </Button>
        </div>
      ) : (
        <p className={styles.content}>
          {item.content}
          {item.status === "streaming" && (
            <span className={styles.cursor} aria-label="Answer in progress">
              ▍
            </span>
          )}
        </p>
      )}
    </div>
  );
}
