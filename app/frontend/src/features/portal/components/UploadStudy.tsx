"use client";

import { useRef, useState, type DragEvent, type FormEvent } from "react";
import type { ApiClient } from "@/services/api";
import {
  MAX_NOTES_LENGTH,
  fileProblem,
  uploadErrorMessage,
} from "../uploadChecks";
import { Alert } from "./Alert";
import { Button } from "./Button";
import styles from "./UploadStudy.module.css";

type UploadState =
  | { status: "idle" }
  | { status: "uploading" }
  | { status: "error"; message: string };

/**
 * Lets a user send an MRI image, picked or dropped, with optional clinical
 * notes into the system. The file is checked before it is sent and kept
 * after a failed upload so it can be retried. On success, `onUploaded` gets
 * the new study's id so the caller can open its workspace.
 */
export function UploadStudy({
  client,
  onUploaded,
}: {
  client: ApiClient;
  onUploaded: (studyId: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [dragging, setDragging] = useState(false);
  const [state, setState] = useState<UploadState>({ status: "idle" });
  const fileInput = useRef<HTMLInputElement>(null);

  function choose(picked: File | null) {
    const reason = picked ? fileProblem(picked) : null;
    setProblem(reason);
    setFile(reason ? null : picked);
    setState({ status: "idle" });
  }

  function drop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files[0] ?? null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setState({ status: "uploading" });
    try {
      const study = await client.uploadStudy(file, notes.trim() || undefined);
      setFile(null);
      setNotes("");
      if (fileInput.current) fileInput.current.value = "";
      setState({ status: "idle" });
      onUploaded(study.id);
    } catch (error) {
      setState({ status: "error", message: uploadErrorMessage(error) });
    }
  }

  return (
    <form className={styles.upload} onSubmit={submit}>
      <label
        className={`${styles.file} ${dragging ? styles.dragging : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <span className={styles.label}>MRI image</span>
        <input
          ref={fileInput}
          type="file"
          aria-label="MRI image"
          accept=".jpg,.jpeg,.png,image/png,image/jpeg"
          onChange={(event) => choose(event.target.files?.[0] ?? null)}
        />
        <span className={styles.hint}>
          {file ? file.name : "or drop a JPG or PNG here (max 20 MB)"}
        </span>
        {problem && (
          <span className={styles.problem} role="alert">
            {problem}
          </span>
        )}
      </label>
      <label className={styles.notes}>
        <span className={styles.label}>Clinical notes (optional)</span>
        <textarea
          aria-label="Clinical notes (optional)"
          value={notes}
          rows={2}
          maxLength={MAX_NOTES_LENGTH}
          placeholder="e.g. 54F, recurring headaches"
          onChange={(event) => setNotes(event.target.value)}
        />
        <span className={styles.hint}>
          Do not include names, record numbers or other patient identifiers.{" "}
          {notes.length}/{MAX_NOTES_LENGTH}
        </span>
      </label>
      <Button
        type="submit"
        className={styles.send}
        disabled={!file || state.status === "uploading"}
      >
        {state.status === "uploading" ? "Uploading…" : "Upload study"}
      </Button>
      {state.status === "error" && (
        <Alert tone="warning" title="The study could not be uploaded">
          <p>{state.message}</p>
        </Alert>
      )}
    </form>
  );
}
