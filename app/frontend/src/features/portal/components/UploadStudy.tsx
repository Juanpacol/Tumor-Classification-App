"use client";

import { useRef, useState, type FormEvent } from "react";
import type { ApiClient } from "@/services/api";
import { Alert } from "./Alert";
import { Button } from "./Button";
import styles from "./UploadStudy.module.css";

type UploadState =
  | { status: "idle" }
  | { status: "uploading" }
  | { status: "error"; message: string };

/** Lets a user send an MRI image, with optional clinical notes, into the system. */
export function UploadStudy({
  client,
  onUploaded,
}: {
  client: ApiClient;
  onUploaded: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [notes, setNotes] = useState("");
  const [state, setState] = useState<UploadState>({ status: "idle" });
  const fileInput = useRef<HTMLInputElement>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setState({ status: "uploading" });
    try {
      await client.uploadStudy(file, notes.trim() || undefined);
      setFile(null);
      setNotes("");
      if (fileInput.current) fileInput.current.value = "";
      setState({ status: "idle" });
      onUploaded();
    } catch (error) {
      setState({
        status: "error",
        message:
          error instanceof Error ? error.message : "Something went wrong.",
      });
    }
  }

  return (
    <form className={styles.upload} onSubmit={submit}>
      <label className={styles.file}>
        <span className={styles.label}>MRI image</span>
        <input
          ref={fileInput}
          type="file"
          accept="image/png,image/jpeg"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
      </label>
      <label className={styles.notes}>
        <span className={styles.label}>Clinical notes (optional)</span>
        <input
          type="text"
          value={notes}
          placeholder="e.g. 54F, recurring headaches"
          onChange={(event) => setNotes(event.target.value)}
        />
      </label>
      <Button type="submit" disabled={!file || state.status === "uploading"}>
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
