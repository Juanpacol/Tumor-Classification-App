"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { StudyViewer } from "@/features/dicom_viewer/StudyViewer";
import {
  createApiClient,
  type ApiClient,
  type StudyDetail,
} from "@/services/api";
import { Alert } from "./Alert";
import { Card } from "./Card";
import styles from "./StudyWorkspace.module.css";

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; study: StudyDetail };

/** One study: the viewer beside the areas for the summary, the report and the chat. */
export function StudyWorkspace({
  id,
  client,
}: {
  id: string;
  client?: ApiClient;
}) {
  const defaultClient = useMemo(() => createApiClient(), []);
  const api = client ?? defaultClient;
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let current = true;
    setState({ status: "loading" });
    api
      .getStudy(id)
      .then((study) => current && setState({ status: "ready", study }))
      .catch(
        (error: unknown) =>
          current &&
          setState({
            status: "error",
            message:
              error instanceof Error ? error.message : "Something went wrong.",
          }),
      );
    return () => {
      current = false;
    };
  }, [api, id]);

  return (
    <div>
      <Link href="/studies" className={styles.back}>
        ← All studies
      </Link>
      {state.status === "loading" && (
        <p className={styles.message} role="status">
          Loading study…
        </p>
      )}
      {state.status === "error" && (
        <Alert tone="warning" title="The study could not be loaded">
          {state.message}
        </Alert>
      )}
      {state.status === "ready" && (
        <div className={styles.grid}>
          <StudyViewer imageUrl={api.imageUrl(state.study.id)} />
          <div className={styles.side}>
            <Card aria-label="Clinical summary">
              <h2 className={styles.heading}>Clinical summary</h2>
              <p className={styles.message}>Not available yet.</p>
            </Card>
            <Card aria-label="AI clinical report">
              <h2 className={styles.heading}>AI clinical report</h2>
              <p className={styles.message}>Not available yet.</p>
            </Card>
            <Card aria-label="Chat">
              <h2 className={styles.heading}>Chat</h2>
              <p className={styles.message}>Not available yet.</p>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
