"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { StudyViewer } from "@/features/dicom_viewer/StudyViewer";
import {
  createApiClient,
  type ApiClient,
  type StudyDetail,
} from "@/services/api";
import { formatDate } from "../labels";
import { Alert } from "./Alert";
import { AnalysisPanel } from "./AnalysisPanel";
import { WorkspaceBackdrop } from "./WorkspaceBackdrop";
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
    <div className={styles.stage}>
      <WorkspaceBackdrop />
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
          <StudyViewer
            imageUrl={api.imageUrl(state.study.id)}
            caption={formatDate(state.study.created_at)}
            flagged={state.study.prediction?.needs_expert_triage === true}
          />
          <AnalysisPanel client={api} study={state.study} />
        </div>
      )}
    </div>
  );
}
