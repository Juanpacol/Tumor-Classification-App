"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import {
  createApiClient,
  type ApiClient,
  type StudyDetail,
} from "@/services/api";
import {
  formatConfidence,
  formatDate,
  STATUS_LABELS,
  TUMOR_LABELS,
} from "../labels";
import { PAGE_SIZE, useStudies } from "../useStudies";
import { Alert } from "./Alert";
import { Button } from "./Button";
import { StatusTag } from "./StatusTag";
import styles from "./StudyDashboard.module.css";

/** Newest-first list of studies for triage, one page at a time. */
export function StudyDashboard({ client }: { client?: ApiClient }) {
  const defaultClient = useMemo(() => createApiClient(), []);
  const [page, setPage] = useState(1);
  const { state, retry } = useStudies(client ?? defaultClient, page);

  return (
    <div>
      <h1 className={styles.title}>Studies</h1>
      {state.status === "loading" && (
        <p className={styles.message} role="status">
          Loading studies…
        </p>
      )}
      {state.status === "error" && (
        <Alert tone="warning" title="The studies could not be loaded">
          <p>{state.message}</p>
          <Button variant="quiet" onClick={retry}>
            Try again
          </Button>
        </Alert>
      )}
      {state.status === "ready" && state.data.items.length === 0 && (
        <p className={styles.message}>No studies yet.</p>
      )}
      {state.status === "ready" && state.data.items.length > 0 && (
        <>
          <ul className={styles.list}>
            <li className={styles.head} aria-hidden="true">
              <span>Date</span>
              <span>Status</span>
              <span>Predicted class</span>
              <span>Confidence</span>
              <span>Triage</span>
            </li>
            {state.data.items.map((study) => (
              <StudyRow key={study.id} study={study} />
            ))}
          </ul>
          <Pagination
            page={state.data.page}
            total={state.data.total}
            onPage={setPage}
          />
        </>
      )}
    </div>
  );
}

function StudyRow({ study }: { study: StudyDetail }) {
  const prediction = study.prediction;
  const flagged = prediction?.needs_expert_triage === true;
  return (
    <li>
      <Link
        href={`/studies/${study.id}`}
        className={`${styles.row} ${flagged ? styles.flagged : ""}`}
      >
        <span>{formatDate(study.created_at)}</span>
        <span>
          <StatusTag
            tone={study.status === "completed" ? "neutral" : "attention"}
          >
            {STATUS_LABELS[study.status]}
          </StatusTag>
        </span>
        <span>
          <span className={styles.label}>Predicted class: </span>
          {prediction ? TUMOR_LABELS[prediction.label] : "—"}
        </span>
        <span>
          <span className={styles.label}>Confidence: </span>
          {prediction ? formatConfidence(prediction.confidence) : "—"}
        </span>
        <span>
          {flagged && (
            <StatusTag tone="attention">Needs expert review</StatusTag>
          )}
        </span>
      </Link>
    </li>
  );
}

function Pagination({
  page,
  total,
  onPage,
}: {
  page: number;
  total: number;
  onPage: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  return (
    <nav className={styles.pagination} aria-label="Pagination">
      <Button
        variant="quiet"
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
      >
        Previous
      </Button>
      <span>
        Page {page} of {pages}
      </span>
      <Button
        variant="quiet"
        disabled={page >= pages}
        onClick={() => onPage(page + 1)}
      >
        Next
      </Button>
    </nav>
  );
}
