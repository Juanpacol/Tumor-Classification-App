"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { flushSync } from "react-dom";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
} from "react";
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
import { useCountUp } from "../useCountUp";
import {
  byOrder,
  countByFilter,
  needsReview,
  rowDetail,
  matches,
  matchesQuery,
  type WorklistFilter,
  type WorklistOrder,
} from "../worklist";
import { Alert } from "./Alert";
import { Button } from "./Button";
import { StatusTag } from "./StatusTag";
import styles from "./StudyDashboard.module.css";

function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Runs a state change inside a view transition, so rows that stay glide to
 * their new place and the others fade. Browsers without the API just update.
 */
function withTransition(update: () => void) {
  const doc = document as Document & {
    startViewTransition?: (callback: () => void) => unknown;
  };
  if (!doc.startViewTransition || prefersReducedMotion()) {
    update();
    return;
  }
  doc.startViewTransition(() => flushSync(update));
}

const OPENING_MS = 280;

/** Studies ordered for triage, one page at a time. */
export function StudyDashboard({ client }: { client?: ApiClient }) {
  const defaultClient = useMemo(() => createApiClient(), []);
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<WorklistFilter>("all");
  const [order, setOrder] = useState<WorklistOrder>("priority");
  const [query, setQuery] = useState("");
  const [opening, setOpening] = useState<string | null>(null);
  const [entry, setEntry] = useState(true);
  const router = useRouter();
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    if ("startViewTransition" in document) {
      const done = window.setTimeout(() => setEntry(false), 900);
      return () => window.clearTimeout(done);
    }
  }, []);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  function open(event: MouseEvent<HTMLAnchorElement>, id: string) {
    const modified =
      event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
    if (modified || event.button !== 0 || prefersReducedMotion()) return;
    event.preventDefault();
    setOpening(id);
    timer.current = window.setTimeout(
      () => router.push(`/studies/${id}`),
      OPENING_MS,
    );
  }
  const { state, retry } = useStudies(client ?? defaultClient, page);
  const studies = useMemo(
    () => (state.status === "ready" ? state.data.items : []),
    [state],
  );
  const counts = useMemo(() => countByFilter(studies), [studies]);
  const visible = useMemo(
    () =>
      byOrder(studies, order).filter(
        (study) => matches(study, filter) && matchesQuery(study, query),
      ),
    [studies, filter, order, query],
  );

  return (
    <div>
      <header className={styles.hero}>
        <span className={styles.wordmark}>STUDIES</span>
        <h1 className={styles.headline}>What needs a reader, first.</h1>
        <p className={styles.tagline}>Ordered by urgency, so nothing waits.</p>
      </header>
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
          <Summary
            counts={counts}
            filter={filter}
            onSelect={(next) => withTransition(() => setFilter(next))}
          />
          <div className={styles.toolbar}>
            <label className={styles.search}>
              <span className={styles.srOnly}>Search studies</span>
              <input
                type="search"
                value={query}
                placeholder="Search by class, status or date"
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
            <div
              className={styles.order}
              role="group"
              aria-label="Sort studies"
            >
              {ORDERS.map(({ key, label }) => (
                <button
                  key={key}
                  type="button"
                  aria-pressed={order === key}
                  className={styles.chip}
                  onClick={() => withTransition(() => setOrder(key))}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          {visible.length === 0 && (
            <p className={styles.message}>No studies match this filter.</p>
          )}
          <ul
            key={entry ? `${filter}-${order}` : undefined}
            className={styles.list}
            data-opening={opening ?? undefined}
          >
            {visible.map((study, index) => (
              <StudyRow
                key={study.id}
                study={study}
                index={index}
                enter={entry}
                opening={opening === study.id}
                onOpen={open}
              />
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

const ORDERS: { key: WorklistOrder; label: string }[] = [
  { key: "priority", label: "Priority" },
  { key: "newest", label: "Newest" },
];

const KPIS = [
  { key: "all", label: "Studies", tone: "plain" },
  { key: "review", label: "Need expert review", tone: "review" },
  { key: "open", label: "In progress", tone: "open" },
  { key: "problem", label: "Rejected or failed", tone: "problem" },
] as const;

function Summary({
  counts,
  filter,
  onSelect,
}: {
  counts: Record<WorklistFilter, number>;
  filter: WorklistFilter;
  onSelect: (filter: WorklistFilter) => void;
}) {
  const shown = {
    all: useCountUp(counts.all),
    review: useCountUp(counts.review),
    open: useCountUp(counts.open),
    problem: useCountUp(counts.problem),
  };
  return (
    <div className={styles.summary} role="group" aria-label="Summary">
      {KPIS.map(({ key, label, tone }, index) => (
        <div
          key={key}
          className={styles.cardWrap}
          style={{ "--i": index } as CSSProperties}
        >
          <button
            type="button"
            aria-pressed={filter === key}
            className={`${styles.card} ${styles[tone]}`}
            onClick={() => onSelect(filter === key ? "all" : key)}
          >
            <span className={styles.value}>{shown[key]}</span>
            <span className={styles.cardLabel}>{label}</span>
            {key === "all" ? (
              <Mix counts={counts} />
            ) : (
              <Share value={counts[key]} total={counts.all} />
            )}
          </button>
        </div>
      ))}
    </div>
  );
}

const MIX_SEGMENTS = ["ok", "review", "open", "problem"] as const;

/** How the studies split: reviewed, needing review, in progress, rejected or failed. */
function Mix({ counts }: { counts: Record<WorklistFilter, number> }) {
  const values = {
    ok: counts.completed - counts.review,
    review: counts.review,
    open: counts.open,
    problem: counts.problem,
  };
  return (
    <span className={styles.mix} aria-hidden="true">
      {MIX_SEGMENTS.map((segment, index) => (
        <span
          key={segment}
          className={styles[`mix_${segment}`]}
          style={{ flexGrow: values[segment], "--i": index } as CSSProperties}
        />
      ))}
    </span>
  );
}

const STATUS_ICONS = {
  pending: "…",
  processing: "…",
  completed: "✓",
  rejected: "✕",
  failed: "!",
} as const;

function rowTone(study: StudyDetail): "review" | "open" | "problem" | "ok" {
  if (needsReview(study)) return "review";
  if (study.status === "rejected" || study.status === "failed")
    return "problem";
  if (study.status === "pending" || study.status === "processing")
    return "open";
  return "ok";
}

/** The share of all studies a category holds, in the color of its card. */
function Share({ value, total }: { value: number; total: number }) {
  const percent = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <span className={styles.share} aria-hidden="true">
      <span style={{ width: `${percent}%` }} />
    </span>
  );
}

function StudyRow({
  study,
  index,
  enter,
  opening,
  onOpen,
}: {
  study: StudyDetail;
  index: number;
  enter: boolean;
  opening: boolean;
  onOpen: (event: MouseEvent<HTMLAnchorElement>, id: string) => void;
}) {
  const prediction = study.prediction;
  const tone = rowTone(study);
  const percent = prediction ? Math.round(prediction.confidence * 100) : 0;
  return (
    <li
      style={
        {
          "--i": index,
          viewTransitionName: `study-${study.id}`,
        } as CSSProperties
      }
      className={`${styles.itemWrap} ${enter ? styles.enter : ""} ${
        opening ? styles.opening : ""
      }`}
    >
      <Link
        href={`/studies/${study.id}`}
        onClick={(event) => onOpen(event, study.id)}
        className={`${styles.row} ${styles[`row_${tone}`]}`}
      >
        <span className={styles.main}>
          <span className={styles.name}>
            <span className={styles.label}>Predicted class: </span>
            {prediction ? TUMOR_LABELS[prediction.label] : "No prediction"}
          </span>
          <span className={styles.meta}>
            <span className={styles.date}>{formatDate(study.created_at)}</span>
            <span className={styles.detail}>{rowDetail(study)}</span>
          </span>
        </span>
        <span className={styles.confidence}>
          <span className={styles.label}>Confidence: </span>
          {prediction ? (
            <>
              <span className={styles.track} aria-hidden="true">
                <span
                  className={styles.fill}
                  style={{ width: `${percent}%` }}
                />
              </span>
              <span className={styles.percent}>
                {formatConfidence(prediction.confidence)}
              </span>
            </>
          ) : (
            <span className={styles.percent}>—</span>
          )}
        </span>
        <span className={styles.state}>
          <StatusTag
            icon={STATUS_ICONS[study.status]}
            tone={
              tone === "problem" ? "problem" : tone === "open" ? "open" : "ok"
            }
          >
            {STATUS_LABELS[study.status]}
          </StatusTag>
          {tone === "review" && (
            <StatusTag icon="▲" tone="attention">
              Needs expert review
            </StatusTag>
          )}
        </span>
        <span className={styles.arrow} aria-hidden="true">
          →
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
      <button
        type="button"
        className={styles.pageButton}
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
      >
        ← Previous
      </button>
      <span className={styles.pageLabel}>
        Page {page} of {pages}
      </span>
      <button
        type="button"
        className={styles.pageButton}
        disabled={page >= pages}
        onClick={() => onPage(page + 1)}
      >
        Next →
      </button>
    </nav>
  );
}
