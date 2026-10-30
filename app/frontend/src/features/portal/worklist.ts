import type { StudyDetail } from "@/services/api";
import { STATUS_LABELS, TUMOR_LABELS } from "./labels";

export type WorklistFilter =
  | "all"
  | "review"
  | "completed"
  | "open"
  | "problem";

export const FILTERS: { key: WorklistFilter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "review", label: "Needs expert review" },
  { key: "completed", label: "Completed" },
  { key: "open", label: "In progress" },
  { key: "problem", label: "Rejected or failed" },
];

export function needsReview(study: StudyDetail): boolean {
  return study.prediction?.needs_expert_triage === true;
}

const MATCHES: Record<WorklistFilter, (study: StudyDetail) => boolean> = {
  all: () => true,
  review: needsReview,
  completed: (study) => study.status === "completed",
  open: (study) => study.status === "pending" || study.status === "processing",
  problem: (study) => study.status === "rejected" || study.status === "failed",
};

export function matches(study: StudyDetail, filter: WorklistFilter): boolean {
  return MATCHES[filter](study);
}

export function countByFilter(
  studies: StudyDetail[],
): Record<WorklistFilter, number> {
  const counts = {} as Record<WorklistFilter, number>;
  for (const { key } of FILTERS) {
    counts[key] = studies.filter((study) => matches(study, key)).length;
  }
  return counts;
}

/** Lowest first: what a reader should look at before anything else. */
function rank(study: StudyDetail): number {
  if (needsReview(study)) return 0;
  if (study.status === "rejected" || study.status === "failed") return 1;
  if (study.status === "pending" || study.status === "processing") return 2;
  return 3;
}

/** Studies that need a person first, the least confident among them on top; the rest newest first. */
export function byPriority(studies: StudyDetail[]): StudyDetail[] {
  return [...studies].sort((a, b) => {
    const byRank = rank(a) - rank(b);
    if (byRank !== 0) return byRank;
    if (needsReview(a) && needsReview(b)) {
      return (a.prediction?.confidence ?? 0) - (b.prediction?.confidence ?? 0);
    }
    return b.created_at.localeCompare(a.created_at);
  });
}

export type WorklistOrder = "priority" | "newest";

export function byOrder(
  studies: StudyDetail[],
  order: WorklistOrder,
): StudyDetail[] {
  if (order === "priority") return byPriority(studies);
  return [...studies].sort((a, b) => b.created_at.localeCompare(a.created_at));
}

/** True when the query appears in the class, the status or the date of the study. */
export function matchesQuery(study: StudyDetail, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  const haystack = [
    study.prediction ? TUMOR_LABELS[study.prediction.label] : "",
    STATUS_LABELS[study.status],
    needsReview(study) ? "needs expert review" : "",
    study.created_at,
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(needle);
}

/** What a reader would otherwise open the study to learn, in one line. */
export function rowDetail(study: StudyDetail): string {
  if (study.prediction) {
    return (
      Object.entries(study.prediction.probabilities) as [
        keyof typeof TUMOR_LABELS,
        number,
      ][]
    )
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(
        ([label, probability]) =>
          `${TUMOR_LABELS[label]} ${Math.round(probability * 100)}%`,
      )
      .join(" · ");
  }
  if (study.status === "rejected") {
    return study.rejection_reason ?? "Rejected by the quality check.";
  }
  if (study.status === "failed") {
    return "The analysis did not finish. Try uploading again.";
  }
  return "The analysis is still running.";
}
