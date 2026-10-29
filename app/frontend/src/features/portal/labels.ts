import type { StudyStatus, TumorLabel } from "@/services/api";

export const TUMOR_LABELS: Record<TumorLabel, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  pituitary: "Pituitary",
  no_tumor: "No tumor",
};

export const STATUS_LABELS: Record<StudyStatus, string> = {
  pending: "Pending",
  processing: "Processing",
  completed: "Completed",
  rejected: "Rejected",
  failed: "Failed",
};

export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

export function formatDate(iso: string): string {
  const text = new Date(iso).toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  });
  return `${text} UTC`;
}
