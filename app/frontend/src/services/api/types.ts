/**
 * Shapes the portal expects from the backend. They mirror the backend domain
 * models and `ml_engine` contracts; field names are confirmed against the
 * endpoints as #118, #119, #122 and #123 land.
 */

export type StudyStatus =
  | "pending"
  | "processing"
  | "completed"
  | "rejected"
  | "failed";

export type TumorLabel = "glioma" | "meningioma" | "pituitary" | "no_tumor";

export type AiAgreement = "agrees" | "disagrees" | "uncertain";

export interface QualityMetrics {
  snr: number;
  entropy: number;
  skull_stripped: boolean;
}

export interface FeatureContribution {
  name: string;
  value: number;
  z_score: number;
}

export interface Prediction {
  label: TumorLabel;
  probabilities: Record<TumorLabel, number>;
  confidence: number;
  needs_expert_triage: boolean;
  quality: QualityMetrics;
  top_features: FeatureContribution[];
  model_version: string;
}

export interface ClinicalReport {
  findings: string;
  visual_observations: string;
  impression: string;
  differential_considerations: string;
  suggested_next_steps: string[];
  ai_agreement: AiAgreement;
}

export type ReportStatus = "pending" | "completed" | "failed";

export interface Report {
  status: ReportStatus;
  model: string;
  created_at: string;
  content: ClinicalReport | null;
  error: string | null;
}

export interface Study {
  id: string;
  status: StudyStatus;
  created_at: string;
  clinical_notes: string | null;
  rejection_reason: string | null;
}

/** A study with the results the dashboard and the workspace show. */
export interface StudyDetail extends Study {
  prediction: Prediction | null;
  report: Report | null;
}

export interface Page<T> {
  items: T[];
  page: number;
  page_size: number;
  total: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

/** One server-sent event of a streamed chat answer. */
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "done"; message: ChatMessage }
  | { type: "error"; code: string; message: string };

export interface ListStudiesParams {
  page?: number;
  pageSize?: number;
}
