import { formatConfidence } from "@/features/portal/labels";
import type { StudyDetail } from "@/services/api";

/** `note` is a standing rule rather than a result, so it never reads as done. */
export type CheckState = "pass" | "attention" | "pending" | "note";

export interface SafetyCheck {
  key: string;
  label: string;
  detail: string;
  state: CheckState;
}

function quality(study: StudyDetail): SafetyCheck {
  const key = "quality";
  if (study.prediction) {
    const { snr, entropy } = study.prediction.quality;
    return {
      key,
      label: "Image quality check passed",
      detail: `Signal-to-noise ${snr.toFixed(1)} · entropy ${entropy.toFixed(2)}`,
      state: "pass",
    };
  }
  if (study.status === "rejected") {
    return {
      key,
      label: "Image quality check failed",
      detail: study.rejection_reason ?? "The image did not pass the check.",
      state: "attention",
    };
  }
  if (study.status === "failed") {
    return {
      key,
      label: "Image quality check",
      detail: "The analysis stopped before it could finish.",
      state: "attention",
    };
  }
  return {
    key,
    label: "Image quality check",
    detail: "The analysis is still running.",
    state: "pending",
  };
}

function confidence(study: StudyDetail): SafetyCheck | null {
  const { prediction } = study;
  if (!prediction) return null;
  const percent = formatConfidence(prediction.confidence);
  return prediction.needs_expert_triage
    ? {
        key: "confidence",
        label: "Low confidence flagged",
        detail: `${percent} confidence: the model did not force a class.`,
        state: "attention",
      }
    : {
        key: "confidence",
        label: "Confidence not flagged",
        detail: `${percent} confidence.`,
        state: "pass",
      };
}

function review(study: StudyDetail): SafetyCheck {
  const flagged = study.prediction?.needs_expert_triage === true;
  return {
    key: "review",
    label: flagged ? "Expert review required" : "Radiologist review",
    detail: flagged
      ? "A specialist must read this study before anyone relies on it."
      : "A qualified radiologist must review every result.",
    state: flagged ? "attention" : "note",
  };
}

const AGREEMENT = {
  agrees: {
    label: "AI assistant agrees with the classifier",
    state: "pass",
  },
  disagrees: {
    label: "AI assistant disagrees with the classifier",
    state: "attention",
  },
  uncertain: {
    label: "AI assistant could not confirm the classifier",
    state: "attention",
  },
} as const;

function report(study: StudyDetail): SafetyCheck[] {
  const content = study.report?.content;
  if (content) {
    const agreement = AGREEMENT[content.ai_agreement];
    return [
      {
        key: "agreement",
        label: agreement.label,
        detail:
          "The assistant read the image with the classifier's output in hand.",
        state: agreement.state,
      },
      {
        key: "wording",
        label: "Report wording checked",
        detail:
          "Checked against rules for diagnosis, treatment and expert-review wording.",
        state: "pass",
      },
    ];
  }
  if (study.status === "completed") {
    return [
      {
        key: "agreement",
        label: "AI report agreement",
        detail: "Waiting for the report.",
        state: "pending",
      },
    ];
  }
  return [];
}

/** The safeguards that apply to one study, each with its real state. */
export function safetyChecks(study: StudyDetail): SafetyCheck[] {
  return [
    quality(study),
    confidence(study),
    review(study),
    ...report(study),
  ].filter((check): check is SafetyCheck => check !== null);
}
