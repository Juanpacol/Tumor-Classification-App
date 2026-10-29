import { Alert } from "@/features/portal/components";
import type { StudyDetail } from "@/services/api";
import { PredictionBadge } from "./PredictionBadge";
import { ProbabilityBars } from "./ProbabilityBars";
import { QualityMetrics } from "./QualityMetrics";
import { UncertaintyAlert } from "./UncertaintyAlert";
import styles from "./ClinicalSummary.module.css";

/** What the model concluded about a study, or why there is no conclusion. */
export function ClinicalSummary({ study }: { study: StudyDetail }) {
  const { prediction } = study;

  if (study.status === "rejected") {
    return (
      <Alert tone="warning" title="Image rejected">
        {study.rejection_reason ?? "The image did not pass the quality check."}
      </Alert>
    );
  }
  if (study.status === "failed") {
    return (
      <Alert tone="warning" title="The analysis failed">
        The study could not be analyzed. Try uploading the image again.
      </Alert>
    );
  }
  if (!prediction) {
    return (
      <Alert>
        {study.status === "completed"
          ? "No prediction is available for this study."
          : "The analysis is still running."}
      </Alert>
    );
  }

  return (
    <div className={styles.summary}>
      {prediction.needs_expert_triage && (
        <UncertaintyAlert confidence={prediction.confidence} />
      )}
      <PredictionBadge prediction={prediction} />
      <ProbabilityBars probabilities={prediction.probabilities} />
      <QualityMetrics prediction={prediction} />
    </div>
  );
}
