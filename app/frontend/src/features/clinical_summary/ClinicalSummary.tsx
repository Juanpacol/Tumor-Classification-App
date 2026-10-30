import { Alert } from "@/features/portal/components";
import type { StudyDetail } from "@/services/api";
import { ProbabilityBars } from "./ProbabilityBars";
import { QualityMetrics } from "./QualityMetrics";
import { SafetyChecks } from "./SafetyChecks";
import { UncertaintyAlert } from "./UncertaintyAlert";
import styles from "./ClinicalSummary.module.css";

/** What the model concluded about a study, or why there is no conclusion. */
export function ClinicalSummary({ study }: { study: StudyDetail }) {
  const { prediction } = study;

  if (study.status === "rejected") {
    return (
      <div className={styles.summary}>
        <Alert tone="warning" title="Image rejected">
          {study.rejection_reason ??
            "The image did not pass the quality check."}
        </Alert>
        <SafetyChecks study={study} />
      </div>
    );
  }
  if (study.status === "failed") {
    return (
      <div className={styles.summary}>
        <Alert tone="warning" title="The analysis failed">
          The study could not be analyzed. Try uploading the image again.
        </Alert>
        <SafetyChecks study={study} />
      </div>
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
      <ProbabilityBars probabilities={prediction.probabilities} />
      <QualityMetrics prediction={prediction} />
      <SafetyChecks study={study} />
    </div>
  );
}
