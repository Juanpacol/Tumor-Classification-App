import { formatConfidence, TUMOR_LABELS } from "@/features/portal/labels";
import type { Prediction } from "@/services/api";
import styles from "./PredictionBadge.module.css";

/**
 * The predicted class and its confidence. When the model is uncertain the
 * badge does not name a class: a weak prediction must not read as a verdict.
 */
export function PredictionBadge({ prediction }: { prediction: Prediction }) {
  const uncertain = prediction.needs_expert_triage;
  return (
    <div className={`${styles.badge} ${uncertain ? styles.uncertain : ""}`}>
      <span className={styles.caption}>
        {uncertain ? "Model confidence" : "Predicted class"}
      </span>
      <strong className={styles.label}>
        {uncertain ? "Needs expert review" : TUMOR_LABELS[prediction.label]}
      </strong>
      <span className={styles.confidence}>
        {formatConfidence(prediction.confidence)} confidence
      </span>
    </div>
  );
}
