import { formatConfidence, TUMOR_LABELS } from "@/features/portal/labels";
import type { StudyDetail } from "@/services/api";
import styles from "./VerdictHeader.module.css";

const RADIUS = 30;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

/**
 * What the model concluded, in one glance. When it asks for expert review it
 * does not name a class: a weak prediction must not read as a verdict.
 */
export function VerdictHeader({ study }: { study: StudyDetail }) {
  const { prediction } = study;

  if (!prediction) {
    const text =
      study.status === "rejected"
        ? { caption: "Quality check", label: "Rejected" }
        : study.status === "failed"
          ? { caption: "Analysis", label: "Failed" }
          : { caption: "Analysis", label: "In progress" };
    return (
      <header className={styles.header}>
        <div className={styles.text}>
          <span className={styles.caption}>{text.caption}</span>
          <strong className={styles.label}>{text.label}</strong>
        </div>
      </header>
    );
  }

  const flagged = prediction.needs_expert_triage;
  const percent = Math.round(prediction.confidence * 100);
  return (
    <header
      className={`${styles.header} ${flagged ? styles.flagged : ""}`}
      style={
        {
          "--verdict": `var(--class-${prediction.label})`,
        } as React.CSSProperties
      }
    >
      <div className={styles.text}>
        <span className={styles.caption}>
          {flagged ? "Model confidence" : "Predicted class"}
        </span>
        <strong className={styles.label}>
          {flagged ? "Needs expert review" : TUMOR_LABELS[prediction.label]}
        </strong>
        <span className={styles.model}>Model {prediction.model_version}</span>
      </div>
      <div
        className={styles.ring}
        role="img"
        aria-label={`${formatConfidence(prediction.confidence)} confidence`}
      >
        <svg viewBox="0 0 72 72" aria-hidden="true">
          <circle className={styles.track} cx="36" cy="36" r={RADIUS} />
          <circle
            className={styles.arc}
            cx="36"
            cy="36"
            r={RADIUS}
            strokeDasharray={CIRCUMFERENCE}
            style={
              {
                "--offset": CIRCUMFERENCE * (1 - prediction.confidence),
                "--circumference": CIRCUMFERENCE,
              } as React.CSSProperties
            }
          />
        </svg>
        <span className={styles.percent}>{percent}%</span>
      </div>
    </header>
  );
}
