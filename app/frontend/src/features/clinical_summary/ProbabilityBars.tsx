import { TUMOR_LABELS } from "@/features/portal/labels";
import type { Prediction, TumorLabel } from "@/services/api";
import styles from "./ProbabilityBars.module.css";

/** The four class probabilities, highest first. */
export function ProbabilityBars({
  probabilities,
}: {
  probabilities: Prediction["probabilities"];
}) {
  const rows = (Object.entries(probabilities) as [TumorLabel, number][]).sort(
    (a, b) => b[1] - a[1],
  );
  return (
    <ul className={styles.list}>
      {rows.map(([label, probability]) => {
        const percent = Math.round(probability * 100);
        return (
          <li key={label} className={styles.row}>
            <span>{TUMOR_LABELS[label]}</span>
            <div
              role="meter"
              aria-label={`${TUMOR_LABELS[label]} probability`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percent}
              className={styles.track}
            >
              <div className={styles.fill} style={{ width: `${percent}%` }} />
            </div>
            <span className={styles.value}>{percent}%</span>
          </li>
        );
      })}
    </ul>
  );
}
