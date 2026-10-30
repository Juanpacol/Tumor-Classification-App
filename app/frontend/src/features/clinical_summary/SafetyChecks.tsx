import type { CSSProperties } from "react";
import type { StudyDetail } from "@/services/api";
import { safetyChecks, type CheckState } from "./safetyRules";
import styles from "./SafetyChecks.module.css";

const GLYPHS: Record<CheckState, string> = {
  pass: "✓",
  attention: "▲",
  pending: "…",
  note: "•",
};

/** The safeguards applied to this study, each with its real state. */
export function SafetyChecks({ study }: { study: StudyDetail }) {
  const checks = safetyChecks(study);
  return (
    <section aria-label="Safety checks" className={styles.section}>
      <h3 className={styles.heading}>Safety checks</h3>
      <ul className={styles.list}>
        {checks.map((check, index) => (
          <li
            key={check.key}
            className={styles.item}
            style={{ "--i": index } as CSSProperties}
          >
            <span
              className={`${styles.glyph} ${styles[check.state]}`}
              aria-hidden="true"
            >
              {GLYPHS[check.state]}
            </span>
            <span>
              <strong className={styles.label}>{check.label}</strong>
              <span className={styles.detail}>{check.detail}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
