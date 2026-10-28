"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./SafetyRubric.module.css";

const CHECKS = [
  "Scan quality check before any prediction is attempted.",
  "Confidence threshold enforced — low-confidence cases are flagged, not force-classified.",
  "Cases below threshold are marked for mandatory human review.",
  "The system never outputs a definitive diagnosis — only decision-support language.",
];

export function SafetyRubric() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <div className={styles.brainAccent} />
      <div className={styles.gearAccent} />
      <div className={styles.inner}>
        <h2 className={styles.title}>We know when not to trust ourselves.</h2>
        <p className={styles.tagline}>And we tell you.</p>
        <div className={styles.card}>
          <ul className={styles.list}>
            {CHECKS.map((check) => (
              <li key={check} className={styles.item}>
                <span className={styles.check}>✓</span>
                <span>{check}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
