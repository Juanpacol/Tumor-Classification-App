"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./ProblemSolution.module.css";

export function ProblemSolution() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <div className={styles.inner}>
        <div className={styles.rule} />
        <p className={styles.stat}>1 radiologist</p>
        <p className={styles.statLabel}>
          per 100,000+ people in many underserved regions.
        </p>
        <p className={styles.consequence}>
          So scans pile up, and patients wait for an answer that should take
          minutes, not weeks.
        </p>
        <p className={styles.solution}>
          Ether gives a trustworthy second opinion in minutes — not a diagnosis,
          a head start for the radiologist who reviews the case.
        </p>
      </div>
    </section>
  );
}
