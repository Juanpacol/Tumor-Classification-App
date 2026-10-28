"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./OpenSource.module.css";

export function OpenSource() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <h2 className={styles.title}>
        This only gets better with more hands on it.
      </h2>
      <p className={styles.text}>
        Radiology support shouldn&apos;t be a black box owned by one team. Ether
        is open source so clinicians, researchers, and engineers anywhere can
        inspect it, challenge it, and push it forward.
      </p>
      <a
        href="https://github.com/Juanpacol/Tumor-Classification-App"
        target="_blank"
        rel="noopener noreferrer"
        className={styles.cta}
      >
        View the repository
      </a>
    </section>
  );
}
