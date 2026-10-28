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
      <h2 className={styles.title}>Open source, by design</h2>
      <p className={styles.text}>
        Ether runs on auditable code, not a black box — the architecture, every
        decision, and the model behind each result are public.
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
