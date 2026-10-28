"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./CallToAction.module.css";

export function CallToAction() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <div className={styles.inner}>
        <h2 className={styles.title}>Request a demo of Ether.</h2>
        <button className={styles.cta}>Request a demo</button>
        <p className={styles.credits}>
          Built by Juan Pablo &amp; Mateo — Ether
        </p>
      </div>
    </section>
  );
}
