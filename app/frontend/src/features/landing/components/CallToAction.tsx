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
        <div className={styles.social}>
          <a
            href="https://github.com/Juanpacol/Tumor-Classification-App"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          <a
            href="https://x.com/Juannlun"
            target="_blank"
            rel="noopener noreferrer"
          >
            X
          </a>
          <a
            href="https://www.linkedin.com/in/juan-pablo-botero-442580257/"
            target="_blank"
            rel="noopener noreferrer"
          >
            LinkedIn
          </a>
        </div>
        <p className={styles.credits}>
          Built by Juan Pablo &amp; Mateo — Ether
        </p>
      </div>
    </section>
  );
}
