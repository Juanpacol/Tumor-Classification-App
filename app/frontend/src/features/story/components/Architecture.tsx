"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./Architecture.module.css";

const LAYERS = [
  {
    number: "01",
    title: "Frontend",
    tag: "Next.js, Feature-Sliced",
    text: "The clinical portal radiologists use day to day: DICOM visualization, clinical summaries, and an LLM assistant, each isolated in its own feature module.",
  },
  {
    number: "02",
    title: "Backend",
    tag: "FastAPI, Service-Client",
    text: "Thin HTTP routes hand off to a services layer for business logic, with external integrations like storage and the LLM kept in their own client layer.",
  },
  {
    number: "03",
    title: "ML Engine",
    tag: "Standalone pipeline",
    text: "An importable, testable package: DICOM preprocessing, classical radiomics features, and a CNN with Grad-CAM, orchestrated end to end.",
  },
];

export function Architecture() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <h2 className={styles.title}>How it&apos;s structured</h2>
      <div className={styles.stack}>
        {LAYERS.map((layer) => (
          <div key={layer.number} className={styles.layer}>
            <span className={styles.layerNumber}>{layer.number}</span>
            <div className={styles.layerBody}>
              <h3 className={styles.layerTitle}>{layer.title}</h3>
              <span className={styles.layerTag}>{layer.tag}</span>
              <p className={styles.layerText}>{layer.text}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
