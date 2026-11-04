"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./Architecture.module.css";

const LAYERS = [
  {
    number: "01",
    title: "Frontend",
    tag: "Next.js, Feature-Sliced",
    text: "The clinical portal: a ranked worklist, the MRI viewer, a verified report and a chat, each isolated in its own feature module.",
  },
  {
    number: "02",
    title: "Backend",
    tag: "FastAPI, Hexagonal",
    text: "Use cases sit in the middle and talk to the classifier, the language model, storage and the database through ports. Swapping Muse Spark for Llama, or Postgres for another store, touches one adapter.",
  },
  {
    number: "03",
    title: "ML Engine",
    tag: "Standalone pipeline",
    text: "An importable, testable package: image quality gate, radiomic features, CNN features and a calibrated classifier, trained on 3,434 cleaned images and released as versioned artifacts.",
  },
];

export function Architecture() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <div className={styles.art} aria-hidden="true" />
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
