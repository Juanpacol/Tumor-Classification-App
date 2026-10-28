"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./HowItWorks.module.css";

const STEPS = [
  {
    number: "01",
    title: "Texture Analysis",
    text: "Extracts texture and structural features from the scan.",
  },
  {
    number: "02",
    title: "Visual Patterns",
    text: "Analyzes visual patterns with a deep learning model.",
  },
  {
    number: "03",
    title: "Single Finding",
    text: "Combines both signals into a single finding.",
  },
  {
    number: "04",
    title: "Show the Why",
    text: "Shows exactly which part of the brain drove the conclusion.",
  },
];

export function HowItWorks() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <h2 className={styles.title}>How It Works</h2>
      <div className={styles.grid}>
        {STEPS.map((step) => (
          <div key={step.number} className={styles.step}>
            <span className={styles.stepNumber}>{step.number}</span>
            <h3 className={styles.stepTitle}>{step.title}</h3>
            <p className={styles.stepText}>{step.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
