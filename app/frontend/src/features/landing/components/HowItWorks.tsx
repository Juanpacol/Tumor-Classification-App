"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./HowItWorks.module.css";

const STEPS = [
  {
    number: "01",
    title: "Reads the scan twice",
    text: "One model checks structure and texture, another checks visual patterns — two independent readings, not one guess.",
  },
  {
    number: "02",
    title: "Agrees with itself, or says so",
    text: "Combines both readings into one finding, and flags the case when they disagree.",
  },
  {
    number: "03",
    title: "Shows its work",
    text: "Reports the probability of every class and the measurements behind the call, so the radiologist can check it, not just trust it.",
  },
  {
    number: "04",
    title: "Hands off cleanly",
    text: "Packages the finding, the confidence, and the report with its cited evidence into one case file — ready for the radiologist to review.",
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
