"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./TechStack.module.css";
import art from "./Honesty.module.css";

const STEPS = [
  {
    title: "1. Classify",
    text: "The hybrid classifier reads the scan. When it is not confident, Ether flags the study for an expert instead of forcing a class.",
  },
  {
    title: "2. Draft",
    text: "Muse Spark writes the report from the classifier's output and the image. It describes; it does not decide the class.",
  },
  {
    title: "3. Check",
    text: "Fixed rules compare every claim with the classifier's evidence. A claim whose numbers or class do not match is not verified, and wording that states a diagnosis or a treatment is rejected.",
  },
  {
    title: "4. Show",
    text: "The radiologist sees each claim labelled verified or not verified, with the evidence behind it, and the whole report as a draft to confirm.",
  },
];

/** How Ether keeps the language model from inventing, and what it does not claim. */
export function Honesty() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();
  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <h2 className={styles.title}>How we keep the AI honest</h2>
      <div className={styles.grid}>
        {STEPS.map((step) => (
          <div key={step.title} className={styles.card}>
            <h3 className={styles.cardTitle}>{step.title}</h3>
            <p className={styles.cardText}>{step.text}</p>
          </div>
        ))}
      </div>
      <p className={styles.limits}>
        What Ether does not claim: it is decision support, not a medical device
        and not a diagnosis. The classifier is a baseline model trained on a
        public dataset and not clinically validated, and the studies shown in
        the demo are samples. The chat answers are not yet checked by the
        verifier, and are labelled that way.
      </p>
      <div className={art.spacer} aria-hidden="true">
        <div className={art.hand} />
        <div className={art.face} />
      </div>
    </section>
  );
}
