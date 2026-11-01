"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./Differentiators.module.css";

const ITEMS = [
  {
    title: "Nothing to take on faith",
    text: "Every result comes with a heatmap of the exact region that drove it — not a black-box score you're asked to trust.",
  },
  {
    title: "Every claim shows its source",
    text: "Each statement in the report points to the study data it came from. Anything that can't be traced is flagged as not verified instead of passed off as fact.",
  },
  {
    title: "Works where the cloud doesn't",
    text: "Runs on an on-premise, open-source model instead of a locked-in cloud API — built for clinics without reliable connectivity.",
  },
  {
    title: "Built for the data you actually have",
    text: "Combines classical radiomics with deep learning, which generalizes better on the small, uneven datasets real clinics work with.",
  },
];

export function Differentiators() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <h2 className={styles.title}>Why it holds up outside a demo</h2>
      <div className={styles.grid}>
        {ITEMS.map((item) => (
          <div key={item.title} className={styles.card}>
            <h3 className={styles.cardTitle}>{item.title}</h3>
            <p className={styles.cardText}>{item.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
