"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./TechStack.module.css";

const STACK = [
  {
    title: "Muse Spark (Meta)",
    text: "Writes the draft report and answers questions through the Meta Model API, chosen for its multimodal and health capabilities. It never classifies. Open Llama stays as a fallback behind the same interface.",
  },
  {
    title: "PyTorch + Radiomics",
    text: "A hybrid classifier: 45 classical radiomic features alongside 1,280 deep features from a pretrained CNN, which generalizes better on small, heterogeneous MRI datasets than deep learning alone.",
  },
  {
    title: "FastAPI + Postgres",
    text: "Async Python backend with automatic OpenAPI docs, storing studies, reports and conversations in Postgres.",
  },
  {
    title: "Next.js + Cornerstone3D",
    text: "The clinical portal radiologists use day to day, with an MRI viewer built on Cornerstone3D. Uploads are JPG and PNG images.",
  },
];

export function TechStack() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <h2 className={styles.title}>What it&apos;s built with</h2>
      <div className={styles.grid}>
        {STACK.map((item) => (
          <div key={item.title} className={styles.card}>
            <h3 className={styles.cardTitle}>{item.title}</h3>
            <p className={styles.cardText}>{item.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
