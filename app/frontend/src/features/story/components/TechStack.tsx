"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./TechStack.module.css";

const STACK = [
  {
    title: "FastAPI",
    text: "Async Python backend serving the inference API, with automatic OpenAPI docs for the clinical team.",
  },
  {
    title: "PyTorch + Radiomics",
    text: "A hybrid pipeline: classical GLCM/LBP/wavelet features alongside a CNN, which generalizes better on small, heterogeneous MRI datasets than deep learning alone.",
  },
  {
    title: "Next.js + Cornerstone3D",
    text: "The clinical portal radiologists use day to day, with DICOM visualization built on the medical imaging standard Cornerstone3D.",
  },
  {
    title: "Llama (open source)",
    text: "An on-premise LLM instead of a locked-in cloud API — built for clinics without reliable cloud access, and for data sovereignty.",
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
