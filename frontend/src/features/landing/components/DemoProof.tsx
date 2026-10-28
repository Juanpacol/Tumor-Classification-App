"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./DemoProof.module.css";

export function DemoProof() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <div className={styles.brainAccent} />
      <div className={styles.nerveAccent} />
      <div className={styles.inner}>
        <h2 className={styles.title}>See it in action</h2>
        <div className={styles.frame}>
          <div className={styles.frameHeader}>
            <span className={styles.dot} />
            <span className={styles.dot} />
            <span className={styles.dot} />
          </div>
          <div className={`${styles.frameBody} halftone`}>
            DICOM viewer with Grad-CAM overlay — demo recording goes here
          </div>
        </div>
      </div>
    </section>
  );
}
