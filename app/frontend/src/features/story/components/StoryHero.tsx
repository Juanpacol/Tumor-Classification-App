"use client";

import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./StoryHero.module.css";

export function StoryHero() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <span className={styles.eyebrow}>How we built Ether</span>
      <h1 className={styles.title}>
        A second opinion, built where it&apos;s hardest to get one.
      </h1>
      <p className={styles.text}>
        Ether started from a simple gap: radiologists in resource-limited
        environments often read brain MRIs alone, with no second set of eyes and
        no specialist on call. We set out to build an assistant that earns a
        seat in that room — one that shows its reasoning instead of just a
        verdict.
      </p>
    </section>
  );
}
