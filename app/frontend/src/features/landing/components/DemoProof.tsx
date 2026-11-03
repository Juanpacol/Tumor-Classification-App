"use client";

import { useEffect, useRef, useState } from "react";
import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./DemoProof.module.css";

/**
 * "See it in action": a muted, looping clip of the real portal with burned-in
 * captions. The link to the full video shows only when
 * `NEXT_PUBLIC_DEMO_VIDEO_URL` is set. With reduced motion the clip does not play
 * on its own and shows its controls.
 */
export function DemoProof() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const fullVideoUrl = process.env.NEXT_PUBLIC_DEMO_VIDEO_URL;

  useEffect(() => {
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    setReducedMotion(true);
    videoRef.current?.pause();
  }, []);

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
          <video
            ref={videoRef}
            className={styles.video}
            src="/demo/ether-demo-loop.mp4"
            poster="/demo/ether-demo-poster.jpg"
            aria-label="Silent clip of the portal: upload a scan, read the calibrated result, check the cited evidence and see a low-confidence case flagged for expert review"
            muted
            loop
            playsInline
            preload="metadata"
            autoPlay={!reducedMotion}
            controls={reducedMotion}
          />
        </div>
        <p className={styles.note}>
          Sample data from a public dataset. A prototype, not a medical device.
        </p>
        {fullVideoUrl ? (
          <a
            className={styles.fullVideo}
            href={fullVideoUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            Watch the full video (2 min)
          </a>
        ) : null}
      </div>
    </section>
  );
}
