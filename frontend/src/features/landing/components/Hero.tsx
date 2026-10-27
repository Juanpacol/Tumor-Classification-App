"use client";

import { useEffect, useRef } from "react";
import styles from "./Hero.module.css";

export function Hero() {
  const imageRef = useRef<HTMLDivElement>(null);
  const imageLeftRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onScroll() {
      const y = window.scrollY;
      if (imageRef.current) {
        imageRef.current.style.transform = `scaleX(-1) translateY(${y * 0.12}px)`;
      }
      if (imageLeftRef.current) {
        imageLeftRef.current.style.transform = `scaleX(-1) translateY(${y * -0.12}px)`;
      }
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <section className={styles.hero}>
      <div ref={imageRef} className={styles.image} />
      <div ref={imageLeftRef} className={styles.imageLeft} />
      <div className={styles.content}>
        <span className={styles.wordmark}>ETHER</span>
        <p className={styles.tagline}>Built to see what others miss.</p>
        <h1 className={styles.headline}>
          What the eye misses, Ether finds.
        </h1>
      </div>
    </section>
  );
}
