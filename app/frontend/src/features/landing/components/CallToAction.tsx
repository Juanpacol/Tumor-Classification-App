"use client";

import Link from "next/link";
import { useFadeUp } from "@/hooks/useFadeUp";
import styles from "./CallToAction.module.css";

const TEAM = [
  {
    name: "German Joel Ramos",
    x: "#",
    linkedin: "#",
  },
  {
    name: "Juan Pablo Botero",
    x: "https://x.com/Juannlun",
    linkedin: "https://www.linkedin.com/in/juan-pablo-botero-442580257/",
  },
  {
    name: "Pete Paul Narvaez",
    x: "#",
    linkedin: "#",
  },
];

export function CallToAction() {
  const { ref, isVisible } = useFadeUp<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.section} fadeUp ${isVisible ? "isVisible" : ""}`}
    >
      <div className={styles.inner}>
        <h2 className={styles.title}>Try the Ether demo.</h2>
        <Link href="/studies" className={styles.cta}>
          Open the portal
        </Link>

        <a
          href="https://github.com/Juanpacol/Tumor-Classification-App"
          target="_blank"
          rel="noopener noreferrer"
          className={styles.repoLink}
        >
          GitHub repository
        </a>

        <Link href="/how-we-built-it" className={styles.storyLink}>
          How we built Ether
        </Link>

        <div className={styles.team}>
          {TEAM.map((member) => (
            <div key={member.name} className={styles.teamMember}>
              <p className={styles.teamName}>{member.name}</p>
              <div className={styles.teamLinks}>
                <a href={member.x} target="_blank" rel="noopener noreferrer">
                  X
                </a>
                <span aria-hidden="true">·</span>
                <a
                  href={member.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  LinkedIn
                </a>
              </div>
            </div>
          ))}
        </div>

        <p className={styles.credits}>Ether</p>
      </div>
    </section>
  );
}
