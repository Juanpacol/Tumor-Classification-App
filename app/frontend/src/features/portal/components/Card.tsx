import type { HTMLAttributes } from "react";
import styles from "./Card.module.css";

interface CardProps extends HTMLAttributes<HTMLElement> {
  /** `plate` is the dark surface used for the MRI viewer. */
  tone?: "paper" | "plate";
}

export function Card({ tone = "paper", className = "", ...props }: CardProps) {
  return (
    <section
      className={`${styles.card} ${styles[tone]} ${className}`}
      {...props}
    />
  );
}
