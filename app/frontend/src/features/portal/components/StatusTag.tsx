import type { ReactNode } from "react";
import styles from "./StatusTag.module.css";

interface StatusTagProps {
  /**
   * `attention` is the clinical flag (needs expert review), `ok` a normal
   * result, `open` work in progress and `problem` a technical failure.
   */
  tone?: "neutral" | "attention" | "ok" | "open" | "problem";
  /** Decorative glyph; the text always carries the meaning. */
  icon?: ReactNode;
  children: string;
}

export function StatusTag({
  tone = "neutral",
  icon,
  children,
}: StatusTagProps) {
  return (
    <span className={`${styles.tag} ${styles[tone]}`}>
      {icon && (
        <span aria-hidden="true" className={styles.icon}>
          {icon}
        </span>
      )}
      {children}
    </span>
  );
}
