import type { ReactNode } from "react";
import styles from "./Alert.module.css";

interface AlertProps {
  /** `warning` interrupts the reader (uncertainty, discrepancy); `neutral` informs. */
  tone?: "neutral" | "warning";
  title?: string;
  children: ReactNode;
}

export function Alert({ tone = "neutral", title, children }: AlertProps) {
  return (
    <div
      role={tone === "warning" ? "alert" : "status"}
      className={`${styles.alert} ${styles[tone]}`}
    >
      {title && <strong className={styles.title}>{title}</strong>}
      <div>{children}</div>
    </div>
  );
}
