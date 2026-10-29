import styles from "./StatusTag.module.css";

interface StatusTagProps {
  /** `attention` marks what needs a clinician's eyes first. */
  tone?: "neutral" | "attention";
  children: string;
}

export function StatusTag({ tone = "neutral", children }: StatusTagProps) {
  return <span className={`${styles.tag} ${styles[tone]}`}>{children}</span>;
}
