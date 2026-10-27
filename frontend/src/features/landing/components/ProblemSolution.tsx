import styles from "./ProblemSolution.module.css";

export function ProblemSolution() {
  return (
    <section className={styles.section}>
      <p className={styles.stat}>1 radiologist</p>
      <p className={styles.statLabel}>
        per 100,000+ people in many underserved regions.
      </p>
      <p className={styles.solution}>
        Ether gives a trustworthy second opinion in minutes.
      </p>
    </section>
  );
}
