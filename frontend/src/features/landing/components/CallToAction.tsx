import styles from "./CallToAction.module.css";

export function CallToAction() {
  return (
    <section className={styles.section}>
      <h2 className={styles.title}>Request a demo of Ether.</h2>
      <button className={styles.cta}>Request a demo</button>
      <p className={styles.credits}>Built by Juan Pablo &amp; Mateo — Ether</p>
    </section>
  );
}
