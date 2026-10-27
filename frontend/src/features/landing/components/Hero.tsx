import styles from "./Hero.module.css";

export function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.bg} />
      <div className={styles.overlay} />
      <div className={styles.content}>
        <span className={styles.wordmark}>ETHER</span>
        <h1 className={styles.headline}>
          Give every hospital access to an expert radiologist, in minutes.
        </h1>
        <p className={styles.subcopy}>
          Ether gives radiologists in resource-limited settings a fast,
          trustworthy second opinion on brain MRI scans.
        </p>
        <div className={styles.ctaRow}>
          <button className={styles.ctaPrimary}>Request a demo</button>
          <button className={styles.ctaSecondary}>See it in action</button>
        </div>
      </div>
    </section>
  );
}
