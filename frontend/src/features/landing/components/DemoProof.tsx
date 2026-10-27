import styles from "./DemoProof.module.css";

export function DemoProof() {
  return (
    <section className={styles.section}>
      <h2 className={styles.title}>See it in action</h2>
      <div className={styles.frame}>
        <div className={styles.frameHeader}>
          <span className={styles.dot} />
          <span className={styles.dot} />
          <span className={styles.dot} />
        </div>
        <div className={`${styles.frameBody} halftone`}>
          DICOM viewer with Grad-CAM overlay — demo recording goes here
        </div>
      </div>
    </section>
  );
}
