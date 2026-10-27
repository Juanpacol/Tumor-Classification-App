import styles from "./Differentiators.module.css";

const ITEMS = [
  {
    title: "Explainability by design",
    text: "Grad-CAM isn't a feature checkbox — it's the trust mechanism at the core of every result.",
  },
  {
    title: "Data sovereignty",
    text: "Runs on an on-premise, open-source LLM instead of a locked-in cloud API — built for clinics without reliable cloud access.",
  },
  {
    title: "Hybrid ML architecture",
    text: "Classical radiomics plus deep learning generalize better on small, heterogeneous datasets.",
  },
];

export function Differentiators() {
  return (
    <section className={styles.section}>
      <h2 className={styles.title}>For the technical judges</h2>
      <div className={styles.grid}>
        {ITEMS.map((item) => (
          <div key={item.title} className={styles.card}>
            <h3 className={styles.cardTitle}>{item.title}</h3>
            <p className={styles.cardText}>{item.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
