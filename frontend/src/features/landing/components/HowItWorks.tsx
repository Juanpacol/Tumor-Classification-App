import styles from "./HowItWorks.module.css";

const STEPS = [
  {
    number: "#1 Extract",
    title: "Texture Analysis",
    text: "Extracts texture and structural features from the scan.",
    image: "Radiomics feature map",
  },
  {
    number: "#2 Analyze",
    title: "Visual Patterns",
    text: "Analyzes visual patterns with a deep learning model.",
    image: "CNN activation preview",
  },
  {
    number: "#3 Combine",
    title: "Single Finding",
    text: "Combines both signals into a single finding.",
    image: "Fusion score diagram",
  },
  {
    number: "#4 Explain",
    title: "Show the Why",
    text: "Shows exactly which part of the brain drove the conclusion.",
    image: "Grad-CAM overlay",
  },
];

export function HowItWorks() {
  return (
    <section className={styles.section}>
      <h2 className={styles.title}>How It Works</h2>
      <div className={styles.grid}>
        {STEPS.map((step) => (
          <div key={step.number} className={styles.step}>
            <span className={styles.stepNumber}>{step.number}</span>
            <h3 className={styles.stepTitle}>{step.title}</h3>
            <p className={styles.stepText}>{step.text}</p>
            <div className={`${styles.stepImage} halftone`}>{step.image}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
