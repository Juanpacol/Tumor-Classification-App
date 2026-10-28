import { Hero } from "@/features/landing/components/Hero";
import { ProblemSolution } from "@/features/landing/components/ProblemSolution";
import { HowItWorks } from "@/features/landing/components/HowItWorks";
import { SafetyRubric } from "@/features/landing/components/SafetyRubric";
import { DemoProof } from "@/features/landing/components/DemoProof";
import { Differentiators } from "@/features/landing/components/Differentiators";
import { CallToAction } from "@/features/landing/components/CallToAction";
import styles from "./page.module.css";

export default function HomePage() {
  return (
    <main>
      <Hero />
      <div className={styles.connector}>
        <div className={styles.veinLeft} />
        <div className={styles.veinRight} />
        <ProblemSolution />
        <HowItWorks />
      </div>
      <SafetyRubric />
      <DemoProof />
      <div className={styles.footerConnector}>
        <div className={styles.footerFoot} />
        <div className={styles.footerSkull} />
        <span className={styles.footerGhostText}>ETHER</span>
        <Differentiators />
        <CallToAction />
      </div>
    </main>
  );
}
