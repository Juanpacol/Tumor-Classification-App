import { Hero } from "@/features/landing/components/Hero";
import { ProblemSolution } from "@/features/landing/components/ProblemSolution";
import { HowItWorks } from "@/features/landing/components/HowItWorks";
import { SafetyRubric } from "@/features/landing/components/SafetyRubric";
import { DemoProof } from "@/features/landing/components/DemoProof";
import { Differentiators } from "@/features/landing/components/Differentiators";
import { OpenSource } from "@/features/landing/components/OpenSource";
import { CallToAction } from "@/features/landing/components/CallToAction";
import styles from "./page.module.css";

export default function HomePage() {
  return (
    <main>
      <link rel="preload" as="image" href="/test-hand-bones.png" />
      <link rel="preload" as="image" href="/test-skull-eye.png" />
      <link rel="preload" as="image" href="/test-foot-bones.png" />
      <Hero />
      <div className={styles.connector}>
        <div className={styles.veinLeftBackdrop}>
          <div className={styles.veinLeft} />
        </div>
        <div className={styles.veinRightBackdrop}>
          <div className={styles.veinRightFade}>
            <div className={styles.veinRight} />
          </div>
        </div>
        <ProblemSolution />
        <HowItWorks />
      </div>
      <SafetyRubric />
      <DemoProof />
      <OpenSource />
      <div className={styles.footerConnector}>
        <div className={styles.footerFootBackdrop}>
          <div className={styles.footerFoot} />
        </div>
        <div className={styles.footerSkullBackdrop}>
          <div className={styles.footerSkull} />
        </div>
        <span className={styles.footerGhostText}>ETHER</span>
        <Differentiators />
        <CallToAction />
      </div>
    </main>
  );
}
