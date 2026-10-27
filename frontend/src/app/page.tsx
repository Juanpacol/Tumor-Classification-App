import { Hero } from "@/features/landing/components/Hero";
import { ProblemSolution } from "@/features/landing/components/ProblemSolution";
import { HowItWorks } from "@/features/landing/components/HowItWorks";
import { SafetyRubric } from "@/features/landing/components/SafetyRubric";
import { DemoProof } from "@/features/landing/components/DemoProof";
import { Differentiators } from "@/features/landing/components/Differentiators";
import { CallToAction } from "@/features/landing/components/CallToAction";

export default function HomePage() {
  return (
    <main>
      <Hero />
      <ProblemSolution />
      <HowItWorks />
      <SafetyRubric />
      <DemoProof />
      <Differentiators />
      <CallToAction />
    </main>
  );
}
