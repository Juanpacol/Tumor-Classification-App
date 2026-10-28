import type { Metadata } from "next";
import { StoryHero } from "@/features/story/components/StoryHero";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "How we built Ether",
  description:
    "The story, stack, and architecture behind Ether's hybrid brain tumor detection support system.",
};

export default function HowWeBuiltItPage() {
  return (
    <main className={styles.main}>
      <StoryHero />
    </main>
  );
}
