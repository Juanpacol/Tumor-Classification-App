import type { Metadata } from "next";
import Link from "next/link";
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
      <div className={styles.storyConnector}>
        <div className={styles.ribcageLeft} />
        <div className={styles.brainRight} />
        <Link href="/" className={styles.backLink}>
          ← Back to Ether
        </Link>
        <StoryHero />
      </div>
    </main>
  );
}
