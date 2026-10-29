import { Alert } from "@/features/portal/components";
import type { AiAgreement, ClinicalReport } from "@/services/api";
import styles from "./ReportView.module.css";

export const REPORT_DISCLAIMER =
  "AI-generated draft for decision support only. It is not a diagnosis and must be reviewed and confirmed by a qualified radiologist.";

const DISCREPANCY: Record<Exclude<AiAgreement, "agrees">, string> = {
  disagrees:
    "The classifier and the AI assistant reached different conclusions about this image.",
  uncertain:
    "The AI assistant could not confirm the classifier's result from this image.",
};

/** The five sections of the drafted report, with the discrepancy alert and the disclaimer. */
export function ReportView({ content }: { content: ClinicalReport }) {
  const { ai_agreement: agreement } = content;
  return (
    <div className={styles.report}>
      {agreement !== "agrees" && (
        <Alert tone="warning" title="Classifier and AI assistant do not agree">
          {DISCREPANCY[agreement]} Ask a specialist to review this study.
        </Alert>
      )}
      <Section title="Findings">{content.findings}</Section>
      <Section title="Visual observations">
        {content.visual_observations}
      </Section>
      <Section title="Impression">{content.impression}</Section>
      <Section title="Differential considerations">
        {content.differential_considerations}
      </Section>
      <section>
        <h3 className={styles.heading}>Suggested next steps</h3>
        <ul className={styles.steps}>
          {content.suggested_next_steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ul>
      </section>
      <p className={styles.disclaimer}>{REPORT_DISCLAIMER}</p>
    </div>
  );
}

function Section({ title, children }: { title: string; children: string }) {
  return (
    <section>
      <h3 className={styles.heading}>{title}</h3>
      <p>{children}</p>
    </section>
  );
}
