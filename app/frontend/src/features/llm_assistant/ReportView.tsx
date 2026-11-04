import { Alert } from "@/features/portal/components";
import { TUMOR_LABELS, formatConfidence } from "@/features/portal/labels";
import type {
  AiAgreement,
  Citation,
  ClinicalReport,
  Prediction,
} from "@/services/api";
import styles from "./ReportView.module.css";

export const REPORT_DISCLAIMER =
  "AI-generated draft for decision support only. It is not a diagnosis and must be reviewed and confirmed by a qualified radiologist.";

const DISCREPANCY: Record<Exclude<AiAgreement, "agrees">, string> = {
  disagrees:
    "The classifier and the AI assistant reached different conclusions about this image.",
  uncertain:
    "The AI assistant could not confirm the classifier's result from this image.",
};

/** What the AI assistant concluded about the classifier's result, in a few words. */
const AGREEMENT: Record<AiAgreement, string> = {
  agrees: "AI assistant agrees",
  disagrees: "AI assistant disagrees",
  uncertain: "AI assistant is not sure",
};

/**
 * The drafted report: a one-line headline, the conclusion first, the
 * supporting sections after it, and the evidence folded away to audit.
 */
export function ReportView({
  content,
  prediction,
}: {
  content: ClinicalReport;
  prediction: Prediction | null;
}) {
  const { ai_agreement: agreement } = content;
  return (
    <div className={styles.report}>
      {prediction && (
        <p className={styles.headline}>
          <strong>{TUMOR_LABELS[prediction.label]}</strong>
          <span>{formatConfidence(prediction.confidence)}</span>
          <span>{AGREEMENT[agreement]}</span>
        </p>
      )}
      {agreement !== "agrees" && (
        <Alert tone="warning" title="Classifier and AI assistant do not agree">
          {DISCREPANCY[agreement]} Ask a specialist to review this study.
        </Alert>
      )}
      <Section title="Findings">{content.findings}</Section>
      <div className={styles.impression}>
        <Section title="Impression">{content.impression}</Section>
      </div>
      <div className={styles.secondary}>
        <Section title="Visual observations">
          {content.visual_observations}
        </Section>
        <Section title="Differential considerations">
          {content.differential_considerations}
        </Section>
        <section>
          <h3 className={styles.heading}>Suggested next steps</h3>
          <ol className={styles.steps}>
            {content.suggested_next_steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        </section>
      </div>
      <Evidence citations={content.citations} />
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

/** Every claim with the evidence it cites, folded away until someone wants to audit it. */
function Evidence({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  const verified = citations.filter((citation) => citation.verified).length;
  return (
    <details className={styles.evidence}>
      <summary className={styles.evidenceSummary}>
        <span>Evidence ({citations.length})</span>
        <span className={styles.tally}>
          {verified} verified · {citations.length - verified} not verified
        </span>
      </summary>
      <ul className={styles.citations}>
        {citations.map((citation) => (
          <li key={citation.claim}>
            <span
              className={
                citation.verified ? styles.verified : styles.unverified
              }
            >
              {citation.verified ? "Verified" : "Not verified"}
            </span>{" "}
            {citation.claim}
            {citation.evidence.length > 0 && (
              <ul className={styles.refs}>
                {citation.evidence.map((ref) => (
                  <li key={ref.key}>
                    <code>{ref.key}</code>: {ref.description}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}
