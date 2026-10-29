"use client";

import { useState } from "react";
import { Alert, Button, Card } from "@/features/portal/components";
import type { ApiClient, StudyDetail } from "@/services/api";
import { ReportView } from "./ReportView";
import { useReport } from "./useReport";
import styles from "./ReportSection.module.css";

/**
 * The AI-drafted clinical report of a study. It is hidden for rejected and
 * failed studies, which have nothing to report on.
 */
export function ReportSection({
  client,
  study,
}: {
  client: ApiClient;
  study: StudyDetail;
}) {
  if (study.status === "rejected" || study.status === "failed") return null;
  return <Report client={client} study={study} />;
}

function Report({ client, study }: { client: ApiClient; study: StudyDetail }) {
  const { report, loading, error, generate } = useReport(client, study);
  const [confirming, setConfirming] = useState(false);

  async function regenerate() {
    setConfirming(false);
    await generate();
  }

  return (
    <Card aria-label="AI clinical report">
      <h2 className={styles.heading}>AI clinical report</h2>
      {study.status !== "completed" && (
        <p className={styles.message}>Available when the analysis finishes.</p>
      )}
      {loading && (
        <p className={styles.message} role="status">
          Generating the report…
        </p>
      )}
      {error && (
        <Alert tone="warning" title="The report could not be generated">
          <p>{error}</p>
          <Button variant="quiet" onClick={generate} disabled={loading}>
            Retry
          </Button>
        </Alert>
      )}
      {report?.content && (
        <>
          <ReportView content={report.content} />
          {confirming ? (
            <div className={styles.confirm} role="group" aria-label="Confirm">
              <p>Replace this report with a new draft?</p>
              <Button onClick={regenerate} disabled={loading}>
                Regenerate
              </Button>
              <Button variant="quiet" onClick={() => setConfirming(false)}>
                Cancel
              </Button>
            </div>
          ) : (
            <Button
              variant="quiet"
              className={styles.regenerate}
              disabled={loading}
              onClick={() => setConfirming(true)}
            >
              Regenerate report
            </Button>
          )}
        </>
      )}
    </Card>
  );
}
