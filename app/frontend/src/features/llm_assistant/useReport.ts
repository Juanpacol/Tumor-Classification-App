"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient, Report, StudyDetail } from "@/services/api";

const GENERIC_ERROR = "The report could not be generated.";

function completed(report: Report | null): Report | null {
  return report?.status === "completed" && report.content ? report : null;
}

/**
 * The AI report of a study. A completed study with no usable report asks for
 * one automatically, once; any later attempt is `generate`, called by the
 * reader.
 */
export function useReport(client: ApiClient, study: StudyDetail) {
  const [report, setReport] = useState<Report | null>(completed(study.report));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requested = useRef(false);

  const generate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const generated = await client.generateReport(study.id);
      const usable = completed(generated);
      if (usable) setReport(usable);
      else setError(GENERIC_ERROR);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : GENERIC_ERROR);
    } finally {
      setLoading(false);
    }
  }, [client, study.id]);

  const shouldRequest = study.status === "completed" && report === null;
  useEffect(() => {
    if (shouldRequest && !requested.current) {
      requested.current = true;
      void generate();
    }
  }, [shouldRequest, generate]);

  return { report, loading, error, generate };
}
