"use client";

import { useCallback, useEffect, useState } from "react";
import type { ApiClient, Page, StudyDetail } from "@/services/api";

export const PAGE_SIZE = 10;

export type StudiesState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: Page<StudyDetail> };

/** Loads one page of studies; `retry` repeats the request after a failure. */
export function useStudies(client: ApiClient, page: number) {
  const [state, setState] = useState<StudiesState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let current = true;
    setState({ status: "loading" });
    client
      .listStudies({ page, pageSize: PAGE_SIZE })
      .then((data) => current && setState({ status: "ready", data }))
      .catch(
        (error: unknown) =>
          current &&
          setState({
            status: "error",
            message:
              error instanceof Error ? error.message : "Something went wrong.",
          }),
      );
    return () => {
      current = false;
    };
  }, [client, page, attempt]);

  const retry = useCallback(() => setAttempt((count) => count + 1), []);
  return { state, retry };
}
