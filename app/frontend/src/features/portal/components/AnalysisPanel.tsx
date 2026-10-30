"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { ClinicalSummary } from "@/features/clinical_summary/ClinicalSummary";
import { VerdictHeader } from "@/features/clinical_summary/VerdictHeader";
import { InteractiveChat } from "@/features/llm_assistant/InteractiveChat";
import { ReportSection } from "@/features/llm_assistant/ReportSection";
import type { ApiClient, StudyDetail } from "@/services/api";
import styles from "./AnalysisPanel.module.css";

type TabKey = "summary" | "report" | "chat";

const LABELS: Record<TabKey, string> = {
  summary: "Summary",
  report: "Report",
  chat: "Chat",
};

/** A rejected or failed study has nothing to report on, and the chat needs a result. */
function tabsFor(study: StudyDetail): TabKey[] {
  const tabs: TabKey[] = ["summary"];
  if (study.status !== "rejected" && study.status !== "failed") {
    tabs.push("report");
  }
  if (study.status === "completed") tabs.push("chat");
  return tabs;
}

/**
 * The analysis beside the image: the verdict on top, then the summary, the AI
 * report and the chat in tabs. The report and the chat load only the first time
 * their tab opens (the report asks the model for a draft), and then stay
 * mounted so a conversation survives a trip to another tab.
 */
export function AnalysisPanel({
  client,
  study,
}: {
  client: ApiClient;
  study: StudyDetail;
}) {
  const tabs = tabsFor(study);
  const [selected, setSelected] = useState<TabKey>("summary");
  const [visited, setVisited] = useState<Set<TabKey>>(new Set(["summary"]));
  const buttons = useRef<Partial<Record<TabKey, HTMLButtonElement | null>>>({});
  const active = tabs.includes(selected) ? selected : "summary";

  function open(tab: TabKey) {
    setSelected(tab);
    setVisited((current) => new Set(current).add(tab));
  }

  function onKeyDown(event: KeyboardEvent, index: number) {
    const step =
      event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (step === 0) return;
    event.preventDefault();
    const next = tabs[(index + step + tabs.length) % tabs.length]!;
    open(next);
    buttons.current[next]?.focus();
  }

  return (
    <div className={styles.frame}>
      <div className={styles.frameShadow}>
        <aside aria-label="Study analysis" className={styles.panel}>
          <div className={styles.backdrop} data-tab={active} aria-hidden="true">
            <span className={styles.layer} data-for="summary" />
            <span className={styles.layer} data-for="report" />
            <span className={styles.layer} data-for="chat" />
          </div>
          <VerdictHeader study={study} />
          <div className={styles.tabs} role="tablist" aria-label="Analysis">
            {tabs.map((tab, index) => (
              <button
                key={tab}
                ref={(element) => {
                  buttons.current[tab] = element;
                }}
                type="button"
                role="tab"
                id={`tab-${tab}`}
                aria-selected={active === tab}
                aria-controls={`panel-${tab}`}
                tabIndex={active === tab ? 0 : -1}
                className={styles.tab}
                onClick={() => open(tab)}
                onKeyDown={(event) => onKeyDown(event, index)}
              >
                {LABELS[tab]}
              </button>
            ))}
          </div>
          {tabs.map((tab) => (
            <div
              key={tab}
              role="tabpanel"
              id={`panel-${tab}`}
              aria-labelledby={`tab-${tab}`}
              hidden={active !== tab}
              className={styles.body}
            >
              {visited.has(tab) && tab === "summary" && (
                <ClinicalSummary study={study} />
              )}
              {visited.has(tab) && tab === "report" && (
                <ReportSection client={client} study={study} />
              )}
              {visited.has(tab) && tab === "chat" && (
                <InteractiveChat client={client} study={study} />
              )}
            </div>
          ))}
        </aside>
      </div>
    </div>
  );
}
