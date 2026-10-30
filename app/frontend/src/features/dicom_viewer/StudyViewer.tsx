"use client";

import { useState, type ReactNode } from "react";
import styles from "./StudyViewer.module.css";
import type { UnavailableReason, ViewerTool } from "./viewerEngine";
import { DEFAULT_TOOL } from "./viewerEngine";
import { useViewer } from "./useViewer";
import { formatZoom, ZOOM_STEP } from "./zoom";

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      className={styles.icon}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const TOOLS: {
  tool: ViewerTool;
  label: string;
  hint: string;
  icon: ReactNode;
}[] = [
  {
    tool: "windowLevel",
    label: "Window/level",
    hint: "Drag to adjust the contrast",
    icon: (
      <Icon>
        <circle cx="8" cy="8" r="6" />
        <path d="M8 2a6 6 0 0 1 0 12z" fill="currentColor" />
      </Icon>
    ),
  },
  {
    tool: "zoom",
    label: "Zoom",
    hint: "Drag up or down to zoom",
    icon: (
      <Icon>
        <circle cx="7" cy="7" r="4.5" />
        <path d="M10.5 10.5 14 14" />
      </Icon>
    ),
  },
  {
    tool: "pan",
    label: "Pan",
    hint: "Drag to move the image",
    icon: (
      <Icon>
        <path d="M8 1.5v13M1.5 8h13M8 1.5 6.5 3M8 1.5 9.5 3M8 14.5 6.5 13M8 14.5 9.5 13M1.5 8 3 6.5M1.5 8 3 9.5M14.5 8 13 6.5M14.5 8 13 9.5" />
      </Icon>
    ),
  },
];

const UNAVAILABLE_MESSAGES: Record<UnavailableReason, string> = {
  image_not_found: "The image for this study is not available.",
  failed: "The viewer could not load the image.",
};

interface StudyViewerProps {
  imageUrl: string;
  /** Short line under the title, such as the date of the study. */
  caption?: string;
  /** Marks the plate when the model asked for expert review. */
  flagged?: boolean;
}

/** MRI viewer on a relief plate, with window/level, zoom, pan and reset. */
export function StudyViewer({
  imageUrl,
  caption,
  flagged = false,
}: StudyViewerProps) {
  const { containerRef, state, zoom, setTool, zoomBy, reset } =
    useViewer(imageUrl);
  const [active, setActive] = useState<ViewerTool>(DEFAULT_TOOL);
  const unavailable = state.status === "unavailable";
  const hint = TOOLS.find(({ tool }) => tool === active)?.hint;

  function choose(tool: ViewerTool) {
    setActive(tool);
    setTool(tool);
  }

  return (
    <div className={styles.frame}>
      <section
        aria-label="MRI viewer"
        className={`${styles.plate} ${flagged ? styles.flagged : ""}`}
      >
        <header className={styles.head}>
          <span className={styles.title}>MRI viewer</span>
          {caption && <span className={styles.caption}>{caption}</span>}
          {flagged && (
            <span className={styles.flag}>▲ Needs expert review</span>
          )}
        </header>
        <div className={styles.stage}>
          <div
            ref={containerRef}
            className={styles.canvas}
            data-tool={active}
            hidden={unavailable}
            onContextMenu={(event) => event.preventDefault()}
          />
          {state.status === "loading" && (
            <div className={styles.scan} role="status">
              <span className={styles.scanText}>Loading image…</span>
            </div>
          )}
          {unavailable && (
            <div className={styles.empty} role="status">
              <span className={styles.cross} aria-hidden="true" />
              <p>{UNAVAILABLE_MESSAGES[state.reason]}</p>
            </div>
          )}
          {state.status === "ready" && (
            <>
              <span className={styles.hint}>{hint} · Scroll to zoom</span>
              <span className={styles.scale}>
                Grayscale · no real-world scale
              </span>
            </>
          )}
        </div>
        {state.status === "ready" && (
          <div
            className={styles.toolbar}
            role="toolbar"
            aria-label="Viewer tools"
          >
            <div className={styles.group} role="group" aria-label="Tool">
              {TOOLS.map(({ tool, label, icon }) => (
                <button
                  key={tool}
                  type="button"
                  aria-pressed={active === tool}
                  className={styles.tool}
                  onClick={() => choose(tool)}
                >
                  {icon}
                  {label}
                </button>
              ))}
            </div>
            <div className={styles.group} role="group" aria-label="View">
              {active === "zoom" && (
                <>
                  <button
                    type="button"
                    aria-label="Zoom out"
                    className={`${styles.tool} ${styles.step}`}
                    onClick={() => zoomBy(1 / ZOOM_STEP)}
                  >
                    −
                  </button>
                  <button
                    type="button"
                    aria-label="Zoom in"
                    className={`${styles.tool} ${styles.step}`}
                    onClick={() => zoomBy(ZOOM_STEP)}
                  >
                    +
                  </button>
                </>
              )}
              <output aria-label="Zoom level" className={styles.readout}>
                {formatZoom(zoom)}
              </output>
              <button type="button" className={styles.tool} onClick={reset}>
                Reset view
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
