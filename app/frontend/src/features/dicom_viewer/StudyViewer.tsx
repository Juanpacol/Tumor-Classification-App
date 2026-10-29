"use client";

import { useState } from "react";
import { Button } from "@/features/portal/components";
import { Card } from "@/features/portal/components";
import styles from "./StudyViewer.module.css";
import type { UnavailableReason, ViewerTool } from "./viewerEngine";
import { DEFAULT_TOOL } from "./viewerEngine";
import { useViewer } from "./useViewer";

const TOOLS: { tool: ViewerTool; label: string }[] = [
  { tool: "windowLevel", label: "Window/level" },
  { tool: "zoom", label: "Zoom" },
  { tool: "pan", label: "Pan" },
];

const UNAVAILABLE_MESSAGES: Record<UnavailableReason, string> = {
  no_webgl:
    "This browser cannot show the image viewer because WebGL is not available. The rest of the study is still available.",
  image_not_found: "The image for this study is not available.",
  failed: "The viewer could not load the image.",
};

/** MRI viewer on the dark plate, with zoom, pan, window/level and reset. */
export function StudyViewer({ imageUrl }: { imageUrl: string }) {
  const { containerRef, state, setTool, reset } = useViewer(imageUrl);
  const [active, setActive] = useState<ViewerTool>(DEFAULT_TOOL);
  const unavailable = state.status === "unavailable";

  function choose(tool: ViewerTool) {
    setActive(tool);
    setTool(tool);
  }

  return (
    <Card tone="plate" aria-label="MRI viewer" className={styles.plate}>
      {unavailable && (
        <p className={styles.placeholder} role="status">
          {UNAVAILABLE_MESSAGES[state.reason]}
        </p>
      )}
      <div
        ref={containerRef}
        className={styles.canvas}
        hidden={unavailable}
        onContextMenu={(event) => event.preventDefault()}
      />
      {state.status === "loading" && (
        <p className={styles.loading} role="status">
          Loading image…
        </p>
      )}
      {state.status === "ready" && (
        <div
          className={styles.toolbar}
          role="toolbar"
          aria-label="Viewer tools"
        >
          {TOOLS.map(({ tool, label }) => (
            <Button
              key={tool}
              variant={active === tool ? "primary" : "quiet"}
              aria-pressed={active === tool}
              className={styles.tool}
              onClick={() => choose(tool)}
            >
              {label}
            </Button>
          ))}
          <Button variant="quiet" className={styles.tool} onClick={reset}>
            Reset view
          </Button>
        </div>
      )}
    </Card>
  );
}
