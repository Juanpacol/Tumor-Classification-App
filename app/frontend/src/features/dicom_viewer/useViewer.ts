"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  createViewer as defaultCreateViewer,
  ViewerUnavailableError,
  type CreateViewer,
  type UnavailableReason,
  type Viewer,
  type ViewerTool,
} from "./viewerEngine";

export type ViewerState =
  | { status: "loading" }
  | { status: "ready" }
  | { status: "unavailable"; reason: UnavailableReason };

/**
 * Creates the viewer for `imageUrl` inside the element behind `containerRef`
 * and destroys it when the image changes or the component unmounts, so that
 * navigating between studies does not leak WebGL contexts.
 */
export function useViewer(
  imageUrl: string,
  createViewer: CreateViewer = defaultCreateViewer,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const [state, setState] = useState<ViewerState>({ status: "loading" });

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    let current = true;
    setState({ status: "loading" });
    createViewer(element, imageUrl)
      .then((viewer) => {
        if (!current) {
          viewer.destroy();
          return;
        }
        viewerRef.current = viewer;
        setState({ status: "ready" });
      })
      .catch((error: unknown) => {
        if (!current) return;
        setState({
          status: "unavailable",
          reason:
            error instanceof ViewerUnavailableError ? error.reason : "failed",
        });
      });
    return () => {
      current = false;
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, [imageUrl, createViewer]);

  const setTool = useCallback(
    (tool: ViewerTool) => viewerRef.current?.setTool(tool),
    [],
  );
  const reset = useCallback(() => viewerRef.current?.reset(), []);

  return { containerRef, state, setTool, reset };
}
