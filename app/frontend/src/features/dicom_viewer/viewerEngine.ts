import type { Types } from "@cornerstonejs/core";
import { ImageFetchError } from "./webImage";
import { nextZoom, ZOOM_STEP } from "./zoom";
import {
  preloadWebImage,
  registerWebImageLoader,
  webImageId,
} from "./webImageLoader";

export type ViewerTool = "zoom" | "pan" | "windowLevel";

export type UnavailableReason = "image_not_found" | "failed";

/** The viewer cannot show the image; `reason` picks the message the reader sees. */
export class ViewerUnavailableError extends Error {
  constructor(readonly reason: UnavailableReason) {
    super(reason);
    this.name = "ViewerUnavailableError";
  }
}

export interface Viewer {
  setTool(tool: ViewerTool): void;
  /** Multiplies the zoom, keeping it inside the limits. */
  zoomBy(factor: number): void;
  /** Calls `listener` with the zoom (1 is the whole image) each time it changes. */
  onZoom(listener: (zoom: number) => void): () => void;
  reset(): void;
  destroy(): void;
}

export type CreateViewer = (
  element: HTMLDivElement,
  imageUrl: string,
) => Promise<Viewer>;

const TOOL_NAMES: Record<ViewerTool, string> = {
  zoom: "Zoom",
  pan: "Pan",
  windowLevel: "WindowLevel",
};

export const DEFAULT_TOOL: ViewerTool = "windowLevel";

/** The web image loader serves 8-bit grayscale, so the whole range is 0 to 255. */
const FULL_RANGE = { lower: 0, upper: 255 };

let counter = 0;
let ready: Promise<void> | undefined;

function initialize(
  core: typeof import("@cornerstonejs/core"),
  tools: typeof import("@cornerstonejs/tools"),
): Promise<void> {
  ready ??= (async () => {
    await core.init();
    tools.init();
    registerWebImageLoader(core);
    [tools.ZoomTool, tools.PanTool, tools.WindowLevelTool].forEach((tool) => {
      if (!tools.state.tools[tool.toolName]) tools.addTool(tool);
    });
  })();
  return ready;
}

/**
 * Cornerstone3D Tools logs an error from its annotation renderer while an
 * element is disabled. It comes from its own teardown and changes nothing, but
 * it fills the console (and the development overlay) on every study, so it is
 * muted for the duration of the teardown only.
 */
function withoutTeardownNoise(run: () => void) {
  const original = console.error;
  console.error = (...args: unknown[]) => {
    const first = String(args[0] ?? "");
    if (
      first.startsWith("error in event listener of type:") &&
      first.includes("CORNERSTONE_ELEMENT_DISABLED")
    ) {
      return;
    }
    original(...args);
  };
  try {
    run();
  } finally {
    console.error = original;
  }
}

function unavailable(error: unknown): ViewerUnavailableError {
  console.error("The viewer could not show the image.", error);
  if (error instanceof ImageFetchError && error.status === 404) {
    return new ViewerUnavailableError("image_not_found");
  }
  return new ViewerUnavailableError("failed");
}

/**
 * Shows one JPG or PNG in a Cornerstone3D stack viewport inside `element`.
 * Without WebGL, Cornerstone3D renders on the CPU, so the viewer still works.
 */
export const createViewer: CreateViewer = async (element, imageUrl) => {
  let core: typeof import("@cornerstonejs/core");
  let tools: typeof import("@cornerstonejs/tools");
  try {
    await preloadWebImage(imageUrl);
    [core, tools] = await Promise.all([
      import("@cornerstonejs/core"),
      import("@cornerstonejs/tools"),
    ]);
    await initialize(core, tools);
  } catch (error) {
    throw unavailable(error);
  }

  counter += 1;
  const viewportId = `viewport-${counter}`;
  const engineId = `engine-${counter}`;
  const toolGroupId = `tools-${counter}`;
  const imageId = webImageId(imageUrl);
  let engine: InstanceType<typeof core.RenderingEngine> | undefined;

  const cleanups: (() => void)[] = [];
  let destroyed = false;
  function destroy() {
    if (destroyed) return;
    destroyed = true;
    cleanups.forEach((cleanup) => cleanup());
    // The engine goes first: disabling its viewports tells the tools package,
    // which still needs the tool group to exist.
    withoutTeardownNoise(() => {
      engine?.destroy();
      tools.ToolGroupManager.destroyToolGroup(toolGroupId);
    });
    try {
      core.cache.removeImageLoadObject(imageId);
    } catch {
      // The image was never cached, so there is nothing to release.
    }
  }

  try {
    await core.imageLoader.loadAndCacheImage(imageId);
    engine = new core.RenderingEngine(engineId);
    engine.enableElement({
      viewportId,
      type: core.Enums.ViewportType.STACK,
      element,
    });
    const viewport = engine.getViewport(viewportId) as Types.IStackViewport;
    await viewport.setStack([imageId]);

    const toolGroup = tools.ToolGroupManager.createToolGroup(toolGroupId)!;
    Object.values(TOOL_NAMES).forEach((name) => toolGroup.addTool(name));
    toolGroup.addViewport(viewportId, engineId);
    activate(DEFAULT_TOOL);
    viewport.render();

    // Cornerstone3D reads the zoom differently by renderer: on the GPU it is a
    // ratio to the fitted view, on the CPU the absolute scale. Both are brought
    // to "1 is the whole image in view" using the value right after fitting.
    const cpu = core.getShouldUseCPURendering();
    const fit = viewport.getZoom() || 1;
    const currentZoom = () => viewport.getZoom() / fit;
    const zoomListeners = new Set<(zoom: number) => void>();
    const announceZoom = () =>
      zoomListeners.forEach((listener) => listener(currentZoom()));
    element.addEventListener(core.Enums.Events.CAMERA_MODIFIED, announceZoom);
    cleanups.push(() =>
      element.removeEventListener(
        core.Enums.Events.CAMERA_MODIFIED,
        announceZoom,
      ),
    );

    function zoomBy(factor: number) {
      const zoom = nextZoom(currentZoom(), factor);
      viewport.setZoom(cpu ? zoom * fit : zoom);
      viewport.render();
      announceZoom();
    }

    // The zoom tool of Cornerstone3D only follows a drag, so the wheel is ours.
    function onWheel(event: WheelEvent) {
      event.preventDefault();
      zoomBy(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
    }
    element.addEventListener("wheel", onWheel, { passive: false });
    cleanups.push(() => element.removeEventListener("wheel", onWheel));

    function activate(tool: ViewerTool) {
      Object.entries(TOOL_NAMES).forEach(([key, name]) => {
        if (key === tool) {
          toolGroup.setToolActive(name, {
            bindings: [{ mouseButton: tools.Enums.MouseBindings.Primary }],
          });
        } else {
          toolGroup.setToolPassive(name);
        }
      });
    }

    return {
      setTool: activate,
      zoomBy,
      onZoom(listener) {
        zoomListeners.add(listener);
        return () => zoomListeners.delete(listener);
      },
      reset() {
        // resetProperties() throws on the CPU renderer, which has no image
        // actor, so the contrast goes back to the whole range directly.
        viewport.setProperties({ voiRange: FULL_RANGE });
        viewport.resetCamera();
        viewport.render();
        announceZoom();
      },
      destroy,
    };
  } catch (error) {
    destroy();
    throw unavailable(error);
  }
};
