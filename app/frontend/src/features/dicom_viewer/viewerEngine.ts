import type { Types } from "@cornerstonejs/core";
import { ImageFetchError } from "./webImage";
import { webImageId, registerWebImageLoader } from "./webImageLoader";

export type ViewerTool = "zoom" | "pan" | "windowLevel";

export type UnavailableReason = "no_webgl" | "image_not_found" | "failed";

/** The viewer cannot show the image; `reason` picks the message the reader sees. */
export class ViewerUnavailableError extends Error {
  constructor(readonly reason: UnavailableReason) {
    super(reason);
    this.name = "ViewerUnavailableError";
  }
}

export interface Viewer {
  setTool(tool: ViewerTool): void;
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

let counter = 0;
let ready: Promise<void> | undefined;

function webglAvailable(): boolean {
  const canvas = document.createElement("canvas");
  return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
}

function initialize(
  core: typeof import("@cornerstonejs/core"),
  tools: typeof import("@cornerstonejs/tools"),
): Promise<void> {
  ready ??= (async () => {
    await core.init();
    tools.init();
    registerWebImageLoader(core);
    tools.addTool(tools.ZoomTool);
    tools.addTool(tools.PanTool);
    tools.addTool(tools.WindowLevelTool);
  })();
  return ready;
}

/** Shows one JPG or PNG in a Cornerstone3D stack viewport inside `element`. */
export const createViewer: CreateViewer = async (element, imageUrl) => {
  if (!webglAvailable()) throw new ViewerUnavailableError("no_webgl");

  const [core, tools] = await Promise.all([
    import("@cornerstonejs/core"),
    import("@cornerstonejs/tools"),
  ]);
  await initialize(core, tools);

  counter += 1;
  const viewportId = `viewport-${counter}`;
  const engineId = `engine-${counter}`;
  const toolGroupId = `tools-${counter}`;
  const imageId = webImageId(imageUrl);
  let engine: InstanceType<typeof core.RenderingEngine> | undefined;

  function destroy() {
    tools.ToolGroupManager.destroyToolGroup(toolGroupId);
    engine?.destroy();
    core.cache.removeImageLoadObject(imageId);
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
      reset() {
        viewport.resetProperties();
        viewport.resetCamera();
        viewport.render();
      },
      destroy,
    };
  } catch (error) {
    destroy();
    if (error instanceof ImageFetchError && error.status === 404) {
      throw new ViewerUnavailableError("image_not_found");
    }
    throw new ViewerUnavailableError("failed");
  }
};
