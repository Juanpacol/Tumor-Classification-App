export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 20;
/** One notch of the mouse wheel and one press of the zoom buttons. */
export const ZOOM_STEP = 1.25;

/** The zoom after multiplying by `factor`, kept inside what the viewer can show. */
export function nextZoom(current: number, factor: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current * factor));
}

export function formatZoom(zoom: number): string {
  return `${Math.round(zoom * 100)}%`;
}
