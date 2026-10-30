import { formatZoom, MAX_ZOOM, MIN_ZOOM, nextZoom, ZOOM_STEP } from "./zoom";

describe("nextZoom", () => {
  it("multiplies the zoom by the factor", () => {
    expect(nextZoom(1, ZOOM_STEP)).toBeCloseTo(1.25);
    expect(nextZoom(2, 1 / ZOOM_STEP)).toBeCloseTo(1.6);
  });

  it("never goes past the limits", () => {
    expect(nextZoom(MAX_ZOOM, ZOOM_STEP)).toBe(MAX_ZOOM);
    expect(nextZoom(MIN_ZOOM, 1 / ZOOM_STEP)).toBe(MIN_ZOOM);
  });
});

describe("formatZoom", () => {
  it("shows a percentage", () => {
    expect(formatZoom(1)).toBe("100%");
    expect(formatZoom(1.256)).toBe("126%");
  });
});
