import { createViewer } from "./viewerEngine";

describe("createViewer", () => {
  afterEach(() => jest.restoreAllMocks());

  it("reports missing WebGL support without loading Cornerstone", async () => {
    jest.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    await expect(
      createViewer(document.createElement("div"), "http://api.test/image"),
    ).rejects.toMatchObject({ reason: "no_webgl" });
  });
});
