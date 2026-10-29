/** @jest-environment node */
import { fetchWebImage, ImageFetchError, toGrayscale } from "./webImage";

describe("toGrayscale", () => {
  it("converts each RGBA pixel to its luminance", () => {
    const rgba = new Uint8ClampedArray([
      255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255,
    ]);
    expect(Array.from(toGrayscale(rgba))).toEqual([255, 0, 76, 150]);
  });

  it("returns one value per pixel", () => {
    expect(toGrayscale(new Uint8ClampedArray(4 * 6))).toHaveLength(6);
  });
});

describe("fetchWebImage", () => {
  afterEach(() => jest.restoreAllMocks());

  it("reports the status of a missing image", async () => {
    global.fetch = jest.fn(async () => new Response("", { status: 404 }));
    await expect(fetchWebImage("http://api.test/x")).rejects.toMatchObject({
      name: "ImageFetchError",
      status: 404,
    });
  });

  it("reports a download that never reached the server", async () => {
    global.fetch = jest.fn(async () => {
      throw new TypeError("fetch failed");
    });
    const failure = fetchWebImage("http://api.test/x");
    await expect(failure).rejects.toBeInstanceOf(ImageFetchError);
    await expect(failure).rejects.toMatchObject({ status: undefined });
  });
});
