/** Thrown when a study image cannot be fetched; `status` is the HTTP status, if any. */
export class ImageFetchError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ImageFetchError";
  }
}

/** Luminance of each RGBA pixel, so a JPG or PNG is shown as one grayscale channel. */
export function toGrayscale(rgba: Uint8ClampedArray): Uint8Array {
  const gray = new Uint8Array(rgba.length / 4);
  for (let i = 0; i < gray.length; i += 1) {
    const offset = i * 4;
    gray[i] = Math.round(
      0.299 * rgba[offset]! +
        0.587 * rgba[offset + 1]! +
        0.114 * rgba[offset + 2]!,
    );
  }
  return gray;
}

export interface WebImage {
  width: number;
  height: number;
  pixels: Uint8Array;
  canvas: HTMLCanvasElement;
}

/** Downloads a JPG or PNG and decodes it into grayscale pixels. */
export async function fetchWebImage(url: string): Promise<WebImage> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new ImageFetchError("The image could not be downloaded.");
  }
  if (!response.ok) {
    throw new ImageFetchError("The image is not available.", response.status);
  }
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d");
  if (!context) throw new ImageFetchError("The image could not be decoded.");
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  return {
    width: bitmap.width,
    height: bitmap.height,
    pixels: toGrayscale(data),
    canvas,
  };
}
