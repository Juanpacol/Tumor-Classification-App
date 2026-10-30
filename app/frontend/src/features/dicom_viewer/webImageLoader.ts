import type { Types } from "@cornerstonejs/core";
import { fetchWebImage, type WebImage } from "./webImage";

export const WEB_SCHEME = "web";

export function webImageId(url: string): string {
  return `${WEB_SCHEME}:${url}`;
}

type Core = typeof import("@cornerstonejs/core");

const decoded = new Map<string, WebImage>();

/**
 * Downloads and decodes the image ahead of Cornerstone3D. Its errors keep
 * their own type, which a loader called through Cornerstone3D cannot promise,
 * and the loader below reuses the result instead of downloading it twice.
 */
export async function preloadWebImage(url: string): Promise<void> {
  decoded.set(url, await fetchWebImage(url));
}

/**
 * Cornerstone3D reads DICOM out of the box but has no loader for JPG or PNG,
 * which is what the portal uploads. This one registers a `web:` image id
 * scheme that serves the picture as an 8-bit grayscale image with no real
 * spatial calibration.
 */
export function registerWebImageLoader(core: Core): void {
  core.imageLoader.registerImageLoader(WEB_SCHEME, (imageId) => ({
    promise: loadWebImage(imageId),
  }));
  core.metaData.addProvider((type, imageId) => webMetadata(type, imageId));
}

async function loadWebImage(imageId: string): Promise<Types.IImage> {
  const url = imageId.slice(WEB_SCHEME.length + 1);
  const image = decoded.get(url) ?? (await fetchWebImage(url));
  decoded.delete(url);
  const { width, height, pixels, canvas } = image;
  return {
    imageId,
    minPixelValue: 0,
    maxPixelValue: 255,
    slope: 1,
    intercept: 0,
    windowCenter: 128,
    windowWidth: 256,
    voiLUTFunction: "LINEAR",
    getPixelData: () => pixels,
    getCanvas: () => canvas,
    rows: height,
    columns: width,
    height,
    width,
    color: false,
    rgba: false,
    numComps: 1,
    columnPixelSpacing: 1,
    rowPixelSpacing: 1,
    invert: false,
    photometricInterpretation: "MONOCHROME2",
    sizeInBytes: pixels.byteLength,
  };
}

function webMetadata(type: string, imageId: string): unknown {
  if (!imageId.startsWith(`${WEB_SCHEME}:`)) return undefined;
  if (type === "imagePlaneModule") {
    return {
      rowCosines: [1, 0, 0],
      columnCosines: [0, 1, 0],
      imagePositionPatient: [0, 0, 0],
      pixelSpacing: [1, 1],
      rowPixelSpacing: 1,
      columnPixelSpacing: 1,
    };
  }
  if (type === "generalSeriesModule") {
    return { modality: "OT" };
  }
  if (type === "imagePixelModule") {
    return {
      samplesPerPixel: 1,
      photometricInterpretation: "MONOCHROME2",
      bitsAllocated: 8,
      bitsStored: 8,
      highBit: 7,
      pixelRepresentation: 0,
    };
  }
  return undefined;
}
