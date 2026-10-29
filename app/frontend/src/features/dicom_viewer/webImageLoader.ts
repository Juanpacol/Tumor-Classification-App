import type { Types } from "@cornerstonejs/core";
import { fetchWebImage } from "./webImage";

export const WEB_SCHEME = "web";

export function webImageId(url: string): string {
  return `${WEB_SCHEME}:${url}`;
}

type Core = typeof import("@cornerstonejs/core");

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
  const { width, height, pixels, canvas } = await fetchWebImage(url);
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
