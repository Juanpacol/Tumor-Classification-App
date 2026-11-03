import { ApiError } from "@/services/api";

/** Largest image the backend accepts. */
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
/** Longest clinical notes the backend accepts. */
export const MAX_NOTES_LENGTH = 2000;

const ACCEPTED_NAME = /\.(jpe?g|png)$/i;

/** Why `file` cannot be uploaded, or `null` when it can. */
export function fileProblem(file: File): string | null {
  if (!ACCEPTED_NAME.test(file.name)) {
    return "Use a JPG or PNG image (.jpg, .jpeg or .png).";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "The image is larger than 20 MB.";
  }
  return null;
}

const MESSAGE_BY_STATUS: Record<number, string> = {
  413: "The image is larger than 20 MB.",
  415: "The server could not read this file as a JPG or PNG image.",
  422: `The clinical notes are longer than ${MAX_NOTES_LENGTH} characters.`,
  500: "The server failed while processing the image. Try again.",
  503: "The classification model is not available right now. Try again in a few minutes.",
};

/** A plain-language message for a failed upload. */
export function uploadErrorMessage(error: unknown): string {
  if (error instanceof ApiError && error.status !== undefined) {
    return MESSAGE_BY_STATUS[error.status] ?? error.message;
  }
  return error instanceof Error ? error.message : "Something went wrong.";
}
