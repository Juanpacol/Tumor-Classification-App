import type { ApiClient } from "./client";
import { createHttpClient } from "./httpClient";
import { createMockClient } from "./mockClient";

export type { ApiClient } from "./client";
export { ApiError } from "./apiError";
export * from "./types";

/** `NEXT_PUBLIC_API_MODE=mock` serves sample data; anything else calls the backend. */
export function createApiClient(
  env: { mode?: string; baseUrl?: string } = {
    mode: process.env.NEXT_PUBLIC_API_MODE,
    baseUrl: process.env.NEXT_PUBLIC_API_URL,
  },
): ApiClient {
  if (env.mode === "mock") return createMockClient();
  return createHttpClient(env.baseUrl ?? "http://localhost:8000");
}
