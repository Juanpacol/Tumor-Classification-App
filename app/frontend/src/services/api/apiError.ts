/** Every failure the portal handles, whatever its source, as `{code, message}`. */
export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const NETWORK_ERROR = "NETWORK_ERROR";
export const UNKNOWN_ERROR = "UNKNOWN_ERROR";

function isErrorBody(body: unknown): body is { code: string; message: string } {
  return (
    typeof body === "object" &&
    body !== null &&
    typeof (body as { code?: unknown }).code === "string" &&
    typeof (body as { message?: unknown }).message === "string"
  );
}

/** Reads a failed response; the backend answers `{code, message}`. */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  if (isErrorBody(body)) {
    return new ApiError(body.code, body.message, response.status);
  }
  return new ApiError(
    UNKNOWN_ERROR,
    `The server answered ${response.status}.`,
    response.status,
  );
}

export function networkError(): ApiError {
  return new ApiError(NETWORK_ERROR, "The server could not be reached.");
}
