import type { ApiClient } from "./client";
import { errorFromResponse, networkError } from "./apiError";
import { parseChatStream } from "./sse";
import type {
  ChatEvent,
  ChatMessage,
  ListStudiesParams,
  Page,
  Report,
  StudyDetail,
} from "./types";

/**
 * The study in a 422 answer to an upload the classifier rejected, or `null`.
 * The backend stores a rejected study and returns it with the reason, so the
 * portal opens it like any other; other 422s (invalid fields) carry no study.
 */
async function rejectedStudy(response: Response): Promise<StudyDetail | null> {
  if (response.status !== 422) return null;
  const body: unknown = await response
    .clone()
    .json()
    .catch(() => undefined);
  const study = (body as { study?: unknown } | undefined)?.study;
  return typeof study === "object" && study !== null
    ? (study as StudyDetail)
    : null;
}

export function createHttpClient(baseUrl: string): ApiClient {
  const url = (path: string) => `${baseUrl}${path}`;

  async function request(path: string, init?: RequestInit): Promise<Response> {
    try {
      return await fetch(url(path), init);
    } catch (error) {
      if (init?.signal?.aborted) throw error;
      throw networkError();
    }
  }

  async function send(path: string, init?: RequestInit): Promise<Response> {
    const response = await request(path, init);
    if (!response.ok) throw await errorFromResponse(response);
    return response;
  }

  async function json<T>(path: string, init?: RequestInit): Promise<T> {
    return (await send(path, init)).json() as Promise<T>;
  }

  return {
    listStudies({ page = 1, pageSize = 20 }: ListStudiesParams = {}) {
      return json<Page<StudyDetail>>(
        `/api/studies?page=${page}&page_size=${pageSize}`,
      );
    },
    getStudy(id) {
      return json<StudyDetail>(`/api/studies/${id}`);
    },
    async uploadStudy(file, clinicalNotes) {
      const form = new FormData();
      form.append("file", file);
      if (clinicalNotes) form.append("clinical_notes", clinicalNotes);
      const response = await request("/api/studies", {
        method: "POST",
        body: form,
      });
      const rejected = await rejectedStudy(response);
      if (rejected) return rejected;
      if (!response.ok) throw await errorFromResponse(response);
      return response.json() as Promise<StudyDetail>;
    },
    imageUrl(studyId) {
      return url(`/api/studies/${studyId}/image`);
    },
    generateReport(studyId) {
      return json<Report>(`/api/studies/${studyId}/report`, { method: "POST" });
    },
    listChatMessages(studyId) {
      return json<ChatMessage[]>(`/api/studies/${studyId}/chat`);
    },
    async *streamChat(studyId, message, signal): AsyncGenerator<ChatEvent> {
      const response = await send(`/api/studies/${studyId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        signal,
      });
      if (!response.body) throw networkError();
      yield* parseChatStream(response.body);
    },
  };
}
