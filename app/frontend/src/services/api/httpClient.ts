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

export function createHttpClient(baseUrl: string): ApiClient {
  const url = (path: string) => `${baseUrl}${path}`;

  async function send(path: string, init?: RequestInit): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(url(path), init);
    } catch (error) {
      if (init?.signal?.aborted) throw error;
      throw networkError();
    }
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
    uploadStudy(file, clinicalNotes) {
      const form = new FormData();
      form.append("file", file);
      if (clinicalNotes) form.append("clinical_notes", clinicalNotes);
      return json<StudyDetail>("/api/studies", { method: "POST", body: form });
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
