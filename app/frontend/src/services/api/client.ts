import type {
  ChatEvent,
  ChatMessage,
  ListStudiesParams,
  Page,
  Report,
  StudyDetail,
} from "./types";

/**
 * What the portal needs from the backend. Failures reject with `ApiError`.
 * The HTTP and mock implementations are interchangeable.
 */
export interface ApiClient {
  listStudies(params?: ListStudiesParams): Promise<Page<StudyDetail>>;
  getStudy(id: string): Promise<StudyDetail>;
  uploadStudy(file: File, clinicalNotes?: string): Promise<StudyDetail>;
  imageUrl(studyId: string): string;
  generateReport(studyId: string): Promise<Report>;
  listChatMessages(studyId: string): Promise<ChatMessage[]>;
  /** Streams the assistant's answer; aborting `signal` stops it. */
  streamChat(
    studyId: string,
    message: string,
    signal?: AbortSignal,
  ): AsyncIterable<ChatEvent>;
}
