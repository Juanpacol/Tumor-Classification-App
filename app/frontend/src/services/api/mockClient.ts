import type { ApiClient } from "./client";
import { ApiError } from "./apiError";
import { SAMPLE_REPORT, SAMPLE_STUDIES } from "./sampleData";
import type { ChatEvent, ChatMessage, StudyDetail } from "./types";

export interface MockOptions {
  /** Pauses between streamed tokens, in milliseconds. */
  tokenDelayMs?: number;
  /** Time a report takes to generate, in milliseconds. */
  reportDelayMs?: number;
}

/** Studies whose report cannot be generated, to exercise the 503 path. */
const REPORT_UNAVAILABLE = new Set(["study-report-unavailable"]);

const ANSWER =
  "This is a sample answer from the mock assistant. It is not a diagnosis.";

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted)
      return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/**
 * In-memory client with a sample study in every state. A chat message that
 * contains `[error]` makes the stream fail, and one that contains
 * `[interrupt]` ends it without a `done` event.
 */
export function createMockClient({
  tokenDelayMs = 0,
  reportDelayMs = 0,
}: MockOptions = {}): ApiClient {
  const studies = new Map<string, StudyDetail>(
    SAMPLE_STUDIES.map((study) => [study.id, study]),
  );

  function find(id: string): StudyDetail {
    const study = studies.get(id);
    if (!study) {
      throw new ApiError("STUDY_NOT_FOUND", "The study does not exist.", 404);
    }
    return study;
  }

  return {
    async listStudies({ page = 1, pageSize = 20 } = {}) {
      const all = [...studies.values()].sort((a, b) =>
        b.created_at.localeCompare(a.created_at),
      );
      const start = (page - 1) * pageSize;
      return {
        items: all.slice(start, start + pageSize),
        page,
        page_size: pageSize,
        total: all.length,
      };
    },
    async getStudy(id) {
      return find(id);
    },
    async uploadStudy(file, clinicalNotes) {
      if (!/\.(jpe?g|png)$/i.test(file.name)) {
        throw new ApiError(
          "UNSUPPORTED_FORMAT",
          "Use a JPG or PNG image.",
          415,
        );
      }
      const study: StudyDetail = {
        id: `mock-${studies.size + 1}`,
        status: "processing",
        created_at: new Date().toISOString(),
        clinical_notes: clinicalNotes ?? null,
        rejection_reason: null,
        prediction: null,
        report: null,
      };
      studies.set(study.id, study);
      return study;
    },
    imageUrl(studyId) {
      return `/mock/${studyId}.png`;
    },
    async generateReport(studyId) {
      const study = find(studyId);
      if (study.status !== "completed") {
        throw new ApiError(
          "STUDY_NOT_COMPLETED",
          "The study has no result to report on yet.",
          409,
        );
      }
      await wait(reportDelayMs);
      if (REPORT_UNAVAILABLE.has(studyId)) {
        throw new ApiError(
          "LLM_UNAVAILABLE",
          "The report service is not available.",
          503,
        );
      }
      const report = {
        ...(study.report ?? SAMPLE_REPORT),
        created_at: new Date().toISOString(),
      };
      studies.set(studyId, { ...study, report });
      return report;
    },
    async listChatMessages(studyId) {
      find(studyId);
      return [];
    },
    async *streamChat(studyId, message, signal): AsyncGenerator<ChatEvent> {
      find(studyId);
      const words = ANSWER.split(" ");
      for (const [index, word] of words.entries()) {
        await wait(tokenDelayMs, signal);
        if (message.includes("[error]") && index === 2) {
          yield {
            type: "error",
            code: "LLM_UNAVAILABLE",
            message: "Stream failed.",
          };
          return;
        }
        if (message.includes("[interrupt]") && index === 2) return;
        yield { type: "token", text: index === 0 ? word : ` ${word}` };
      }
      const reply: ChatMessage = {
        id: `msg-${studyId}`,
        role: "assistant",
        content: ANSWER,
        created_at: new Date().toISOString(),
      };
      yield { type: "done", message: reply };
    },
  };
}
