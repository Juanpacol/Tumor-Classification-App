import { ApiError } from "./apiError";
import { createMockClient } from "./mockClient";
import type { ChatEvent } from "./types";

async function collect(stream: AsyncIterable<ChatEvent>): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe("mock client", () => {
  const client = createMockClient();

  it("lists studies newest first with pagination", async () => {
    const first = await client.listStudies({ page: 1, pageSize: 2 });
    expect(first.items).toHaveLength(2);
    expect(first.total).toBeGreaterThan(2);
    expect(first.items[0]!.created_at >= first.items[1]!.created_at).toBe(true);
    const second = await client.listStudies({ page: 2, pageSize: 2 });
    expect(second.items[0]!.id).not.toBe(first.items[0]!.id);
  });

  it("covers every study status", async () => {
    const { items } = await client.listStudies({ pageSize: 50 });
    const statuses = new Set(items.map((study) => study.status));
    expect(statuses).toEqual(
      new Set(["completed", "processing", "rejected", "failed"]),
    );
  });

  it("includes a low-confidence study that needs expert triage", async () => {
    const study = await client.getStudy("study-uncertain");
    expect(study.prediction?.needs_expert_triage).toBe(true);
  });

  it("rejects an unknown study with a normalized error", async () => {
    await expect(client.getStudy("nope")).rejects.toMatchObject({
      code: "STUDY_NOT_FOUND",
      status: 404,
    });
  });

  it("rejects an unsupported upload and accepts a PNG", async () => {
    const bad = new File(["x"], "scan.pdf");
    await expect(client.uploadStudy(bad)).rejects.toBeInstanceOf(ApiError);
    const study = await client.uploadStudy(new File(["x"], "scan.png"), "note");
    expect(study.clinical_notes).toBe("note");
  });

  it("answers 503 when a study has no report to return", async () => {
    await expect(
      client.generateReport("study-no-report"),
    ).rejects.toMatchObject({ code: "LLM_UNAVAILABLE" });
  });

  it("streams tokens and ends with the full message", async () => {
    const events = await collect(client.streamChat("study-confident", "hello"));
    const text = events
      .filter((event) => event.type === "token")
      .map((event) => (event as { text: string }).text)
      .join("");
    const last = events.at(-1);
    expect(last?.type).toBe("done");
    expect(last?.type === "done" && last.message.content).toBe(text);
  });

  it("can simulate an error event", async () => {
    const events = await collect(
      client.streamChat("study-confident", "[error]"),
    );
    expect(events.at(-1)).toMatchObject({ type: "error" });
  });

  it("can simulate an interruption without a done event", async () => {
    const events = await collect(
      client.streamChat("study-confident", "[interrupt]"),
    );
    expect(events.some((event) => event.type === "done")).toBe(false);
  });

  it("stops streaming when aborted", async () => {
    const slow = createMockClient({ tokenDelayMs: 50 });
    const controller = new AbortController();
    const stream = slow.streamChat("study-confident", "hi", controller.signal);
    const pending = collect(stream);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});
