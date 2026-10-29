/** @jest-environment node */
import { createHttpClient } from "./httpClient";
import type { ChatEvent } from "./types";

const BASE = "http://api.test";

function mockFetch(response: Response | Error) {
  const fetchMock = jest.fn(async () => {
    if (response instanceof Error) throw response;
    return response;
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

function sse(...events: object[]): Response {
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  return new Response(body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("http client", () => {
  const client = createHttpClient(BASE);

  it("builds the list URL from the base URL and paging", async () => {
    const fetchMock = mockFetch(
      Response.json({ items: [], page: 2, page_size: 5, total: 0 }),
    );
    await client.listStudies({ page: 2, pageSize: 5 });
    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/api/studies?page=2&page_size=5`,
      undefined,
    );
  });

  it("normalizes a backend error body", async () => {
    mockFetch(
      Response.json(
        { code: "STUDY_NOT_FOUND", message: "Missing." },
        { status: 404 },
      ),
    );
    await expect(client.getStudy("x")).rejects.toMatchObject({
      code: "STUDY_NOT_FOUND",
      message: "Missing.",
      status: 404,
    });
  });

  it("falls back to UNKNOWN_ERROR when the error body is not {code, message}", async () => {
    mockFetch(new Response("<html>bad gateway</html>", { status: 502 }));
    await expect(client.getStudy("x")).rejects.toMatchObject({
      code: "UNKNOWN_ERROR",
      status: 502,
    });
  });

  it("maps an unreachable server to NETWORK_ERROR", async () => {
    mockFetch(new TypeError("fetch failed"));
    await expect(client.getStudy("x")).rejects.toMatchObject({
      code: "NETWORK_ERROR",
    });
  });

  it("builds the image URL", () => {
    expect(client.imageUrl("abc")).toBe(`${BASE}/api/studies/abc/image`);
  });

  it("sends the upload as multipart form data", async () => {
    const fetchMock = mockFetch(Response.json({ id: "1" }, { status: 201 }));
    await client.uploadStudy(new Blob(["x"]) as File, "a note");
    const init = (fetchMock.mock.calls[0] as unknown[])[1] as RequestInit;
    const form = init.body as FormData;
    expect(init.method).toBe("POST");
    expect(form.get("clinical_notes")).toBe("a note");
    expect(form.get("file")).toBeInstanceOf(Blob);
  });

  it("parses a streamed chat answer into events", async () => {
    mockFetch(
      sse(
        { type: "token", text: "Hel" },
        { type: "token", text: "lo" },
        {
          type: "done",
          message: {
            id: "m",
            role: "assistant",
            content: "Hello",
            created_at: "t",
          },
        },
      ),
    );
    const events: ChatEvent[] = [];
    for await (const event of client.streamChat("s", "hi")) events.push(event);
    expect(events.map((event) => event.type)).toEqual([
      "token",
      "token",
      "done",
    ]);
  });

  it("rejects a chat stream refused with 503", async () => {
    mockFetch(
      Response.json(
        { code: "LLM_UNAVAILABLE", message: "No key." },
        { status: 503 },
      ),
    );
    const read = async () => {
      for await (const event of client.streamChat("s", "hi")) void event;
    };
    await expect(read()).rejects.toMatchObject({
      code: "LLM_UNAVAILABLE",
      status: 503,
    });
  });

  it("rethrows an aborted request instead of reporting a network error", async () => {
    const controller = new AbortController();
    controller.abort();
    mockFetch(Object.assign(new Error("Aborted"), { name: "AbortError" }));
    const read = async () => {
      for await (const event of client.streamChat("s", "hi", controller.signal))
        void event;
    };
    await expect(read()).rejects.toMatchObject({ name: "AbortError" });
  });
});
