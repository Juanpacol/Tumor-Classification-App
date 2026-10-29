/** @jest-environment node */
import { parseChatStream } from "./sse";
import type { ChatEvent } from "./types";

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(body: ReadableStream<Uint8Array>): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const event of parseChatStream(body)) events.push(event);
  return events;
}

describe("parseChatStream", () => {
  it("joins an event split across chunks", async () => {
    const events = await collect(
      streamOf('data: {"type":"tok', 'en","text":"hi"}\n\n'),
    );
    expect(events).toEqual([{ type: "token", text: "hi" }]);
  });

  it("reads several events from one chunk", async () => {
    const events = await collect(
      streamOf(
        'data: {"type":"token","text":"a"}\n\ndata: {"type":"token","text":"b"}\n\n',
      ),
    );
    expect(events).toHaveLength(2);
  });

  it("ignores blocks without data, such as comments", async () => {
    const events = await collect(
      streamOf(': keep-alive\n\ndata: {"type":"token","text":"a"}\n\n'),
    );
    expect(events).toEqual([{ type: "token", text: "a" }]);
  });

  it("drops an unfinished trailing event", async () => {
    const events = await collect(
      streamOf('data: {"type":"token","text":"a"}\n\ndata: {"type":"to'),
    );
    expect(events).toHaveLength(1);
  });
});
