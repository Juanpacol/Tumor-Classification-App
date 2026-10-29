import type { ChatMessage } from "@/services/api";
import {
  chatReducer,
  initialChatState,
  MAX_MESSAGE_LENGTH,
  type ChatAction,
  type ChatState,
} from "./chatReducer";

function message(
  id: string,
  role: "user" | "assistant",
  content: string,
): ChatMessage {
  return { id, role, content, created_at: "2026-10-28T10:00:00Z" };
}

function run(
  actions: ChatAction[],
  from: ChatState = initialChatState,
): ChatState {
  return actions.reduce(chatReducer, from);
}

const ready = run([{ type: "historyLoaded", messages: [] }]);
const streaming = run([{ type: "send", sendId: 1, text: "hello" }], ready);
const failed = run(
  [{ type: "fail", sendId: 1, message: "Stream failed.", unavailable: false }],
  streaming,
);

describe("chatReducer history", () => {
  it("loads the earlier messages as finished ones", () => {
    const state = run([
      {
        type: "historyLoaded",
        messages: [
          message("1", "user", "hi"),
          message("2", "assistant", "hello"),
        ],
      },
    ]);
    expect(state.phase).toBe("ready");
    expect(state.items.map((item) => [item.role, item.status])).toEqual([
      ["user", "done"],
      ["assistant", "done"],
    ]);
  });

  it("ignores a second history once the chat is ready", () => {
    expect(
      run(
        [{ type: "historyLoaded", messages: [message("1", "user", "x")] }],
        ready,
      ),
    ).toBe(ready);
  });

  it("disables the chat when the service is unavailable", () => {
    const state = run([
      { type: "historyFailed", message: "No key.", unavailable: true },
    ]);
    expect(state).toMatchObject({ phase: "disabled", notice: "No key." });
  });

  it("stays usable when the history fails for another reason", () => {
    const state = run([
      { type: "historyFailed", message: "Boom.", unavailable: false },
    ]);
    expect(state).toMatchObject({ phase: "ready", notice: "Boom." });
  });
});

describe("chatReducer send", () => {
  it("adds the question and an answer that is streaming", () => {
    expect(streaming.activeSend).toBe(1);
    expect(
      streaming.items.map((item) => [item.role, item.content, item.status]),
    ).toEqual([
      ["user", "hello", "done"],
      ["assistant", "", "streaming"],
    ]);
  });

  it("trims the question", () => {
    const state = run([{ type: "send", sendId: 1, text: "  hello  " }], ready);
    expect(state.items[0]?.content).toBe("hello");
  });

  it.each([
    ["an empty message", ""],
    ["a blank message", "   "],
    ["a message over the limit", "x".repeat(MAX_MESSAGE_LENGTH + 1)],
  ])("refuses %s", (_name, text) => {
    expect(run([{ type: "send", sendId: 1, text }], ready)).toBe(ready);
  });

  it("accepts a message of exactly the limit", () => {
    const state = run(
      [{ type: "send", sendId: 1, text: "x".repeat(MAX_MESSAGE_LENGTH) }],
      ready,
    );
    expect(state.activeSend).toBe(1);
  });

  it("refuses a second question while one is streaming", () => {
    expect(run([{ type: "send", sendId: 2, text: "again" }], streaming)).toBe(
      streaming,
    );
  });

  it("refuses a question before the history is loaded", () => {
    expect(run([{ type: "send", sendId: 1, text: "hello" }])).toBe(
      initialChatState,
    );
  });
});

describe("chatReducer streaming", () => {
  it("appends tokens to the answer", () => {
    const state = run(
      [
        { type: "token", sendId: 1, text: "Hel" },
        { type: "token", sendId: 1, text: "lo" },
      ],
      streaming,
    );
    expect(state.items[1]?.content).toBe("Hello");
  });

  it("ignores tokens of a stream that is not the active one", () => {
    expect(run([{ type: "token", sendId: 9, text: "x" }], streaming)).toBe(
      streaming,
    );
  });

  it("finishes the answer with the final message and frees the chat", () => {
    const state = run(
      [
        { type: "token", sendId: 1, text: "Hel" },
        {
          type: "done",
          sendId: 1,
          message: message("m", "assistant", "Hello there"),
        },
      ],
      streaming,
    );
    expect(state.activeSend).toBeNull();
    expect(state.items[1]).toMatchObject({
      content: "Hello there",
      status: "done",
    });
  });

  it("drops the partial text when the answer fails", () => {
    const state = run(
      [
        { type: "token", sendId: 1, text: "Hel" },
        {
          type: "fail",
          sendId: 1,
          message: "Stream failed.",
          unavailable: false,
        },
      ],
      streaming,
    );
    expect(state.activeSend).toBeNull();
    expect(state.items[1]).toMatchObject({
      content: "",
      status: "failed",
      error: "Stream failed.",
    });
  });

  it("ignores a late token after the answer failed", () => {
    expect(run([{ type: "token", sendId: 1, text: "late" }], failed)).toBe(
      failed,
    );
  });

  it("ignores a late done after the answer failed", () => {
    const late: ChatAction = {
      type: "done",
      sendId: 1,
      message: message("m", "assistant", "x"),
    };
    expect(run([late], failed)).toBe(failed);
  });

  it("disables the chat when a failure says the service is unavailable", () => {
    const state = run(
      [{ type: "fail", sendId: 1, message: "No key.", unavailable: true }],
      streaming,
    );
    expect(state).toMatchObject({ phase: "disabled", notice: "No key." });
  });
});

describe("chatReducer retry", () => {
  it("asks the failed answer again as a new stream", () => {
    const state = run(
      [{ type: "retry", sendId: 2, itemId: "assistant-1" }],
      failed,
    );
    expect(state.activeSend).toBe(2);
    expect(state.items[1]).toMatchObject({
      status: "streaming",
      content: "",
      sendId: 2,
    });
    expect(state.items[1]?.error).toBeUndefined();
  });

  it("ignores tokens of the stream that was replaced", () => {
    const retried = run(
      [{ type: "retry", sendId: 2, itemId: "assistant-1" }],
      failed,
    );
    expect(run([{ type: "token", sendId: 1, text: "old" }], retried)).toBe(
      retried,
    );
  });

  it("only retries an answer that failed", () => {
    const finished = run(
      [{ type: "done", sendId: 1, message: message("m", "assistant", "ok") }],
      streaming,
    );
    expect(
      run([{ type: "retry", sendId: 2, itemId: "assistant-1" }], finished),
    ).toBe(finished);
  });

  it("does not retry while another stream is running", () => {
    const busy = run(
      [{ type: "send", sendId: 2, text: "second" }],
      run(
        [{ type: "fail", sendId: 1, message: "x", unavailable: false }],
        streaming,
      ),
    );
    expect(
      run([{ type: "retry", sendId: 3, itemId: "assistant-1" }], busy),
    ).toBe(busy);
  });
});
