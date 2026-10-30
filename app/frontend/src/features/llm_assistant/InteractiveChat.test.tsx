import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  ApiError,
  type ApiClient,
  type ChatEvent,
  type ChatMessage,
} from "@/services/api";
import { createMockClient } from "@/services/api/mockClient";
import { SAMPLE_STUDIES } from "@/services/api/sampleData";
import { InteractiveChat } from "./InteractiveChat";

const completed = SAMPLE_STUDIES.find(
  (study) => study.id === "study-confident",
)!;

function reply(content: string): ChatMessage {
  return {
    id: "reply",
    role: "assistant",
    content,
    created_at: "2026-10-28T10:00:00Z",
  };
}

/** A stream the test feeds by hand, so it can look at the screen in between. */
function manualStream() {
  const queue: (ChatEvent | "end" | Error)[] = [];
  const signals: AbortSignal[] = [];
  let wake: () => void = () => {};
  async function* streamChat(
    _studyId: string,
    _message: string,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatEvent> {
    if (signal) signals.push(signal);
    for (;;) {
      while (queue.length === 0) {
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
      }
      const next = queue.shift()!;
      if (next === "end") return;
      if (next instanceof Error) throw next;
      yield next;
    }
  }
  return {
    streamChat,
    signals,
    push: (item: ChatEvent | "end" | Error) =>
      act(async () => {
        queue.push(item);
        wake();
      }),
  };
}

function clientWith(overrides: Partial<ApiClient>): ApiClient {
  return { ...createMockClient(), ...overrides };
}

async function ask(text: string) {
  const input = await screen.findByLabelText("Ask about this study");
  await waitFor(() => expect(input).toBeEnabled());
  await userEvent.type(input, text);
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
}

describe("InteractiveChat", () => {
  it("shows the earlier conversation when it opens", async () => {
    const client = clientWith({
      listChatMessages: async () => [
        { ...reply("An earlier answer"), id: "1" },
      ],
    });
    render(<InteractiveChat client={client} study={completed} />);
    expect(await screen.findByText("An earlier answer")).toBeInTheDocument();
  });

  it("renders the answer as it arrives and blocks a second question meanwhile", async () => {
    const stream = manualStream();
    render(
      <InteractiveChat
        client={clientWith({ streamChat: stream.streamChat })}
        study={completed}
      />,
    );
    await ask("What is this?");
    expect(screen.getByText("What is this?")).toBeInTheDocument();
    expect(screen.getByLabelText("Answer in progress")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    await stream.push({ type: "token", text: "It is " });
    await stream.push({ type: "token", text: "a sample." });
    expect(screen.getByText(/It is a sample\./)).toBeInTheDocument();

    await stream.push({ type: "done", message: reply("It is a sample.") });
    await waitFor(() =>
      expect(
        screen.queryByLabelText("Answer in progress"),
      ).not.toBeInTheDocument(),
    );
    expect(screen.getByText("It is a sample.")).toBeInTheDocument();
  });

  it("cannot send an empty message and caps the length", async () => {
    render(<InteractiveChat client={createMockClient()} study={completed} />);
    await screen.findByLabelText("Ask about this study");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(screen.getByLabelText("Ask about this study")).toHaveAttribute(
      "maxlength",
      "1000",
    );
    await userEvent.type(screen.getByLabelText("Ask about this study"), "abc");
    expect(screen.getByText("3/1000")).toBeInTheDocument();
  });

  it("sends with Enter and clears the input", async () => {
    const stream = manualStream();
    const streamChat = jest.fn(stream.streamChat);
    render(
      <InteractiveChat client={clientWith({ streamChat })} study={completed} />,
    );
    const input = await screen.findByLabelText("Ask about this study");
    await userEvent.type(input, "hello{Enter}");
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(streamChat.mock.calls[0]?.[1]).toBe("hello");
    expect(input).toHaveValue("");
  });

  it("marks the answer as failed on an error event and drops the partial text", async () => {
    const stream = manualStream();
    render(
      <InteractiveChat
        client={clientWith({ streamChat: stream.streamChat })}
        study={completed}
      />,
    );
    await ask("Question");
    await stream.push({ type: "token", text: "Partial" });
    await stream.push({
      type: "error",
      code: "LLM_UNAVAILABLE",
      message: "Stream failed.",
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Stream failed.",
    );
    expect(screen.queryByText(/Partial/)).not.toBeInTheDocument();
  });

  it("marks a dropped connection as failed", async () => {
    const stream = manualStream();
    render(
      <InteractiveChat
        client={clientWith({ streamChat: stream.streamChat })}
        study={completed}
      />,
    );
    await ask("Question");
    await stream.push({ type: "token", text: "Partial" });
    await stream.push("end");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "connection was interrupted",
    );
    expect(screen.queryByText(/Partial/)).not.toBeInTheDocument();
  });

  it("retries a failed answer with the same question", async () => {
    const first = manualStream();
    const second = manualStream();
    const streams = [first, second];
    const streamChat = jest.fn((...args: Parameters<ApiClient["streamChat"]>) =>
      streams.shift()!.streamChat(...args),
    );
    render(
      <InteractiveChat client={clientWith({ streamChat })} study={completed} />,
    );
    await ask("Question");
    await first.push({ type: "error", code: "X", message: "Stream failed." });
    await userEvent.click(await screen.findByRole("button", { name: "Retry" }));
    expect(streamChat).toHaveBeenCalledTimes(2);
    expect(streamChat.mock.calls[1]?.[1]).toBe("Question");
    await second.push({ type: "done", message: reply("Second try worked.") });
    expect(await screen.findByText("Second try worked.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("disables the chat with an explanation on a 503", async () => {
    const streamChat = jest.fn(async function* (): AsyncGenerator<ChatEvent> {
      throw new ApiError(
        "LLM_UNAVAILABLE",
        "The assistant is not available.",
        503,
      );
    });
    render(
      <InteractiveChat client={clientWith({ streamChat })} study={completed} />,
    );
    await ask("Question");
    expect(
      await screen.findByText("The chat is not available right now"),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Ask about this study")).toBeDisabled();
  });

  it("disables the chat when the history answers 503", async () => {
    const client = clientWith({
      listChatMessages: async () => {
        throw new ApiError(
          "LLM_UNAVAILABLE",
          "The assistant is not available.",
          503,
        );
      },
    });
    render(<InteractiveChat client={client} study={completed} />);
    expect(
      await screen.findByText("The chat is not available right now"),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Ask about this study")).toBeDisabled();
  });

  it("aborts the stream when the page is left", async () => {
    const stream = manualStream();
    const { unmount } = render(
      <InteractiveChat
        client={clientWith({ streamChat: stream.streamChat })}
        study={completed}
      />,
    );
    await ask("Question");
    expect(stream.signals[0]?.aborted).toBe(false);
    unmount();
    expect(stream.signals[0]?.aborted).toBe(true);
  });

  it.each(["study-processing", "study-rejected", "study-failed"])(
    "is hidden for %s",
    (id) => {
      const study = SAMPLE_STUDIES.find((candidate) => candidate.id === id)!;
      render(<InteractiveChat client={createMockClient()} study={study} />);
      expect(
        screen.queryByRole("region", { name: "Chat" }),
      ).not.toBeInTheDocument();
    },
  );
});
