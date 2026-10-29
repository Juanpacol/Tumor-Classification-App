import type { ChatMessage } from "@/services/api";

export const MAX_MESSAGE_LENGTH = 1000;

export type ChatItemStatus = "done" | "streaming" | "failed";

export interface ChatItem {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: ChatItemStatus;
  /** Why an assistant answer failed. */
  error?: string;
  /** The stream that is filling an assistant answer. */
  sendId?: number;
  /** The question an assistant answer replies to, so it can be asked again. */
  prompt?: string;
}

export interface ChatState {
  phase: "loading" | "ready" | "disabled";
  items: ChatItem[];
  /** The stream in progress, if any; only one runs at a time. */
  activeSend: number | null;
  /** Why the history failed or the chat is disabled. */
  notice: string | null;
}

export type ChatAction =
  | { type: "historyLoaded"; messages: ChatMessage[] }
  | { type: "historyFailed"; message: string; unavailable: boolean }
  | { type: "send"; sendId: number; text: string }
  | { type: "retry"; sendId: number; itemId: string }
  | { type: "token"; sendId: number; text: string }
  | { type: "done"; sendId: number; message: ChatMessage }
  | { type: "fail"; sendId: number; message: string; unavailable: boolean };

export const initialChatState: ChatState = {
  phase: "loading",
  items: [],
  activeSend: null,
  notice: null,
};

export function isSendable(text: string): boolean {
  const length = text.trim().length;
  return length > 0 && length <= MAX_MESSAGE_LENGTH;
}

function canStart(state: ChatState): boolean {
  return state.phase === "ready" && state.activeSend === null;
}

function updateStreaming(
  state: ChatState,
  sendId: number,
  change: (item: ChatItem) => ChatItem,
): ChatItem[] {
  return state.items.map((item) =>
    item.sendId === sendId && item.status === "streaming" ? change(item) : item,
  );
}

/**
 * Every transition of the conversation. Actions of a stream that is no longer
 * the active one (cancelled, replaced or finished) change nothing, and an
 * answer that did not finish never keeps its partial text.
 */
export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "historyLoaded":
      if (state.phase !== "loading") return state;
      return {
        ...state,
        phase: "ready",
        items: action.messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: message.content,
          status: "done",
        })),
      };
    case "historyFailed":
      if (state.phase !== "loading") return state;
      return {
        ...state,
        phase: action.unavailable ? "disabled" : "ready",
        notice: action.message,
      };
    case "send": {
      if (!canStart(state) || !isSendable(action.text)) return state;
      const text = action.text.trim();
      return {
        ...state,
        activeSend: action.sendId,
        items: [
          ...state.items,
          {
            id: `user-${action.sendId}`,
            role: "user",
            content: text,
            status: "done",
          },
          {
            id: `assistant-${action.sendId}`,
            role: "assistant",
            content: "",
            status: "streaming",
            sendId: action.sendId,
            prompt: text,
          },
        ],
      };
    }
    case "retry": {
      const failed = state.items.find((item) => item.id === action.itemId);
      if (!canStart(state) || failed?.status !== "failed") return state;
      return {
        ...state,
        activeSend: action.sendId,
        items: state.items.map((item) =>
          item === failed
            ? {
                ...item,
                content: "",
                status: "streaming",
                error: undefined,
                sendId: action.sendId,
              }
            : item,
        ),
      };
    }
    case "token":
      if (state.activeSend !== action.sendId) return state;
      return {
        ...state,
        items: updateStreaming(state, action.sendId, (item) => ({
          ...item,
          content: item.content + action.text,
        })),
      };
    case "done":
      if (state.activeSend !== action.sendId) return state;
      return {
        ...state,
        activeSend: null,
        items: updateStreaming(state, action.sendId, (item) => ({
          ...item,
          content: action.message.content,
          status: "done",
        })),
      };
    case "fail":
      if (state.activeSend !== action.sendId) return state;
      return {
        ...state,
        phase: action.unavailable ? "disabled" : state.phase,
        notice: action.unavailable ? action.message : state.notice,
        activeSend: null,
        items: updateStreaming(state, action.sendId, (item) => ({
          ...item,
          content: "",
          status: "failed",
          error: action.message,
        })),
      };
  }
}
