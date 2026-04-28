import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  AgentSideConnection,
  SessionNotification,
  PromptRequest,
} from "@agentclientprotocol/sdk";
import type { ClaudeAcpAgent as ClaudeAcpAgentType } from "../acp-agent.js";
import type { SDKMessage, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";

// Issue #603: when a backgrounded task completes between turns, the SDK can
// inject a synthetic SDKUserMessage with origin.kind === "task-notification"
// into the iterator. The previous adapter code rendered it as a
// user_message_chunk and returned the synthetic turn's idle as if it were the
// user's RPC response. This test pins the corrected behavior by scripting that
// exact scenario.

const NULL_USAGE = {
  input_tokens: 0,
  output_tokens: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};

type MockController = {
  push: (m: SDKMessage) => void;
  end: () => void;
  onUserInput: (cb: (msg: SDKUserMessage) => void) => void;
};

let controller: MockController | undefined;

vi.mock("@anthropic-ai/claude-agent-sdk", async () => {
  const actual = await vi.importActual<typeof import("@anthropic-ai/claude-agent-sdk")>(
    "@anthropic-ai/claude-agent-sdk",
  );
  return {
    ...actual,
    query: (args: { prompt: AsyncIterable<SDKUserMessage>; options: unknown }) => {
      const queue: SDKMessage[] = [];
      const resolvers: ((r: IteratorResult<SDKMessage, void>) => void)[] = [];
      let done = false;
      let userInputCb: ((msg: SDKUserMessage) => void) | undefined;

      const push = (m: SDKMessage) => {
        const r = resolvers.shift();
        if (r) {
          r({ value: m, done: false });
        } else {
          queue.push(m);
        }
      };
      const end = () => {
        done = true;
        while (resolvers.length > 0) {
          resolvers.shift()!({ value: undefined, done: true });
        }
      };

      controller = {
        push,
        end,
        onUserInput: (cb) => {
          userInputCb = cb;
        },
      };

      // Drain the input stream the agent passes us so tests can react to the
      // user's promptUuid (set by promptToClaude inside agent.prompt).
      (async () => {
        for await (const userMsg of args.prompt) {
          if (userInputCb) userInputCb(userMsg);
        }
      })();

      const next = (): Promise<IteratorResult<SDKMessage, void>> => {
        if (queue.length > 0) {
          return Promise.resolve({ value: queue.shift()!, done: false });
        }
        if (done) {
          return Promise.resolve({ value: undefined, done: true });
        }
        return new Promise((r) => resolvers.push(r));
      };

      const iter = {
        next,
        return: async () => ({ value: undefined as void, done: true as const }),
        throw: async (e: unknown) => {
          throw e;
        },
        [Symbol.asyncIterator]() {
          return this;
        },
      };

      return Object.assign(iter, {
        initializationResult: async () => ({
          models: [{ value: "claude-test", displayName: "Test", description: "" }],
        }),
        setModel: async () => {},
        supportedCommands: async () => [],
        interrupt: async () => {},
        close: () => {
          end();
        },
      });
    },
  };
});

describe("issue #603: synthetic task-notification user messages", () => {
  let agent: ClaudeAcpAgentType;
  let updates: SessionNotification[];

  beforeEach(async () => {
    updates = [];
    controller = undefined;
    vi.resetModules();
    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    agent = new ClaudeAcpAgent({
      sessionUpdate: async (n: SessionNotification) => {
        updates.push(n);
      },
      requestPermission: async () => ({ outcome: { outcome: "cancelled" } }),
      readTextFile: async () => ({ content: "" }),
      writeTextFile: async () => ({}),
    } as unknown as AgentSideConnection);
  });

  afterEach(async () => {
    await agent.dispose();
  });

  it("drops synthetic task-notification user messages and waits for the real turn's idle", async () => {
    const session = await agent.newSession({ cwd: "/test", mcpServers: [] });
    const sessionId = session.sessionId;
    expect(controller).toBeDefined();
    const c = controller!;

    // Buffer a complete synthetic turn (user[task-notification] + assistant +
    // result + idle) BEFORE the user prompts. Pre-fix, the adapter would
    // render the synthetic user as a user_message_chunk and return the
    // synthetic idle as the user's RPC result.
    c.push({
      type: "user",
      message: { role: "user", content: [{ type: "text", text: "background task completed" }] },
      parent_tool_use_id: null,
      origin: { kind: "task-notification" },
      session_id: sessionId,
      uuid: "synthetic-user-uuid",
    } as unknown as SDKMessage);
    c.push({
      type: "assistant",
      parent_tool_use_id: null,
      message: {
        id: "synthetic-asst",
        type: "message",
        role: "assistant",
        model: "claude-test",
        content: [{ type: "text", text: "synthetic ack" }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { ...NULL_USAGE, input_tokens: 1, output_tokens: 1 },
      },
      uuid: "synthetic-asst-uuid",
      session_id: sessionId,
    } as unknown as SDKMessage);
    c.push({
      type: "result",
      subtype: "success",
      duration_ms: 0,
      duration_api_ms: 0,
      is_error: false,
      num_turns: 1,
      result: "synthetic done",
      session_id: sessionId,
      total_cost_usd: 0,
      usage: { ...NULL_USAGE, input_tokens: 1, output_tokens: 1 },
      modelUsage: {},
      permission_denials: [],
      stop_reason: "end_turn",
      uuid: "synthetic-result-uuid",
    } as unknown as SDKMessage);
    c.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
      uuid: "synthetic-idle-uuid",
      session_id: sessionId,
    } as unknown as SDKMessage);

    // When the agent pushes the user's real message into the SDK input
    // stream, capture its uuid and script the real turn (echo + assistant +
    // result + idle).
    c.onUserInput((userMsg) => {
      const realUuid = userMsg.uuid!;
      c.push({
        type: "user",
        message: userMsg.message,
        parent_tool_use_id: null,
        session_id: sessionId,
        uuid: realUuid,
      } as unknown as SDKMessage);
      c.push({
        type: "assistant",
        parent_tool_use_id: null,
        message: {
          id: "real-asst",
          type: "message",
          role: "assistant",
          model: "claude-test",
          content: [{ type: "text", text: "real answer" }],
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { ...NULL_USAGE, input_tokens: 7, output_tokens: 7 },
        },
        uuid: "real-asst-uuid",
        session_id: sessionId,
      } as unknown as SDKMessage);
      c.push({
        type: "result",
        subtype: "success",
        duration_ms: 0,
        duration_api_ms: 0,
        is_error: false,
        num_turns: 1,
        result: "real done",
        session_id: sessionId,
        total_cost_usd: 0,
        usage: { ...NULL_USAGE, input_tokens: 7, output_tokens: 7 },
        modelUsage: {},
        permission_denials: [],
        stop_reason: "end_turn",
        uuid: "real-result-uuid",
      } as unknown as SDKMessage);
      c.push({
        type: "system",
        subtype: "session_state_changed",
        state: "idle",
        uuid: "real-idle-uuid",
        session_id: sessionId,
      } as unknown as SDKMessage);
    });

    const promptReq: PromptRequest = {
      sessionId,
      prompt: [{ type: "text", text: "what is 2+2?" }],
    };

    const result = await agent.prompt(promptReq);

    // Loop returned end_turn (not error/cancel) once the real idle arrived.
    expect(result.stopReason).toBe("end_turn");

    // Synthetic task-notification user message was NOT rendered as user input.
    const userChunks = updates.filter((u) => u.update.sessionUpdate === "user_message_chunk");
    expect(userChunks).toEqual([]);

    // Sanity: accumulated usage reflects the real turn (input_tokens >= 7).
    expect(result.usage?.inputTokens ?? 0).toBeGreaterThanOrEqual(7);
  });

  it("returns immediately on idle when no synthetic turn was buffered", async () => {
    const session = await agent.newSession({ cwd: "/test", mcpServers: [] });
    const sessionId = session.sessionId;
    const c = controller!;

    // Baseline: a normal turn (no synthetic prelude). Confirms the
    // seenOwnPrompt gating doesn't break the happy path.
    c.onUserInput((userMsg) => {
      const realUuid = userMsg.uuid!;
      c.push({
        type: "user",
        message: userMsg.message,
        parent_tool_use_id: null,
        session_id: sessionId,
        uuid: realUuid,
      } as unknown as SDKMessage);
      c.push({
        type: "result",
        subtype: "success",
        duration_ms: 0,
        duration_api_ms: 0,
        is_error: false,
        num_turns: 1,
        result: "ok",
        session_id: sessionId,
        total_cost_usd: 0,
        usage: { ...NULL_USAGE, input_tokens: 3, output_tokens: 3 },
        modelUsage: {},
        permission_denials: [],
        stop_reason: "end_turn",
        uuid: "result-uuid",
      } as unknown as SDKMessage);
      c.push({
        type: "system",
        subtype: "session_state_changed",
        state: "idle",
        uuid: "idle-uuid",
        session_id: sessionId,
      } as unknown as SDKMessage);
    });

    const result = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "ping" }],
    });

    expect(result.stopReason).toBe("end_turn");
  });
});
