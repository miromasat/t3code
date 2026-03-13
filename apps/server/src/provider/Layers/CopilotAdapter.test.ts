import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { ThreadId } from "@t3tools/contracts";
import { it } from "@effect/vitest";
import { Effect, Fiber, Layer, Stream } from "effect";

import { CopilotAdapter } from "../Services/CopilotAdapter.ts";
import { CopilotAdapterLive } from "./CopilotAdapter.ts";

function asThreadId(value: string): ThreadId {
  return ThreadId.makeUnsafe(value);
}

function makeFakeCopilotBinary() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "t3-copilot-cli-"));
  const scriptPath = path.join(directory, "copilot");
  fs.writeFileSync(
    scriptPath,
    `#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "version") {
  process.stdout.write("copilot 1.0.0\\n");
  process.exit(0);
}
const promptIndex = args.indexOf("-p");
const prompt = promptIndex >= 0 ? args[promptIndex + 1] ?? "" : "";
if (prompt.includes("interrupt")) {
  setTimeout(() => {
    process.stdout.write("should-not-finish");
    process.exit(0);
  }, 5000);
  return;
}
process.stdout.write("copilot reply");
process.exit(0);
`,
    { encoding: "utf8", mode: 0o755 },
  );
  return {
    scriptPath,
    cleanup: () => {
      fs.rmSync(directory, { recursive: true, force: true });
    },
  };
}

const layer = it.layer(Layer.mergeAll(CopilotAdapterLive));

layer("CopilotAdapterLive", (it) => {
  it.effect("streams a response and persists resume state for recovery", () =>
    Effect.gen(function* () {
      const { scriptPath, cleanup } = makeFakeCopilotBinary();
      yield* Effect.addFinalizer(() => Effect.sync(cleanup));
      const adapter = yield* CopilotAdapter;
      const threadId = asThreadId("thread-copilot");
      const eventFiber = yield* Stream.take(adapter.streamEvents, 8).pipe(Stream.runCollect, Effect.fork);

      const session = yield* adapter.startSession({
        provider: "copilot",
        threadId,
        cwd: process.cwd(),
        model: "gpt-5.2",
        runtimeMode: "full-access",
        providerOptions: {
          copilot: {
            binaryPath: scriptPath,
          },
        },
      });
      assert.equal(session.provider, "copilot");

      const turn = yield* adapter.sendTurn({
        threadId,
        input: "Hello from Copilot",
      });
      assert.equal(turn.threadId, threadId);
      yield* Effect.sleep("50 millis");

      const sessions = yield* adapter.listSessions();
      const persisted = sessions.find((entry) => entry.threadId === threadId);
      assert.equal(persisted?.status, "ready");
      const resumeCursor = persisted?.resumeCursor as
        | { turns?: Array<{ assistantResponse?: string }> }
        | undefined;
      assert.equal(resumeCursor?.turns?.[0]?.assistantResponse, "copilot reply");

      const thread = yield* adapter.readThread(threadId);
      assert.equal(thread.turns.length, 1);

      const events = yield* Fiber.join(eventFiber);
      const eventTypes = events.map((event) => event.type);
      assert.deepEqual(eventTypes.slice(0, 5), [
        "session.started",
        "session.state.changed",
        "session.state.changed",
        "turn.started",
        "item.started",
      ]);
      assert.equal(eventTypes.includes("content.delta"), true);
      assert.equal(eventTypes.includes("turn.completed"), true);
    }),
  );

  it.effect("rejects image attachments for Copilot turns", () =>
    Effect.gen(function* () {
      const { scriptPath, cleanup } = makeFakeCopilotBinary();
      yield* Effect.addFinalizer(() => Effect.sync(cleanup));
      const adapter = yield* CopilotAdapter;
      const threadId = asThreadId("thread-copilot-attachments");
      yield* adapter.startSession({
        provider: "copilot",
        threadId,
        cwd: process.cwd(),
        runtimeMode: "full-access",
        providerOptions: {
          copilot: {
            binaryPath: scriptPath,
          },
        },
      });

      const result = yield* adapter
        .sendTurn({
          threadId,
          input: "Hello",
          attachments: [
            {
              type: "image",
              id: "image-1",
              name: "shot.png",
              mimeType: "image/png",
              sizeBytes: 10,
            },
          ],
        })
        .pipe(Effect.result);

      assert.equal(result._tag, "Failure");
      if (result._tag === "Failure") {
        assert.equal(result.failure._tag, "ProviderAdapterValidationError");
      }
    }),
  );
});
