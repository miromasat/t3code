import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";

import {
  EventId,
  RuntimeItemId,
  TurnId,
  type ProviderRuntimeEvent,
  type ProviderSendTurnInput,
  type ProviderSession,
  type ProviderSessionStartInput,
  type ProviderTurnStartResult,
  type ThreadId,
} from "@t3tools/contracts";
import { Effect, Layer, Queue, Stream } from "effect";

import {
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
  type ProviderAdapterError,
} from "../Errors.ts";
import { CopilotAdapter, type CopilotAdapterShape } from "../Services/CopilotAdapter.ts";
import type {
  ProviderThreadSnapshot,
  ProviderThreadTurnSnapshot,
} from "../Services/ProviderAdapter.ts";

const PROVIDER = "copilot" as const;
const DEFAULT_BINARY_PATH = "copilot";
const DEFAULT_COPILOT_MODEL = "gpt-5.2";
const INTERRUPT_KILL_TIMEOUT_MS = 2_000;

interface CopilotResumeTurn {
  readonly turnId: string;
  readonly userPrompt: string;
  readonly assistantResponse: string;
  readonly model?: string;
}

interface CopilotResumeCursor {
  readonly version: 1;
  readonly model?: string;
  readonly turns: ReadonlyArray<CopilotResumeTurn>;
}

interface CopilotSessionContext {
  session: ProviderSession;
  turns: Array<ProviderThreadTurnSnapshot>;
  history: Array<CopilotResumeTurn>;
  child: ChildProcessWithoutNullStreams | null;
  activeTurnId: TurnId | null;
  activeItemId: string | null;
  providerOptions: ProviderSessionStartInput["providerOptions"];
}

function nowIso(): string {
  return new Date().toISOString();
}

function makeEventId(): EventId {
  return EventId.makeUnsafe(randomUUID());
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function parseResumeCursor(value: unknown): CopilotResumeCursor | undefined {
  const record = asRecord(value);
  if (!record || record.version !== 1 || !Array.isArray(record.turns)) {
    return undefined;
  }
  const turns = record.turns
    .map((entry) => {
      const turn = asRecord(entry);
      const turnId = asString(turn?.turnId);
      const userPrompt = asString(turn?.userPrompt);
      const assistantResponse =
        typeof turn?.assistantResponse === "string" ? turn.assistantResponse : "";
      const model = asString(turn?.model);
      if (!turnId || !userPrompt) {
        return undefined;
      }
      return {
        turnId,
        userPrompt,
        assistantResponse,
        ...(model ? { model } : {}),
      } satisfies CopilotResumeTurn;
    })
    .filter((turn): turn is CopilotResumeTurn => turn !== undefined);
  return {
    version: 1,
    ...(asString(record.model) ? { model: asString(record.model) } : {}),
    turns,
  };
}

function serializeResumeCursor(context: CopilotSessionContext): CopilotResumeCursor {
  return {
    version: 1,
    ...(context.session.model ? { model: context.session.model } : {}),
    turns: [...context.history],
  };
}

function toThreadTurns(
  history: ReadonlyArray<CopilotResumeTurn>,
): Array<ProviderThreadTurnSnapshot> {
  return history.map((turn) => ({
    id: TurnId.makeUnsafe(turn.turnId),
    items: [
      {
        role: "user",
        text: turn.userPrompt,
      },
      {
        role: "assistant",
        text: turn.assistantResponse,
      },
    ],
  }));
}

function buildPrompt(input: {
  readonly history: ReadonlyArray<CopilotResumeTurn>;
  readonly userPrompt: string;
  readonly interactionMode?: ProviderSendTurnInput["interactionMode"];
}): string {
  const sections: Array<string> = [
    "You are GitHub Copilot CLI running inside T3 Code.",
    "Continue the coding conversation using the full prior transcript below.",
  ];
  if (input.interactionMode === "plan") {
    sections.push("Important: produce a plan only. Do not edit files or apply changes.");
  }
  if (input.history.length > 0) {
    sections.push(
      [
        "Conversation transcript:",
        ...input.history.flatMap((turn, index) => [
          `Turn ${index + 1} user:`,
          turn.userPrompt,
          `Turn ${index + 1} assistant:`,
          turn.assistantResponse,
        ]),
      ].join("\n"),
    );
  }
  sections.push(["Current user request:", input.userPrompt].join("\n"));
  return sections.join("\n\n");
}

function toValidationError(operation: string, issue: string, cause?: unknown) {
  return new ProviderAdapterValidationError({
    provider: PROVIDER,
    operation,
    issue,
    ...(cause !== undefined ? { cause } : {}),
  });
}

function createRuntimeEvent<TType extends ProviderRuntimeEvent["type"]>(input: {
  readonly type: TType;
  readonly threadId: ThreadId;
  readonly turnId?: TurnId;
  readonly itemId?: string;
  readonly payload: Extract<ProviderRuntimeEvent, { type: TType }>["payload"];
}): Extract<ProviderRuntimeEvent, { type: TType }> {
  return {
    eventId: makeEventId(),
    provider: PROVIDER,
    threadId: input.threadId,
    createdAt: nowIso(),
    ...(input.turnId ? { turnId: input.turnId } : {}),
    ...(input.itemId ? { itemId: RuntimeItemId.makeUnsafe(input.itemId) } : {}),
    type: input.type,
    payload: input.payload,
  } as Extract<ProviderRuntimeEvent, { type: TType }>;
}

const makeCopilotAdapter = () =>
  Effect.gen(function* () {
    const sessions = new Map<ThreadId, CopilotSessionContext>();
    const runtimeEventQueue = yield* Queue.unbounded<ProviderRuntimeEvent>();

    const publish = (event: ProviderRuntimeEvent) =>
      Queue.offer(runtimeEventQueue, event).pipe(Effect.asVoid);

    const getContext = (
      threadId: ThreadId,
    ): Effect.Effect<CopilotSessionContext, ProviderAdapterError> => {
      const context = sessions.get(threadId);
      if (!context) {
        return Effect.fail(
          new ProviderAdapterSessionNotFoundError({
            provider: PROVIDER,
            threadId,
          }),
        );
      }
      return Effect.succeed(context);
    };

    const startSession: CopilotAdapterShape["startSession"] = (input) =>
      Effect.gen(function* () {
        const resolvedCwd = input.cwd?.trim() || process.cwd();
        const resume = parseResumeCursor(input.resumeCursor);
        const now = nowIso();
        const session: ProviderSession = {
          provider: PROVIDER,
          status: "ready",
          runtimeMode: input.runtimeMode,
          cwd: resolvedCwd,
          ...(input.model?.trim()
            ? { model: input.model.trim() }
            : resume?.model
              ? { model: resume.model }
              : { model: DEFAULT_COPILOT_MODEL }),
          threadId: input.threadId,
          ...(resume ? { resumeCursor: resume } : {}),
          createdAt: now,
          updatedAt: now,
        };
        const history = resume?.turns ? [...resume.turns] : [];
        const context: CopilotSessionContext = {
          session,
          turns: toThreadTurns(history),
          history,
          child: null,
          activeTurnId: null,
          activeItemId: null,
          providerOptions: input.providerOptions,
        };
        context.session.resumeCursor = serializeResumeCursor(context);
        sessions.set(input.threadId, context);
        yield* publish(
          createRuntimeEvent({
            type: "session.started",
            threadId: input.threadId,
            payload: {
              message: "GitHub Copilot CLI session started.",
              resume: context.session.resumeCursor,
            },
          }),
        );
        yield* publish(
          createRuntimeEvent({
            type: "session.state.changed",
            threadId: input.threadId,
            payload: {
              state: "ready",
              detail: {
                resumeCursor: context.session.resumeCursor,
              },
            },
          }),
        );
        return context.session;
      });

    const sendTurn: CopilotAdapterShape["sendTurn"] = (input) =>
      Effect.async<ProviderTurnStartResult, ProviderAdapterError>((resume) => {
        const context = sessions.get(input.threadId);
        if (!context) {
          resume(
            Effect.fail(
              new ProviderAdapterSessionNotFoundError({
                provider: PROVIDER,
                threadId: input.threadId,
              }),
            ),
          );
          return;
        }
        if (input.attachments && input.attachments.length > 0) {
          resume(
            Effect.fail(
              toValidationError(
                "CopilotAdapter.sendTurn",
                "GitHub Copilot CLI does not support image attachments in T3 Code yet.",
              ),
            ),
          );
          return;
        }

        const prompt = input.input?.trim();
        if (!prompt) {
          resume(
            Effect.fail(
              toValidationError("CopilotAdapter.sendTurn", "Turn input is required for Copilot."),
            ),
          );
          return;
        }

        const turnId = TurnId.makeUnsafe(randomUUID());
        const itemId = randomUUID();
        const model = input.model?.trim() || context.session.model || DEFAULT_COPILOT_MODEL;
        const binaryPath =
          context.providerOptions?.copilot?.binaryPath?.trim() || DEFAULT_BINARY_PATH;
        const copilotHomePath = context.providerOptions?.copilot?.homePath?.trim();
        const promptText = buildPrompt({
          history: context.history,
          userPrompt: prompt,
          interactionMode: input.interactionMode,
        });
        const args = [
          "-p",
          promptText,
          "-s",
          "--no-ask-user",
          `--add-dir=${context.session.cwd ?? process.cwd()}`,
          "--stream=on",
          ...(context.session.runtimeMode === "full-access" ? ["--allow-all"] : []),
          ...(model ? ["--model", model] : []),
        ];
        const child = spawn(binaryPath, args, {
          cwd: context.session.cwd ?? process.cwd(),
          env: {
            ...process.env,
            ...(copilotHomePath ? { COPILOT_HOME: copilotHomePath } : {}),
          },
          stdio: ["ignore", "pipe", "pipe"],
          shell: process.platform === "win32",
        });

        context.child = child;
        context.activeTurnId = turnId;
        context.activeItemId = itemId;
        context.session = {
          ...context.session,
          status: "running",
          activeTurnId: turnId,
          model,
          updatedAt: nowIso(),
        };

        void Effect.runPromise(
          Effect.all([
            publish(
              createRuntimeEvent({
                type: "session.state.changed",
                threadId: input.threadId,
                turnId,
                payload: {
                  state: "running",
                },
              }),
            ),
            publish(
              createRuntimeEvent({
                type: "turn.started",
                threadId: input.threadId,
                turnId,
                payload: {
                  model,
                },
              }),
            ),
            publish(
              createRuntimeEvent({
                type: "item.started",
                threadId: input.threadId,
                turnId,
                itemId,
                payload: {
                  itemType: "assistant_message",
                  title: "Assistant message",
                },
              }),
            ),
          ]),
        );

        let stdout = "";
        let stderr = "";
        let settled = false;

        child.stdout.on("data", (chunk: Buffer | string) => {
          const text = String(chunk);
          if (!text) return;
          stdout += text;
          void Effect.runPromise(
            publish(
              createRuntimeEvent({
                type: "content.delta",
                threadId: input.threadId,
                turnId,
                itemId,
                payload: {
                  streamKind: "assistant_text",
                  delta: text,
                },
              }),
            ),
          );
        });

        child.stderr.on("data", (chunk: Buffer | string) => {
          stderr += String(chunk);
        });

        child.once("error", (error) => {
          if (settled) return;
          settled = true;
          context.child = null;
          context.activeTurnId = null;
          context.activeItemId = null;
          context.session = {
            ...context.session,
            status: "error",
            activeTurnId: undefined,
            updatedAt: nowIso(),
            lastError: error.message,
          };
          void Effect.runPromise(
            Effect.all([
              publish(
                createRuntimeEvent({
                  type: "runtime.error",
                  threadId: input.threadId,
                  turnId,
                  payload: {
                    message: `Failed to start GitHub Copilot CLI: ${error.message}`,
                  },
                }),
              ),
              publish(
                createRuntimeEvent({
                  type: "turn.completed",
                  threadId: input.threadId,
                  turnId,
                  payload: {
                    state: "failed",
                    errorMessage: error.message,
                  },
                }),
              ),
              publish(
                createRuntimeEvent({
                  type: "session.state.changed",
                  threadId: input.threadId,
                  turnId,
                  payload: {
                    state: "error",
                    reason: error.message,
                  },
                }),
              ),
            ]),
          );
        });

        child.once("close", (code, signal) => {
          if (settled) return;
          settled = true;
          context.child = null;
          context.activeTurnId = null;
          context.activeItemId = null;
          const trimmedStdout = stdout.trimEnd();
          const trimmedStderr = stderr.trim();
          const interrupted = signal === "SIGINT" || signal === "SIGTERM";
          if (code === 0) {
            const turn: CopilotResumeTurn = {
              turnId,
              userPrompt: prompt,
              assistantResponse: trimmedStdout,
              ...(model ? { model } : {}),
            };
            context.history.push(turn);
            context.turns = toThreadTurns(context.history);
            context.session = {
              ...context.session,
              status: "ready",
              activeTurnId: undefined,
              updatedAt: nowIso(),
              model,
              resumeCursor: serializeResumeCursor(context),
              lastError: undefined,
            };
            void Effect.runPromise(
              Effect.all([
                publish(
                  createRuntimeEvent({
                    type: "item.completed",
                    threadId: input.threadId,
                    turnId,
                    itemId,
                    payload: {
                      itemType: "assistant_message",
                      status: "completed",
                      ...(trimmedStdout ? { detail: trimmedStdout.slice(0, 400) } : {}),
                    },
                  }),
                ),
                publish(
                  createRuntimeEvent({
                    type: "turn.completed",
                    threadId: input.threadId,
                    turnId,
                    payload: {
                      state: "completed",
                    },
                  }),
                ),
                publish(
                  createRuntimeEvent({
                    type: "session.state.changed",
                    threadId: input.threadId,
                    turnId,
                    payload: {
                      state: "ready",
                      detail: {
                        resumeCursor: context.session.resumeCursor,
                      },
                    },
                  }),
                ),
                ...(trimmedStderr
                  ? [
                      publish(
                        createRuntimeEvent({
                          type: "runtime.warning",
                          threadId: input.threadId,
                          turnId,
                          payload: {
                            message: trimmedStderr,
                          },
                        }),
                      ),
                    ]
                  : []),
              ]),
            );
          } else {
            const detail =
              trimmedStderr ||
              (trimmedStdout
                ? trimmedStdout
                : `GitHub Copilot CLI exited with code ${code ?? "unknown"}.`);
            context.session = {
              ...context.session,
              status: interrupted ? "ready" : "error",
              activeTurnId: undefined,
              updatedAt: nowIso(),
              lastError: interrupted ? undefined : detail,
            };
            void Effect.runPromise(
              Effect.all([
                publish(
                  interrupted
                    ? createRuntimeEvent({
                        type: "turn.aborted",
                        threadId: input.threadId,
                        turnId,
                        payload: {
                          reason: "Interrupted",
                        },
                      })
                    : createRuntimeEvent({
                        type: "runtime.error",
                        threadId: input.threadId,
                        turnId,
                        payload: {
                          message: detail,
                        },
                      }),
                ),
                publish(
                  createRuntimeEvent({
                    type: "turn.completed",
                    threadId: input.threadId,
                    turnId,
                    payload: {
                      state: interrupted ? "interrupted" : "failed",
                      ...(interrupted ? {} : { errorMessage: detail }),
                    },
                  }),
                ),
                publish(
                  createRuntimeEvent({
                    type: "session.state.changed",
                    threadId: input.threadId,
                    turnId,
                    payload: {
                      state: interrupted ? "ready" : "error",
                      ...(interrupted ? {} : { reason: detail }),
                    },
                  }),
                ),
              ]),
            );
          }
        });

        resume(
          Effect.succeed({
            threadId: input.threadId,
            turnId,
            resumeCursor: serializeResumeCursor(context),
          }),
        );

        return Effect.sync(() => {
          if (!settled && !child.killed) {
            child.kill("SIGTERM");
          }
        });
      });

    const interruptTurn: CopilotAdapterShape["interruptTurn"] = (threadId, turnId) =>
      Effect.gen(function* () {
        const context = yield* getContext(threadId);
        const activeTurnId = context.activeTurnId;
        if (!context.child || !activeTurnId || (turnId && activeTurnId !== turnId)) {
          return;
        }
        const child = context.child;
        yield* Effect.sync(() => {
          child.kill("SIGINT");
          setTimeout(() => {
            if (!child.killed) {
              child.kill("SIGKILL");
            }
          }, INTERRUPT_KILL_TIMEOUT_MS);
        });
      });

    const respondToRequest: CopilotAdapterShape["respondToRequest"] = (
      _threadId,
      _requestId,
      _decision,
    ) =>
      Effect.fail(
        toValidationError(
          "CopilotAdapter.respondToRequest",
          "GitHub Copilot CLI permission approvals are not supported by this adapter.",
        ),
      );

    const respondToUserInput: CopilotAdapterShape["respondToUserInput"] = (
      _threadId,
      _requestId,
      _answers,
    ) =>
      Effect.fail(
        toValidationError(
          "CopilotAdapter.respondToUserInput",
          "GitHub Copilot CLI structured user-input requests are not supported by this adapter.",
        ),
      );

    const stopSession: CopilotAdapterShape["stopSession"] = (threadId) =>
      Effect.gen(function* () {
        const context = yield* getContext(threadId);
        if (context.child && !context.child.killed) {
          yield* Effect.sync(() => {
            context.child?.kill("SIGTERM");
          });
        }
        sessions.delete(threadId);
        yield* publish(
          createRuntimeEvent({
            type: "session.exited",
            threadId,
            payload: {
              reason: "Session stopped",
              recoverable: true,
              exitKind: "requested",
            },
          }),
        );
      });

    const listSessions: CopilotAdapterShape["listSessions"] = () =>
      Effect.sync(() => Array.from(sessions.values(), (context) => context.session));

    const hasSession: CopilotAdapterShape["hasSession"] = (threadId) =>
      Effect.sync(() => sessions.has(threadId));

    const readThread: CopilotAdapterShape["readThread"] = (threadId) =>
      Effect.gen(function* () {
        const context = yield* getContext(threadId);
        return {
          threadId,
          turns: context.turns,
        } satisfies ProviderThreadSnapshot;
      });

    const rollbackThread: CopilotAdapterShape["rollbackThread"] = (threadId, numTurns) =>
      Effect.gen(function* () {
        const context = yield* getContext(threadId);
        if (numTurns <= 0) {
          return {
            threadId,
            turns: context.turns,
          } satisfies ProviderThreadSnapshot;
        }
        context.history.splice(Math.max(0, context.history.length - numTurns), numTurns);
        context.turns = toThreadTurns(context.history);
        context.session = {
          ...context.session,
          resumeCursor: serializeResumeCursor(context),
          updatedAt: nowIso(),
        };
        yield* publish(
          createRuntimeEvent({
            type: "session.state.changed",
            threadId,
            payload: {
              state: "ready",
              detail: {
                resumeCursor: context.session.resumeCursor,
              },
            },
          }),
        );
        return {
          threadId,
          turns: context.turns,
        } satisfies ProviderThreadSnapshot;
      });

    const stopAll: CopilotAdapterShape["stopAll"] = () =>
      Effect.forEach([...sessions.keys()], (threadId) => stopSession(threadId)).pipe(Effect.asVoid);

    return {
      provider: PROVIDER,
      capabilities: {
        sessionModelSwitch: "in-session",
      },
      startSession,
      sendTurn,
      interruptTurn,
      respondToRequest,
      respondToUserInput,
      stopSession,
      listSessions,
      hasSession,
      readThread,
      rollbackThread,
      stopAll,
      streamEvents: Stream.fromQueue(runtimeEventQueue),
    } satisfies CopilotAdapterShape;
  });

export const CopilotAdapterLive = Layer.effect(CopilotAdapter, makeCopilotAdapter());
