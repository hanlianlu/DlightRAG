// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Follows one accepted durable Corpus Mutation Run until it settles.

 * Files (upload, delete, reset, workspace deletion) and failed-document
 * recovery (retry) accept Corpus Mutations and show one lifecycle: read the
 * Run's canonical status URL while it is queued or running, stop and offer an
 * explicit same-Run resume while it waits for operator repair, and hand the
 * terminal Run back to its owner. The owner keeps rendering, copy, and
 * toasts; the tracker owns timers, aborts, and stale-result rejection.
 */

import {
  corpusRunActive,
  corpusRunStatusRefused,
  getCorpusRunStatus,
  resumeCorpusRun,
  type WebCorpusRunReceipt,
  type WebCorpusRunStatus,
} from '../api/corpus-runs.ts';
import {isAbortError} from './errors.ts';

const POLL_INTERVAL_MS = 2000;

/** The latest status of a tracked Run plus what only its acceptance receipt knew. */
export type TrackedCorpusRun = WebCorpusRunStatus
  & Pick<WebCorpusRunReceipt, 'workspace' | 'fileCount'>;

/** How a resume request ended; `stale` means newer tracking overtook it. */
export type CorpusRunResumeOutcome = 'accepted' | 'failed' | 'stale';

type StatusRequest = (url: string, signal: AbortSignal) => Promise<WebCorpusRunStatus>;

export interface CorpusRunTrackerOptions {
  /** The tracked Run, or whether a resume is in flight, changed. */
  onChange: () => void;
  /** The tracked Run reached a terminal status. */
  onSettled: (run: TrackedCorpusRun) => void;
  /** Its status read was refused (corpusRunStatusRefused): the Run is gone or no longer visible. */
  onLost: (error: unknown) => void;
  getStatus?: StatusRequest;
  resume?: StatusRequest;
  pollIntervalMs?: number;
}

/** A Run parked until an operator repairs its outcome and resumes it. */
export function waitingForRepair(run: TrackedCorpusRun | null | undefined): boolean {
  return run?.phase === 'waiting_for_repair';
}

export class CorpusRunTracker {
  readonly #options: CorpusRunTrackerOptions;
  readonly #getStatus: StatusRequest;
  readonly #resumeRun: StatusRequest;
  readonly #interval: number;
  #run: TrackedCorpusRun | null = null;
  #read: AbortController | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #resume: AbortController | null = null;

  constructor(options: CorpusRunTrackerOptions) {
    this.#options = options;
    this.#getStatus = options.getStatus ?? getCorpusRunStatus;
    this.#resumeRun = options.resume ?? resumeCorpusRun;
    this.#interval = options.pollIntervalMs ?? POLL_INTERVAL_MS;
  }

  /** The last known state of the tracked Run, kept after it settles. */
  get run(): TrackedCorpusRun | null {
    return this.#run;
  }

  /** Queued or running, a repair wait included. */
  get active(): boolean {
    return corpusRunActive(this.#run);
  }

  get waitingForRepair(): boolean {
    return waitingForRepair(this.#run);
  }

  get resuming(): boolean {
    return this.#resume !== null;
  }

  /** Track a newly accepted Run in place of any earlier one and read it at once. */
  follow(receipt: WebCorpusRunReceipt): void {
    this.#stop();
    this.#run = {
      ...receipt,
      result: null,
      phase: null,
      errorKind: null,
      errorMessage: null,
      repairReason: null,
      repairRemedy: null,
    };
    this.#options.onChange();
    void this.#readStatus();
  }

  /** Read again now when the Run is still moving and nothing is already scheduled. */
  wake(): void {
    if (this.#read || this.#timer !== null || this.#resume) return;
    if (!this.active || this.waitingForRepair) return;
    void this.#readStatus();
  }

  /** Stop reading while the owner is hidden; the Run stays tracked for `wake`. */
  pause(): void {
    const resuming = this.#resume !== null;
    this.#stop();
    if (resuming) this.#options.onChange();
  }

  /** Forget the tracked Run, e.g. when its workspace is no longer shown. */
  clear(): void {
    const changed = this.#run !== null || this.#resume !== null;
    this.#stop();
    this.#run = null;
    if (changed) this.#options.onChange();
  }

  /** Ask the server to resume a Run parked for repair; reading continues from its answer. */
  async resume(): Promise<CorpusRunResumeOutcome> {
    const run = this.#run;
    if (!run || !waitingForRepair(run) || this.#resume) return 'stale';
    const controller = new AbortController();
    this.#resume = controller;
    this.#options.onChange();
    try {
      const status = await this.#resumeRun(run.resumeUrl, controller.signal);
      if (this.#resume !== controller) return 'stale';
      this.#resume = null;
      this.#run = {...status, workspace: run.workspace, fileCount: run.fileCount};
      this.#options.onChange();
      // The next read reports the resumed Run, so the owner's acceptance comes first.
      if (!waitingForRepair(this.#run)) this.#schedule();
      return 'accepted';
    } catch (error) {
      if (this.#resume !== controller) return 'stale';
      this.#resume = null;
      this.#options.onChange();
      return isAbortError(error) ? 'stale' : 'failed';
    }
  }

  async #readStatus(): Promise<void> {
    const run = this.#run;
    if (!run) return;
    const controller = new AbortController();
    this.#read = controller;
    try {
      const status = await this.#getStatus(run.statusUrl, controller.signal);
      if (this.#read !== controller) return;
      this.#read = null;
      const next = {...status, workspace: run.workspace, fileCount: run.fileCount};
      this.#run = next;
      this.#options.onChange();
      if (waitingForRepair(next)) return;
      if (corpusRunActive(next)) this.#schedule();
      else this.#options.onSettled(next);
    } catch (error) {
      if (this.#read !== controller) return;
      this.#read = null;
      if (isAbortError(error)) return;
      if (!corpusRunStatusRefused(error)) {
        this.#schedule();
        return;
      }
      this.#run = null;
      this.#options.onChange();
      this.#options.onLost(error);
    }
  }

  #schedule(): void {
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.#readStatus();
    }, this.#interval);
  }

  #stop(): void {
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = null;
    this.#read?.abort();
    this.#read = null;
    this.#resume?.abort();
    this.#resume = null;
  }
}
