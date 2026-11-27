// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The owner's Schedules, as the sidebar's Scheduled section and its dialogs share them.
 *
 * Each command answers the one Schedule it changed, so a reply replaces its row and nothing else, and a
 * row that is gone stays gone however late a reply for it arrives. A command says how it ended without
 * deciding what the reader is told: the outcome carries the server's reason, and the Feature that asked
 * words it. The store never draws anything.
 */

import {
  createSchedule,
  deleteSchedule,
  listSchedules,
  type NewSchedule,
  pauseSchedule,
  resumeSchedule,
  runScheduleNow,
  type Schedule,
  type ScheduleChanges,
  type ScheduleFire,
  updateSchedule,
} from '../api/schedules.ts';
import {ApiError} from '../api/wire.ts';
import {LatestRequest} from '../lib/latest-request.ts';
import {Store} from './base.ts';

export type ScheduleListState = 'idle' | 'loading' | 'ready' | 'error';

/** How a command ended. `gone`: the server no longer has the Schedule (it was deleted elsewhere), and the
 *  store has dropped it. `failed`: nothing changed, and `reason` is the server's when it gave one. `busy`:
 *  another command on the same Schedule is still under way, so this one was not sent. */
export type ScheduleOutcome<T> =
  | {kind: 'ok'; value: T}
  | {kind: 'gone'}
  | {kind: 'failed'; reason: string | null}
  | {kind: 'busy'};

/** The wire the store reads and writes; injected by tests. */
export interface ScheduleApi {
  list: typeof listSchedules;
  create: typeof createSchedule;
  update: typeof updateSchedule;
  remove: typeof deleteSchedule;
  pause: typeof pauseSchedule;
  resume: typeof resumeSchedule;
  run: typeof runScheduleNow;
}

const WIRE: ScheduleApi = {
  list: listSchedules,
  create: createSchedule,
  update: updateSchedule,
  remove: deleteSchedule,
  pause: pauseSchedule,
  resume: resumeSchedule,
  run: runScheduleNow,
};

export class ScheduleStore extends Store {
  readonly #api: ScheduleApi;
  readonly #reads = new LatestRequest();
  /** Schedules with a command in flight; each is answered by its own request, so rows never wait for each other. */
  readonly #pending = new Set<string>();
  #schedules: readonly Schedule[] = [];
  #defaultCeiling = 0;
  #listState: ScheduleListState = 'idle';

  constructor(api: ScheduleApi = WIRE) {
    super();
    this.#api = api;
  }

  /** The owner's Schedules, in the order the server lists them; empty until the first read answers. */
  get schedules(): readonly Schedule[] { return this.#schedules; }

  /** The daily token ceiling a new Schedule starts with. */
  get defaultCeiling(): number { return this.#defaultCeiling; }

  get listState(): ScheduleListState { return this.#listState; }

  get(scheduleId: string): Schedule | null {
    return this.#schedules.find((schedule) => schedule.scheduleId === scheduleId) ?? null;
  }

  isPending(scheduleId: string): boolean {
    return this.#pending.has(scheduleId);
  }

  /** Read the list once: a page that is already reading it, or has read it, does not ask again. */
  async load(): Promise<void> {
    if (this.#listState === 'loading' || this.#listState === 'ready') return;
    this.#listState = 'loading';
    this.changed();
    await this.#read(true);
  }

  /** Read the list again after something that can have changed it (a run can pause a Schedule); a failed read
   *  leaves the list as the owner last read it. */
  async refresh(): Promise<void> {
    await this.#read(false);
  }

  async #read(first: boolean): Promise<void> {
    const signal = this.#reads.start();
    try {
      const {schedules, defaultMaxTokensPerDay} = await this.#api.list(signal);
      if (!this.#reads.isCurrent(signal)) return;
      this.#schedules = schedules;
      this.#defaultCeiling = defaultMaxTokensPerDay;
      this.#listState = 'ready';
      this.changed();
    } catch {
      if (!this.#reads.isCurrent(signal)) return;
      if (first || this.#listState !== 'ready') {
        this.#listState = 'error';
        this.changed();
      }
    } finally {
      this.#reads.finish(signal);
    }
  }

  /** Forget what the signed-in owner had, as signing out must. */
  forget(): void {
    this.#reads.cancel();
    this.#pending.clear();
    this.#schedules = [];
    this.#defaultCeiling = 0;
    this.#listState = 'idle';
    this.changed();
  }

  async create(fields: NewSchedule): Promise<ScheduleOutcome<Schedule>> {
    try {
      const saved = await this.#api.create(fields);
      this.#schedules = [...this.#schedules, saved];
      this.changed();
      return {kind: 'ok', value: saved};
    } catch (error) {
      return failure(error);
    }
  }

  update(scheduleId: string, changes: ScheduleChanges): Promise<ScheduleOutcome<Schedule>> {
    return this.#command(scheduleId, async () => this.#replace(await this.#api.update(scheduleId, changes)));
  }

  pause(scheduleId: string): Promise<ScheduleOutcome<Schedule>> {
    return this.#command(scheduleId, async () => this.#replace(await this.#api.pause(scheduleId)));
  }

  resume(scheduleId: string): Promise<ScheduleOutcome<Schedule>> {
    return this.#command(scheduleId, async () => this.#replace(await this.#api.resume(scheduleId)));
  }

  /** Fire once now. The Fire says what became of it; the list is read again after, because a Fire can pause its Schedule. */
  async run(scheduleId: string): Promise<ScheduleOutcome<ScheduleFire>> {
    const outcome = await this.#command(scheduleId, () => this.#api.run(scheduleId));
    if (outcome.kind === 'ok') await this.refresh();
    return outcome;
  }

  remove(scheduleId: string): Promise<ScheduleOutcome<null>> {
    return this.#command(scheduleId, async () => {
      await this.#api.remove(scheduleId);
      this.#drop(scheduleId);
      return null;
    });
  }

  /** The Schedule is gone from the server: drop it from the list, so it cannot be asked of again. */
  drop(scheduleId: string): void {
    this.#drop(scheduleId);
  }

  async #command<T>(scheduleId: string, send: () => Promise<T>): Promise<ScheduleOutcome<T>> {
    if (this.#pending.has(scheduleId)) return {kind: 'busy'};
    this.#pending.add(scheduleId);
    this.changed();
    try {
      return {kind: 'ok', value: await send()};
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) this.#drop(scheduleId);
      return failure(error);
    } finally {
      this.#pending.delete(scheduleId);
      this.changed();
    }
  }

  /** A reply replaces its own row, and never brings back one that has gone. */
  #replace(row: Schedule): Schedule {
    this.#schedules = this.#schedules.map((schedule) => (schedule.scheduleId === row.scheduleId ? row : schedule));
    return row;
  }

  #drop(scheduleId: string): void {
    if (!this.#schedules.some((schedule) => schedule.scheduleId === scheduleId)) return;
    this.#schedules = this.#schedules.filter((schedule) => schedule.scheduleId !== scheduleId);
    this.changed();
  }
}

function failure(error: unknown): ScheduleOutcome<never> {
  if (error instanceof ApiError && error.status === 404) return {kind: 'gone'};
  return {kind: 'failed', reason: error instanceof ApiError ? error.detail : null};
}
