// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** A one-second redraw for a page whose words move only as time passes: elapsed times and countdowns.
 *
 *  The page says whether it has anything that moves, and the clock exists only while it says so, so a
 *  settled page schedules nothing. The page draws against its own `now` rather than reading the time
 *  in its template, which keeps one answer per frame when a tick and a parent's update meet.
 */

import type {ReactiveController, ReactiveControllerHost} from 'lit';

const TICK_MILLISECONDS = 1000;

/** Redraws its host once a second while the host says it has a moving time to draw. */
export class ClockController implements ReactiveController {
  readonly #host: ReactiveControllerHost;
  readonly #moving: () => boolean;
  #timer: ReturnType<typeof setInterval> | null = null;

  constructor(host: ReactiveControllerHost, moving: () => boolean) {
    this.#host = host;
    this.#moving = moving;
    host.addController(this);
  }

  hostConnected(): void {
    this.#sync();
  }

  /** The host says what it draws at each render, so this is where a page that stopped moving stops. */
  hostUpdate(): void {
    this.#sync();
  }

  hostDisconnected(): void {
    this.#stop();
  }

  #sync(): void {
    if (this.#moving()) {
      if (this.#timer === null) {
        this.#timer = setInterval(() => { this.#host.requestUpdate(); }, TICK_MILLISECONDS);
      }
    } else {
      this.#stop();
    }
  }

  #stop(): void {
    if (this.#timer === null) return;
    clearInterval(this.#timer);
    this.#timer = null;
  }
}
