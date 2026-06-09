// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { bus, type DlightragEvents } from '../events/bus';

export abstract class Store {
  protected emit<E extends keyof DlightragEvents>(
    event: E,
    payload: DlightragEvents[E],
  ): void {
    (bus.emit as (event: string, payload: unknown) => void)(event as string, payload);
  }
}
