// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { bus, type DlightragEvents } from '../events/bus.ts';

export abstract class Store {
  protected emit<E extends keyof DlightragEvents>(
    event: E,
    ...payload: Parameters<DlightragEvents[E]>
  ): void {
    bus.emit(event, ...payload);
  }
}
