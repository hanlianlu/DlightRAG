// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Reports an uncaught browser error to the server log. */

import {csrfHeaders} from './csrf.ts';

const MAX_DETAIL = 2000;
const MAX_REPORTS = 5;

export function reportUncaughtErrors(): void {
  const reported = new Set<string>();
  const report = (reason: unknown): void => {
    try {
      const text = reason instanceof Error ? [reason.message, reason.stack].join('\n') : String(reason);
      // Cut by characters, as the server counts: a cut inside an astral one leaves half of it, which the server refuses.
      const detail = [...text].slice(0, MAX_DETAIL).join('');
      // One failed Lit update rejects twice with the same error.
      if (reported.has(detail) || reported.size >= MAX_REPORTS) return;
      reported.add(detail);
      fetch('/web/api/client-errors', {
        method: 'POST', headers: csrfHeaders('application/json'), body: JSON.stringify({detail}), keepalive: true,
      }).catch(() => {});
    } catch { /* A report must never raise the error that would report itself. */ }
  };
  window.addEventListener('error', (event) => report(event.error ?? event.message));
  window.addEventListener('unhandledrejection', (event) => report(event.reason));
}
