// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';

/** True when `error` is a fetch/stream abort raised by `AbortController.abort()`. */
export function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}

export function answerErrorMessage(
    payload: unknown,
    fallback: string = msg('Service error. Please try again.', {id: 'errors.service'}),
): string {
    const message =
        payload !== null && typeof payload === 'object' && !Array.isArray(payload)
            ? (payload as {message?: unknown}).message
            : undefined;
    return typeof message === 'string' && message.trim() ? message : fallback;
}
