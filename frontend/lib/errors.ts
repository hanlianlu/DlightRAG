// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** True when `error` is a fetch/stream abort raised by `AbortController.abort()`. */
export function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}

export function answerErrorMessage(
    payload: unknown,
    fallback = 'Service error. Please try again.',
): string {
    const message =
        typeof payload === 'string'
            ? payload
            : payload !== null && typeof payload === 'object' && !Array.isArray(payload)
              ? (payload as {message?: unknown}).message
              : undefined;
    return typeof message === 'string' && message.trim() ? message : fallback;
}
