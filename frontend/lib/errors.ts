// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import {ApiError} from '../api/wire.ts';

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

/** The server's public reason for a refused API request, else the caller's localized copy. */
export function apiErrorMessage(error: unknown, fallback: string): string {
    return error instanceof ApiError && error.detail ? error.detail : fallback;
}
