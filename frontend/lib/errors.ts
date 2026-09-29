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

/** The reader's words for an authorization refusal: sign in again (401) or no permission. */
export function authRefusalMessage(status: number): string {
    return status === 401
        ? msg('Your session has ended. Sign in again to continue.', {id: 'errors.signInRequired'})
        : msg('You do not have permission to do that.', {id: 'errors.accessDenied'});
}

/** What to tell the reader about a refused API request.

 * An authorization refusal names the action and workspace for operators, so
 * the reader gets its meaning in their language instead. Any other refusal
 * shows the server's public reason (a validation, conflict, or availability
 * remedy); without one the caller's localized copy stands.
 */
export function apiErrorMessage(error: unknown, fallback: string): string {
    if (!(error instanceof ApiError)) return fallback;
    if (error.errorType === 'auth') return authRefusalMessage(error.status);
    return error.detail ?? fallback;
}
