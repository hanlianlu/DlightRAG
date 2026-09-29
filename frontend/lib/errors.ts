// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import {ApiError} from '../api/wire.ts';
import {localizedErrorKind} from './run-errors.ts';

/** True when `error` is a fetch/stream abort raised by `AbortController.abort()`. */
export function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}

/** The reader's words for an authorization refusal: sign in again (401) or no permission. */
export function authRefusalMessage(status: number): string {
    return status === 401
        ? msg('Your session has ended. Sign in again to continue.', {id: 'errors.signInRequired'})
        : msg('You do not have permission to do that.', {id: 'errors.accessDenied'});
}

/** What to tell the reader about a refused API request.

 * A refusal whose envelope says `auth` names the action and workspace for
 * operators, so the reader gets its meaning in their language instead. A
 * refusal naming an error kind the UI knows gets that kind's localized copy.
 * Any other refusal, whatever its type, shows the server's public reason;
 * without one (a body that is not the envelope, such as the cross-origin
 * guard's plain-text 403) the caller's localized copy stands.
 */
export function apiErrorMessage(error: unknown, fallback: string): string {
    if (!(error instanceof ApiError)) return fallback;
    if (error.errorType === 'auth') return authRefusalMessage(error.status);
    return localizedErrorKind(error.errorKind) ?? error.detail ?? fallback;
}
