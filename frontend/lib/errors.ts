// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Refused-request copy, kept as the source a view can hold: a component applies
 *  the words with `localizedText` when it draws them, so a language change redraws
 *  the message instead of leaving it in the language it was made in. */

import {ApiError} from '../api/wire.ts';
import {catalogText, serverText, type LocalizedText} from './localized-text.ts';
import {errorKindText} from './run-errors.ts';

/** True when `error` is a fetch/stream abort raised by `AbortController.abort()`. */
export function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}

/** The reader's words for an authorization refusal: sign in again (401) or no permission. */
export function authRefusal(status: number): LocalizedText {
    return status === 401
        ? catalogText('Your session has ended. Sign in again to continue.', 'errors.signInRequired')
        : catalogText('You do not have permission to do that.', 'errors.accessDenied');
}

/** What to tell the reader about a refused API request: the words for an `ApiError` (`api/wire.ts`), which
 * carries the server's fields and no copy of its own.

 * A request the cross-origin guard refused says so: it is not a permission
 * matter, and reloading the page is what helps. A refusal whose envelope says
 * `auth` names the action and workspace for operators, so the reader gets its
 * meaning in their language instead. A refusal naming an error kind the UI
 * knows gets that kind's copy. Any other refusal, whatever its type, shows the
 * server's public reason; without one (a body that is not the envelope, such as
 * a proxy's error page) the caller's copy stands.
 */
export function refusalText(error: unknown, fallback: LocalizedText): LocalizedText {
    if (!(error instanceof ApiError)) return fallback;
    if (error.errorKind === 'cross_origin_rejected') {
        return catalogText(
            'This request was blocked because its origin could not be verified. Reload the page and try again.',
            'errors.crossOriginRejected',
        );
    }
    if (error.errorType === 'auth') return authRefusal(error.status);
    return errorKindText(error.errorKind)
        ?? (error.detail ? serverText(error.detail) : fallback);
}
