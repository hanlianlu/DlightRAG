// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type WarningCode =
    | 'HISTORICAL_DOCUMENT_LOAD_FAILED'
    | 'HISTORICAL_DOCUMENT_UNAVAILABLE'
    | 'HISTORICAL_DOCUMENT_PARSE_FAILED';

export interface WarningDocument {
    code: WarningCode;
    filename: string;
    message: string;
}

export interface WarningPayload {
    message: string;
    documents: WarningDocument[];
}

const WARNING_CODES: ReadonlySet<string> = new Set<WarningCode>([
    'HISTORICAL_DOCUMENT_LOAD_FAILED',
    'HISTORICAL_DOCUMENT_UNAVAILABLE',
    'HISTORICAL_DOCUMENT_PARSE_FAILED',
]);
const WARNING_FALLBACK = 'A document could not be used. The answer will continue without it.';
const MAX_WARNING_DOCUMENTS = 16;
const MAX_WARNING_TEXT_LENGTH = 1024;

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

function isBoundedText(value: unknown): value is string {
    return (
        typeof value === 'string' &&
        value.trim().length > 0 &&
        value.length <= MAX_WARNING_TEXT_LENGTH
    );
}

function isWarningDocument(value: unknown): value is WarningDocument {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
    const document = value as Record<string, unknown>;
    return (
        typeof document.code === 'string' &&
        WARNING_CODES.has(document.code) &&
        isBoundedText(document.filename) &&
        isBoundedText(document.message)
    );
}

function isWarningPayload(value: unknown): value is WarningPayload {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
    const payload = value as Record<string, unknown>;
    return (
        isBoundedText(payload.message) &&
        Array.isArray(payload.documents) &&
        payload.documents.length > 0 &&
        payload.documents.length <= MAX_WARNING_DOCUMENTS &&
        payload.documents.every(isWarningDocument)
    );
}

export function answerWarningMessage(payload: unknown): string {
    return isWarningPayload(payload) ? payload.message : WARNING_FALLBACK;
}

export function createAnswerWarningHandler(
    show: (message: string) => void,
): (message: string) => void {
    let shown = false;
    return (message: string): void => {
        if (shown) return;
        shown = true;
        show(message);
    };
}

export function notifyAnswerWarning(
    payload: unknown,
    onWarning: ((message: string) => void) | undefined,
): void {
    if (!onWarning) return;
    onWarning(answerWarningMessage(payload));
}
